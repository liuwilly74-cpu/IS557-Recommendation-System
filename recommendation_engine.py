# -*- coding: utf-8 -*-
"""
留学选校推荐 — 相似度计算引擎（与 Streamlit 前端解耦）。

六维打分体系（权重可调）：

1. undergrad_major — 本科专业大类完全匹配
2. gpa            — 归一化绝对差线性衰减
3. language       — IELTS/TOEFL 等效托福分差
4. experience_tags — Jaccard 集合相似度
5. experience_embed — 用户经历文本 vs 历史行条目向量余弦相似度（来自 SQLite）
6. target_major   — 目标录取专业大类完全匹配

embedding 数据来源：
- experience_item_embeddings.sqlite（text_hash, model -> 向量）
- experience_item_index.csv（row_id -> text_hash 映射）

用户输入的经历文本会实时调用 DashScope API 做一次 embedding，再与库中向量做余弦。
"""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# .env 支持（密钥仅从环境变量读取）
# ---------------------------------------------------------------------------
from env_loader import load_project_env

load_project_env()

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 全局可调权重（六维，总和 = 1.0；运行时会对有效维再归一）
# ---------------------------------------------------------------------------
DEFAULT_FEATURE_WEIGHTS: dict[str, float] = {
    "undergrad_major": 0.15,
    "gpa": 0.20,
    "language": 0.10,
    "experience_tags": 0.15,
    "experience_embed": 0.30,  # 经历语义向量匹配
    "target_major": 0.10,
}

# ---------------------------------------------------------------------------
# 文件路径
# ---------------------------------------------------------------------------
DEFAULT_CSV_PATH = BASE_DIR / "offers_processed.csv"
DEFAULT_DB_PATH = BASE_DIR / "experience_item_embeddings.sqlite"
DEFAULT_INDEX_PATH = BASE_DIR / "experience_item_index.csv"

DEFAULT_EMBED_MODEL = "text-embedding-v4"
DEFAULT_EMBED_DIM = 1024

# ---------------------------------------------------------------------------
# CSV 列名
# ---------------------------------------------------------------------------
COL_SCHOOL = "录取学校"
COL_PROG = "录取专业"
COL_TARGET_CAT_ZH = "录取专业大类"
COL_UNDER_CAT_ZH = "本科专业大类"
COL_GPA = "GPA"
COL_IELTS = "IELTS"
COL_TOEFL = "TOEFL"
COL_TAGS = "experience_tags"

# ======================== 基础工具 ========================

def _to_float(x: Any) -> float | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in ("nan", "none"):
            return None
        try:
            return float(s)
        except ValueError:
            return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def normalize_gpa(gpa: Any) -> float | None:
    """GPA -> [0,1]；<= 4.5 视为 4 分制，否则百分制。"""
    g = _to_float(gpa)
    if g is None:
        return None
    if g <= 4.5:
        return float(np.clip(g / 4.0, 0, 1))
    return float(np.clip(g / 100.0, 0, 1))


_IELTS_PTS = np.array([4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0])
_TOEFL_PTS = np.array([31, 45, 59, 78, 87, 96, 103, 109, 114, 118, 120], dtype=float)


def _ielts_to_toefl(i: float) -> float:
    return float(np.interp(np.clip(i, 4.0, 9.0), _IELTS_PTS, _TOEFL_PTS))


def language_to_toefl_equiv(ielts: Any, toefl: Any) -> float | None:
    iel = _to_float(ielts)
    tof = _to_float(toefl)
    cand: list[float] = []
    if iel is not None:
        cand.append(_ielts_to_toefl(iel))
    if tof is not None:
        cand.append(tof)
    return max(cand) if cand else None


def parse_tag_set(cell: Any) -> set[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return set()
    s = str(cell).strip()
    if not s:
        return set()
    return {p.strip() for p in re.split(r"[;；,，|]", s) if p.strip()}

# ======================== 专业大类相似度矩阵 ========================
# 24 个专业大类之间的相似度（0~100，对称矩阵）。
# 设计原则：
#   - 同一大类: 100
#   - 同子领域 (如 CS / IT / DS): 75~90
#   - 同大学科门类 (如 CS / 数学/统计): 50~70
#   - 邻接 / 工具关系 (如 数学 / 金融): 30~55
#   - 不同领域、但仍有交叉 (如 法学 / 商科): 20~35
#   - 完全不相关 (如 CS / 法学): DEFAULT_MAJOR_SIM
#   - "其他/待人工细分": 中性 30
# 未在矩阵中显式列出的两两组合，统一使用 DEFAULT_MAJOR_SIM。
DEFAULT_MAJOR_SIM: float = 10.0
OTHER_CAT_NAME: str = "其他/待人工细分"
OTHER_NEUTRAL_SIM: float = 30.0

_RAW_MAJOR_SIMILARITY: dict[tuple[str, str], float] = {
    # ---- 计算/数据/电子 ----
    ("计算机科学", "信息技术"): 90,
    ("计算机科学", "数据科学"): 85,
    ("计算机科学", "电子电气与通信"): 70,
    ("计算机科学", "数学与统计"): 65,
    ("计算机科学", "理学基础学科"): 35,
    ("计算机科学", "传媒设计与艺术"): 30,
    ("信息技术", "数据科学"): 75,
    ("信息技术", "电子电气与通信"): 70,
    ("信息技术", "传媒设计与艺术"): 30,
    ("数据科学", "数学与统计"): 80,
    ("数据科学", "电子电气与通信"): 60,
    ("数据科学", "理学基础学科"): 50,
    ("数据科学", "金融"): 50,
    ("数据科学", "经济与贸易"): 40,
    ("数据科学", "商科"): 30,
    ("数据科学", "生物医学与健康"): 35,

    # ---- 工程 ----
    ("电子电气与通信", "机械与先进制造"): 55,
    ("电子电气与通信", "材料"): 50,
    ("电子电气与通信", "土木建筑与环境"): 35,
    ("机械与先进制造", "材料"): 70,
    ("机械与先进制造", "土木建筑与环境"): 55,
    ("机械与先进制造", "理学基础学科"): 35,
    ("土木建筑与环境", "材料"): 50,
    ("土木建筑与环境", "传媒设计与艺术"): 35,   # 建筑 ↔ 设计
    ("土木建筑与环境", "公共政策"): 30,         # 城市规划/环境政策

    # ---- 自然科学 ----
    ("材料", "化学"): 70,
    ("材料", "理学基础学科"): 55,
    ("化学", "理学基础学科"): 70,
    ("化学", "生物医学与健康"): 70,
    ("化学", "农林与食品"): 50,
    ("生物医学与健康", "理学基础学科"): 60,
    ("生物医学与健康", "农林与食品"): 60,
    ("生物医学与健康", "公共政策"): 25,
    ("农林与食品", "理学基础学科"): 40,

    # ---- 数学/统计/基础科学 ----
    ("数学与统计", "理学基础学科"): 75,
    ("数学与统计", "金融"): 55,
    ("数学与统计", "经济与贸易"): 45,
    ("数学与统计", "会计"): 35,

    # ---- 商科簇 ----
    ("商科", "管理"): 85,
    ("商科", "会计"): 75,
    ("商科", "金融"): 80,
    ("商科", "经济与贸易"): 75,
    ("商科", "公共政策"): 35,
    ("会计", "金融"): 80,
    ("会计", "管理"): 60,
    ("会计", "经济与贸易"): 55,
    ("金融", "经济与贸易"): 85,
    ("金融", "管理"): 55,
    ("金融", "公共政策"): 40,
    ("管理", "经济与贸易"): 60,
    ("管理", "公共政策"): 50,
    ("经济与贸易", "公共政策"): 55,

    # ---- 社科/人文簇 ----
    ("社会科学与人文", "教育"): 65,
    ("社会科学与人文", "语言文学"): 60,
    ("社会科学与人文", "法学"): 60,
    ("社会科学与人文", "公共政策"): 75,
    ("社会科学与人文", "传媒设计与艺术"): 50,
    ("社会科学与人文", "经济与贸易"): 45,
    ("教育", "语言文学"): 55,
    ("教育", "传媒设计与艺术"): 40,
    ("教育", "公共政策"): 50,
    ("语言文学", "传媒设计与艺术"): 60,
    ("法学", "公共政策"): 70,
    ("法学", "经济与贸易"): 35,
    ("法学", "商科"): 30,
    ("法学", "管理"): 25,
}

# 把原始单向 dict 规范化为 (sorted-tuple) -> score，方便对称查询。
MAJOR_SIMILARITY: dict[tuple[str, str], float] = {
    tuple(sorted(k)): float(v) for k, v in _RAW_MAJOR_SIMILARITY.items()
}


def major_similarity(a: str | None, b: str | None) -> float | None:
    """
    返回两个专业大类的相似度（0~100）。
    - 任一为空 -> None
    - 完全相同 -> 100
    - 涉及 "其他/待人工细分" -> 中性 30
    - 矩阵中存在 -> 取矩阵值
    - 否则 -> DEFAULT_MAJOR_SIM
    """
    if not (a and b):
        return None
    a, b = str(a).strip(), str(b).strip()
    if not a or not b:
        return None
    if a == b:
        return 100.0
    if a == OTHER_CAT_NAME or b == OTHER_CAT_NAME:
        return OTHER_NEUTRAL_SIM
    return MAJOR_SIMILARITY.get(tuple(sorted([a, b])), DEFAULT_MAJOR_SIM)

# ======================== 子分数 ========================

def categorical_match(a: str | None, b: str | None) -> float | None:
    """保留：完全相等才得满分（仅作为兜底/对照基线，主流程已改用 major_similarity）。"""
    if not (a and b):
        return None
    a, b = str(a).strip(), str(b).strip()
    if not a or not b:
        return None
    return 100.0 if a == b else 0.0


def gpa_subscore(u: Any, r: Any, span: float = 0.45) -> float | None:
    un = normalize_gpa(u)
    rn = normalize_gpa(r)
    if un is None or rn is None:
        return None
    return float(100.0 * max(0.0, 1.0 - abs(un - rn) / span))


def language_subscore(u_i: Any, u_t: Any, r_i: Any, r_t: Any, band: float = 36.0) -> float | None:
    u = language_to_toefl_equiv(u_i, u_t)
    v = language_to_toefl_equiv(r_i, r_t)
    if u is None or v is None:
        return None
    return float(100.0 * max(0.0, 1.0 - abs(u - v) / band))


def jaccard_score(a: set[str], b: set[str]) -> float | None:
    if not a and not b:
        return None
    if not a or not b:
        return 0.0
    return 100.0 * len(a & b) / len(a | b) if (a | b) else 0.0

# ======================== Embedding 相关 ========================

class EmbeddingStore:
    """
    从 SQLite + index 构建 row_id -> 平均向量 映射；
    同时提供用户经历文本的实时 embedding 能力。
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        index_path: Path = DEFAULT_INDEX_PATH,
        model: str = DEFAULT_EMBED_MODEL,
        dim: int = DEFAULT_EMBED_DIM,
    ):
        self.model = model
        self.dim = dim
        self._row_vecs: dict[int, np.ndarray] = {}
        self._ready = False
        self._client = None  # lazy

        if not db_path.is_file() or not index_path.is_file():
            return

        row_hashes: dict[int, list[str]] = {}
        with index_path.open("r", encoding="utf-8-sig", newline="") as f:
            for rec in csv.DictReader(f):
                rid = int(rec["row_id"])
                row_hashes.setdefault(rid, []).append(rec["text_hash"])

        all_hashes = list({h for hs in row_hashes.values() for h in hs})
        if not all_hashes:
            return

        conn = sqlite3.connect(str(db_path))
        try:
            hash_vec: dict[str, np.ndarray] = {}
            batch = 500
            for i in range(0, len(all_hashes), batch):
                chunk = all_hashes[i:i + batch]
                q = ",".join(["?"] * len(chunk))
                cur = conn.execute(
                    f"SELECT text_hash, embedding_json FROM embeddings_v2 "
                    f"WHERE model = ? AND text_hash IN ({q})",
                    [model] + chunk,
                )
                for h, ej in cur.fetchall():
                    try:
                        hash_vec[h] = np.array(json.loads(ej), dtype=np.float32)
                    except Exception:
                        continue
        finally:
            conn.close()

        for rid, hs in row_hashes.items():
            vecs = [hash_vec[h] for h in hs if h in hash_vec]
            if vecs:
                self._row_vecs[rid] = np.mean(vecs, axis=0)

        if self._row_vecs:
            self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    def get_row_vec(self, row_id: int) -> np.ndarray | None:
        """row_id 为 CSV 数据行序号（1-based，不含表头）。"""
        return self._row_vecs.get(row_id)

    def embed_user_text(self, text: str) -> np.ndarray | None:
        """调用 DashScope API 实时将用户经历文本向量化。"""
        text = (text or "").strip()
        if not text:
            return None

        api_key = (os.environ.get("DASHSCOPE_API_KEY") or "").strip().strip('"').strip("'")
        base_url = (os.environ.get("DASHSCOPE_BASE_URL")
                    or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
        if not api_key:
            return None

        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=base_url)

        try:
            resp = self._client.embeddings.create(
                model=self.model,
                input=[text],
                dimensions=self.dim,
                encoding_format="float",
            )
            return np.array(resp.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"[EmbeddingStore] 用户文本 embed 失败: {e}")
            return None

    def build_matrix(self, n_rows: int) -> np.ndarray:
        """
        构建 (n_rows, dim) 矩阵。row_id = i+1 → 第 i 行。
        没有向量的行填零向量。
        """
        mat = np.zeros((n_rows, self.dim), dtype=np.float32)
        for rid, v in self._row_vecs.items():
            idx = rid - 1
            if 0 <= idx < n_rows:
                mat[idx] = v
        return mat

    def row_has_vec_mask(self, n_rows: int) -> np.ndarray:
        """布尔数组：第 i 行是否有有效向量。"""
        mask = np.zeros(n_rows, dtype=bool)
        for rid in self._row_vecs:
            idx = rid - 1
            if 0 <= idx < n_rows:
                mask[idx] = True
        return mask


def cosine_similarity_1_vs_n(
    user_vec: np.ndarray, mat: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """
    user_vec: (dim,)   mat: (n, dim)   mask: (n,)
    返回 (n,) 余弦相似度 * 100；无向量行为 NaN。
    """
    n = mat.shape[0]
    scores = np.full(n, np.nan, dtype=np.float64)
    if not mask.any():
        return scores

    u_norm = np.linalg.norm(user_vec)
    if u_norm < 1e-12:
        return scores

    sub = mat[mask]  # (m, dim)
    dots = sub @ user_vec  # (m,)
    norms = np.linalg.norm(sub, axis=1)
    norms = np.maximum(norms, 1e-12)
    cos = dots / (norms * u_norm)
    # 映射到 0~100（余弦值在 [-1,1]，经历文本基本不会是负，夹到 [0,1]）
    cos = np.clip(cos, 0.0, 1.0) * 100.0
    scores[mask] = cos
    return scores

# ======================== 数据加载 ========================

def load_offers_csv(path: Path | None = None, nrows: int | None = None) -> pd.DataFrame:
    p = Path(path) if path else DEFAULT_CSV_PATH
    if not p.is_file():
        raise FileNotFoundError(f"未找到数据文件: {p}")
    df = pd.read_csv(p, encoding="utf-8-sig", nrows=nrows, low_memory=False)
    for c in (COL_GPA, COL_IELTS, COL_TOEFL):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in (COL_TARGET_CAT_ZH, COL_UNDER_CAT_ZH, COL_SCHOOL, COL_PROG, COL_TAGS):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).replace({"nan": ""})
    return df


def category_options(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    t = sorted({x for x in df[COL_TARGET_CAT_ZH].unique() if str(x).strip()}) if COL_TARGET_CAT_ZH in df.columns else []
    u = sorted({x for x in df[COL_UNDER_CAT_ZH].unique() if str(x).strip()}) if COL_UNDER_CAT_ZH in df.columns else []
    return t, u


def tag_options_from_data(df: pd.DataFrame, max_tags: int = 200) -> list[str]:
    if COL_TAGS not in df.columns:
        return []
    freq: dict[str, int] = {}
    for cell in df[COL_TAGS].dropna().head(min(len(df), 80000)):
        for t in parse_tag_set(cell):
            freq[t] = freq.get(t, 0) + 1
    top = sorted(freq.keys(), key=lambda x: -freq[x])[:max_tags]
    return sorted(top)

# ======================== 推荐主函数（向量化加速）========================

def recommend_top_k(
    df: pd.DataFrame,
    user: dict[str, Any],
    k: int = 15,
    weights: dict[str, float] | None = None,
    emb_store: EmbeddingStore | None = None,
    user_vec: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    对全表六维打分，返回前 k 行。

    尽量用 numpy 向量化而非逐行 iterrows，以支撑 13 万行级数据。
    """
    w = dict(DEFAULT_FEATURE_WEIGHTS if weights is None else weights)
    n = len(df)

    # --- 1. target_major (graded similarity from MAJOR_SIMILARITY) ---
    u_tgt = (user.get("target_major_cat") or "").strip()
    s_tgt = np.full(n, np.nan)
    if u_tgt and COL_TARGET_CAT_ZH in df.columns:
        col = df[COL_TARGET_CAT_ZH].astype(str).str.strip()
        # 全表只有少量唯一值（<=24 类），先做映射再 .map，O(n) 但非常快
        sim_map = {v: major_similarity(u_tgt, v) for v in col.unique()}
        sims = col.map(sim_map)
        # 空字符串视为缺失
        s_tgt = np.where(col.values != "", sims.astype(float).values, np.nan)

    # --- 2. undergrad_major (graded similarity from MAJOR_SIMILARITY) ---
    u_und = (user.get("undergrad_major_cat") or "").strip()
    s_und = np.full(n, np.nan)
    if u_und and COL_UNDER_CAT_ZH in df.columns:
        col = df[COL_UNDER_CAT_ZH].astype(str).str.strip()
        sim_map = {v: major_similarity(u_und, v) for v in col.unique()}
        sims = col.map(sim_map)
        s_und = np.where(col.values != "", sims.astype(float).values, np.nan)

    # --- 3. GPA ---
    u_gpa = normalize_gpa(user.get("gpa"))
    s_gpa = np.full(n, np.nan)
    if u_gpa is not None and COL_GPA in df.columns:
        r_gpa = df[COL_GPA].apply(normalize_gpa)
        valid = r_gpa.notna()
        diff = np.abs(r_gpa.values.astype(float) - u_gpa)
        raw = 100.0 * np.maximum(0.0, 1.0 - diff / 0.45)
        s_gpa = np.where(valid, raw, np.nan)

    # --- 4. language ---
    u_lang = language_to_toefl_equiv(user.get("ielts"), user.get("toefl"))
    s_lang = np.full(n, np.nan)
    if u_lang is not None and (COL_IELTS in df.columns or COL_TOEFL in df.columns):
        ielts_col = df[COL_IELTS] if COL_IELTS in df.columns else pd.Series(np.nan, index=df.index)
        toefl_col = df[COL_TOEFL] if COL_TOEFL in df.columns else pd.Series(np.nan, index=df.index)
        r_lang = pd.Series(
            [language_to_toefl_equiv(i, t) for i, t in zip(ielts_col, toefl_col)],
            index=df.index,
        )
        valid = r_lang.notna()
        diff = np.abs(r_lang.values.astype(float) - u_lang)
        raw = 100.0 * np.maximum(0.0, 1.0 - diff / 36.0)
        s_lang = np.where(valid, raw, np.nan)

    # --- 5. experience_tags (Jaccard) ---
    u_tags = user.get("tags") if isinstance(user.get("tags"), set) else parse_tag_set(user.get("tags_str"))
    s_tags = np.full(n, np.nan)
    if u_tags and COL_TAGS in df.columns:
        for i, cell in enumerate(df[COL_TAGS].values):
            r_t = parse_tag_set(cell)
            j = jaccard_score(u_tags, r_t)
            if j is not None:
                s_tags[i] = j

    # --- 6. experience_embed (cosine) ---
    s_emb = np.full(n, np.nan)
    if emb_store is not None and emb_store.ready and user_vec is not None:
        mat = emb_store.build_matrix(n)
        mask = emb_store.row_has_vec_mask(n)
        s_emb = cosine_similarity_1_vs_n(user_vec, mat, mask)

    # --- 加权合分 ---
    dim_map: dict[str, np.ndarray] = {
        "target_major": s_tgt,
        "undergrad_major": s_und,
        "gpa": s_gpa,
        "language": s_lang,
        "experience_tags": s_tags,
        "experience_embed": s_emb,
    }

    total = np.zeros(n, dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)

    for key, arr in dim_map.items():
        wt = w.get(key, 0.0)
        if wt <= 0:
            continue
        valid = ~np.isnan(arr)
        total += np.where(valid, wt * arr, 0.0)
        wsum += np.where(valid, wt, 0.0)

    safe_wsum = np.where(wsum > 0, wsum, 1.0)
    final = np.clip(total / safe_wsum, 0.0, 100.0)
    final = np.where(wsum > 0, final, 0.0)

    df_out = df.copy()
    df_out["_match_score"] = np.round(final, 2)

    # 各子分数保留供前端详情展示
    for key, arr in dim_map.items():
        df_out[f"_sub_{key}"] = np.round(np.nan_to_num(arr, nan=-1.0), 1)

    cols_show = [
        "_match_score",
        COL_SCHOOL, COL_PROG, COL_TARGET_CAT_ZH, COL_UNDER_CAT_ZH,
        COL_GPA, COL_IELTS, COL_TOEFL, COL_TAGS,
        "_sub_target_major", "_sub_undergrad_major", "_sub_gpa",
        "_sub_language", "_sub_experience_tags", "_sub_experience_embed",
    ]
    cols_exist = [c for c in cols_show if c in df_out.columns]
    top_k_result = df_out.nlargest(k, "_match_score")[cols_exist]
    # 返回 top-k 结果 + 全表带分数副本（用于查找同项目辅助 case）
    return top_k_result, df_out


# ======================== 同项目辅助 case ========================

def find_supporting_case(
    df_scored: pd.DataFrame,
    primary_idx: int,
    threshold: float = 15.0,
) -> pd.Series | None:
    """
    在全表中找与 primary 同学校+同专业、且匹配分差 <= threshold 的最佳辅助行。
    返回该行 Series（含 _match_score 等列）或 None。
    """
    p = df_scored.loc[primary_idx]
    school = str(p.get(COL_SCHOOL, "")).strip()
    prog = str(p.get(COL_PROG, "")).strip()
    p_score = float(p.get("_match_score", 0))

    if not school or not prog:
        return None

    mask = (
        (df_scored[COL_SCHOOL].str.strip() == school)
        & (df_scored[COL_PROG].str.strip() == prog)
        & (df_scored.index != primary_idx)
    )
    candidates = df_scored.loc[mask]
    if candidates.empty:
        return None

    best_idx = candidates["_match_score"].idxmax()
    best = candidates.loc[best_idx]
    if abs(float(best["_match_score"]) - p_score) <= threshold:
        return best
    return None


# ======================== LLM 推荐理由 ========================

REASON_MODEL = "qwen3.5-flash"

def _get_chat_client():
    """复用 DashScope OpenAI 兼容客户端。"""
    from openai import OpenAI
    api_key = (os.environ.get("DASHSCOPE_API_KEY") or "").strip().strip('"').strip("'")
    base_url = (os.environ.get("DASHSCOPE_BASE_URL")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未配置")
    return OpenAI(api_key=api_key, base_url=base_url)


def _fmt_sub(v: float) -> str:
    return "N/A" if v < 0 else f"{v:.1f}"


def _build_case_block(
    row: pd.Series, label: str, lang: str = "zh"
) -> str:
    school = row.get(COL_SCHOOL, "")
    prog = row.get(COL_PROG, "")
    score = row.get("_match_score", 0)
    gpa = row.get(COL_GPA, "")
    ielts = row.get(COL_IELTS, "")
    toefl = row.get(COL_TOEFL, "")
    tags = row.get(COL_TAGS, "")

    sub_tgt = _fmt_sub(float(row.get("_sub_target_major", -1)))
    sub_und = _fmt_sub(float(row.get("_sub_undergrad_major", -1)))
    sub_gpa = _fmt_sub(float(row.get("_sub_gpa", -1)))
    sub_lang = _fmt_sub(float(row.get("_sub_language", -1)))
    sub_tag = _fmt_sub(float(row.get("_sub_experience_tags", -1)))
    sub_emb = _fmt_sub(float(row.get("_sub_experience_embed", -1)))

    if lang == "zh":
        return (
            f"【{label} — 匹配度 {score}分】\n"
            f"- 录取学校: {school}  录取专业: {prog}\n"
            f"- 该案例 GPA: {gpa}, 雅思: {ielts}, 托福: {toefl}\n"
            f"- 经历标签: {tags}\n"
            f"- 子分数: 目标专业={sub_tgt}, 本科专业={sub_und}, "
            f"GPA={sub_gpa}, 语言={sub_lang}, 标签={sub_tag}, 语义向量={sub_emb}\n"
        )
    return (
        f"[{label} — Match Score {score}]\n"
        f"- School: {school}  Program: {prog}\n"
        f"- Case GPA: {gpa}, IELTS: {ielts}, TOEFL: {toefl}\n"
        f"- Experience Tags: {tags}\n"
        f"- Sub-scores: Target={sub_tgt}, Undergrad={sub_und}, "
        f"GPA={sub_gpa}, Language={sub_lang}, Tags={sub_tag}, Embedding={sub_emb}\n"
    )


def generate_reason_stream(
    user: dict[str, Any],
    primary_row: pd.Series,
    supporting_row: pd.Series | None = None,
    experience_text: str = "",
    lang: str = "zh",
):
    """
    构建 prompt 并调用 Qwen 流式返回推荐理由。
    返回一个 generator，逐块 yield 文本片段（适配 st.write_stream）。
    """
    # 用户背景
    u_und = user.get("undergrad_major_cat", "")
    u_gpa = user.get("gpa", "")
    u_ielts = user.get("ielts", "")
    u_toefl = user.get("toefl", "")
    u_tags = user.get("tags", set())
    tags_str = "; ".join(sorted(u_tags)) if isinstance(u_tags, set) else str(u_tags)

    if lang == "zh":
        user_block = (
            "【申请者背景】\n"
            f"- 本科专业大类: {u_und}\n"
            f"- GPA: {u_gpa if u_gpa else '未提供'}\n"
            f"- 语言: 雅思 {u_ielts if u_ielts else 'N/A'} / 托福 {u_toefl if u_toefl else 'N/A'}\n"
            f"- 经历标签: {tags_str if tags_str else '未提供'}\n"
            f"- 经历描述: {experience_text if experience_text else '未提供'}\n"
        )
        primary_block = _build_case_block(primary_row, "推荐案例 1（主）", "zh")
        support_block = ""
        if supporting_row is not None:
            support_block = (
                _build_case_block(supporting_row, "推荐案例 2（辅助）", "zh")
                + "（说明：该项目有多位相似背景的录取者，进一步印证了匹配度。）\n"
            )
        system_msg = (
            '你是一位经验丰富的留学申请顾问，语气专业但不空洞。\n'
            '写作要求：\n'
            '1. 不要用「高度契合」「完美匹配」等套话开头，直接切入最关键的一个洞察。\n'
            '2. 引用子分数时，必须用普通人能理解的语言解释其含义，例如「语义向量分53」应解释为「您描述的经历与该项目历史录取者的经历在语义上只有中等相似度，说明文书还有优化空间」。\n'
            '3. 结合该学校/专业的特点（课程设置、培养方向、行业口碑等你已知的信息）做针对性分析，不要只分析申请者。\n'
            '4. 对弱项给出具体、可执行的改进建议，避免空泛的「建议加强」。\n'
            '5. 控制在 150-250 字。\n'
        )
        user_msg = (
            "根据下面的数据撰写推荐理由与申请建议。\n\n"
            f"{user_block}\n{primary_block}\n{support_block}"
        )
    else:
        user_block = (
            "[Applicant Profile]\n"
            f"- Undergraduate Field: {u_und}\n"
            f"- GPA: {u_gpa if u_gpa else 'Not provided'}\n"
            f"- Language: IELTS {u_ielts if u_ielts else 'N/A'} / TOEFL {u_toefl if u_toefl else 'N/A'}\n"
            f"- Experience Tags: {tags_str if tags_str else 'Not provided'}\n"
            f"- Experience Description: {experience_text if experience_text else 'Not provided'}\n"
        )
        primary_block = _build_case_block(primary_row, "Recommended Case 1 (Primary)", "en")
        support_block = ""
        if supporting_row is not None:
            support_block = (
                _build_case_block(supporting_row, "Recommended Case 2 (Supporting)", "en")
                + "(Note: Multiple admits with similar profiles further validate the match.)\n"
            )
        system_msg = (
            "You are an experienced study-abroad admissions consultant. Your tone is professional yet conversational.\n"
            "Writing guidelines:\n"
            "1. Do NOT open with generic phrases like 'strong fit' or 'highly aligned'. Lead with your single most important insight.\n"
            "2. When citing sub-scores, always translate them into plain language. For example, an Embedding score of 53 means "
            "'your described experience is only moderately similar to past admits at this program, signaling room to sharpen your narrative.'\n"
            "3. Weave in what you know about the school/program itself — curriculum focus, industry reputation, pedagogy — don't only analyze the applicant.\n"
            "4. For weak dimensions, give concrete, actionable advice (not vague 'consider improving').\n"
            "5. Keep it between 150-250 words.\n"
        )
        user_msg = (
            "Based on the data below, write a recommendation reason and application advice.\n\n"
            f"{user_block}\n{primary_block}\n{support_block}"
        )

    client = _get_chat_client()
    stream = client.chat.completions.create(
        model=REASON_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content
