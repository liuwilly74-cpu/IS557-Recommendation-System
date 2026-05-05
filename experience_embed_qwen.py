# -*- coding: utf-8 -*-
"""
用 DashScope embedding 对 experience 做【条目级】向量化（去重、缓存、断点续跑）。

默认工作目录为本脚本所在文件夹；默认主表为 ``offers_processed.csv``（与仓库中
`raw_offers.csv` / `major_mapping_*.csv` 等命名一致）。

输入：
- 默认读取同目录下的 offers_processed.csv
- 若存在列 experience_items（由 experience_tag_rules.py 生成，'|' 分隔），优先使用
- 否则从 experience 原文本按行切分成条目

输出（同目录）：
- experience_item_embeddings.sqlite — 按 (text_hash, model) 缓存向量
- experience_item_index.csv — row_id / 条目与 text_hash 对照（见模块内说明）
- experience_embed_checkpoint.json — 断点（与 --resume 配合）

行号说明：写入 index 的 row_id 为「数据行序号」（不含表头），与当前 CSV
物理行顺序一一对应；若删改/重排行后需重跑或改用稳定主键列。

环境变量（必填）：
  DASHSCOPE_API_KEY   DashScope API Key

可选环境变量：
  DASHSCOPE_BASE_URL   默认 https://dashscope.aliyuncs.com/compatible-mode/v1
  EMBED_MODEL          默认 text-embedding-v4
  EMBED_DIMENSIONS     默认 1024
  BATCH_SIZE           默认 6（text-embedding-v4 单次 input 条数上限 10）
  REQUEST_DELAY        默认 0.5（秒）

用法：
  python experience_embed_qwen.py
  python experience_embed_qwen.py --input offers_processed.csv --resume
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Iterable

BASE = Path(__file__).resolve().parent

from env_loader import load_project_env

load_project_env()

from openai import OpenAI

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    tqdm = None  # type: ignore


DEFAULT_INPUT = BASE / "offers_processed.csv"

COL_EXPERIENCE = "experience"
COL_ITEMS = "experience_items"

DB_PATH = BASE / "experience_item_embeddings.sqlite"
OUT_INDEX = BASE / "experience_item_index.csv"
CHECKPOINT = BASE / "experience_embed_checkpoint.json"


def get_env(name: str, default: str | None = None) -> str | None:
    """读取环境变量，去掉首尾空白与常见引号包裹。"""
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip().strip('"').strip("'")
    return v


def require_api_key() -> str:
    k = get_env("DASHSCOPE_API_KEY")
    if not k:
        raise SystemExit(
            "未读取到 DASHSCOPE_API_KEY。请确认 .env 中包含此变量，或在 PowerShell 中执行：\n"
            '  $env:DASHSCOPE_API_KEY="sk-..."'
        )
    return k


def norm_text(s: str) -> str:
    """统一换行符与分号类分隔符，便于按行拆分经历条目。"""
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("；", "\n").replace(";", "\n")
    return s


_RE_BULLET = re.compile(r"^\s*(?:[-*]|(?:\d+)[\.\、\)]|[①②③④⑤⑥⑦⑧⑨⑩])\s*")
_RE_SPACE = re.compile(r"[ \t\u3000]+")


def split_items(experience: str) -> list[str]:
    """从一整格 experience 文本中切出单行条目列表（去列表符号、压缩空白）。"""
    txt = norm_text(experience)
    parts: list[str] = []
    for raw in txt.split("\n"):
        line = raw.strip().strip('"').strip()
        if not line:
            continue
        line = _RE_BULLET.sub("", line).strip()
        line = _RE_SPACE.sub(" ", line).strip("；;，, ").strip()
        if line:
            parts.append(line)
    return parts


def sha1_text(s: str) -> str:
    """UTF-8 文本的 SHA1 十六进制串，用作跨文件关联与主键。"""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# =================== SQLite ===================

def db_init(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings_v2 (
          text_hash TEXT NOT NULL,
          model     TEXT NOT NULL,
          text      TEXT NOT NULL,
          dim       INTEGER NOT NULL,
          embedding_json TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          PRIMARY KEY(text_hash, model)
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_model_v2 ON embeddings_v2(model);")
    conn.commit()


def db_get_many(
    conn: sqlite3.Connection, hashes: list[str], model: str
) -> dict[str, list[float]]:
    if not hashes:
        return {}
    q = ",".join(["?"] * len(hashes))
    cur = conn.execute(
        f"SELECT text_hash, embedding_json FROM embeddings_v2 WHERE model = ? AND text_hash IN ({q})",
        [model] + hashes,
    )
    out: dict[str, list[float]] = {}
    for h, ej in cur.fetchall():
        try:
            out[str(h)] = json.loads(ej)
        except Exception:
            continue
    return out


def db_upsert_many(
    conn: sqlite3.Connection,
    rows: list[tuple[str, str, str, int, list[float]]],
) -> None:
    now = int(time.time())
    conn.executemany(
        """
        INSERT OR REPLACE INTO embeddings_v2
          (text_hash, model, text, dim, embedding_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (h, model, t, dim, json.dumps(vec, ensure_ascii=False), now)
            for (h, t, model, dim, vec) in rows
        ],
    )
    conn.commit()


# =================== OpenAI SDK embedding ===================

def embed_batch(
    client: OpenAI,
    model: str,
    texts: list[str],
    dimensions: int,
    retries: int = 5,
) -> list[list[float]]:
    """调用 DashScope 兼容 embeddings 接口，返回与 ``texts`` 等长的向量列表。"""
    for attempt in range(retries):
        try:
            resp = client.embeddings.create(
                model=model,
                input=texts,
                dimensions=dimensions,
                encoding_format="float",
            )
            arr = sorted(resp.data, key=lambda x: x.index)
            out = [item.embedding for item in arr]
            if len(out) != len(texts):
                raise ValueError("embedding count mismatch")
            return out
        except Exception as e:
            err = str(e)
            if "401" in err or "Unauthorized" in err:
                raise SystemExit(
                    f"API 鉴权失败(401)。请检查 DASHSCOPE_API_KEY 是否正确，"
                    f"以及 base_url 与 key 的站点是否匹配。\n原始错误: {e}"
                ) from e
            if "400" in err or "InvalidParameter" in err:
                raise SystemExit(
                    f"请求参数错误(400)，请检查 batch_size 是否超过模型上限(text-embedding-v4 上限 10)。\n"
                    f"原始错误: {e}"
                ) from e
            if "429" in err or "rate" in err.lower():
                wait = 15 + attempt * 10
                print(f"\n限流，等待 {wait}s…")
                time.sleep(wait)
            else:
                wait = 3 + attempt * 3
                print(f"\n请求异常({attempt+1}/{retries}): {e}，等待 {wait}s…")
                time.sleep(wait)
    raise RuntimeError("embedding 请求多次失败")


# =================== Helpers ===================

def iter_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("CSV 无表头")
        for row in reader:
            yield row


def load_checkpoint() -> dict:
    if not CHECKPOINT.exists():
        return {}
    try:
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_checkpoint(state: dict) -> None:
    CHECKPOINT.write_text(json.dumps(state, ensure_ascii=False, indent=0), encoding="utf-8")


# =================== Main ===================

def main() -> None:
    """扫描主表 CSV，写索引文件并向量化尚未缓存的唯一条目文本。"""
    ap = argparse.ArgumentParser(description="DashScope 条目级 experience embeddings")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--index-out", type=Path, default=OUT_INDEX)
    ap.add_argument("--resume", action="store_true", help="从 checkpoint 断点续跑")
    args = ap.parse_args()

    inp = args.input.resolve()
    if not inp.exists():
        raise SystemExit(f"找不到输入文件: {inp}")

    api_key = require_api_key()
    base_url = get_env("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1") or ""
    model = get_env("EMBED_MODEL", "text-embedding-v4") or ""
    dimensions = int(get_env("EMBED_DIMENSIONS", "1024") or "1024")
    batch_size = int(get_env("BATCH_SIZE", "6") or "6")
    delay = float(get_env("REQUEST_DELAY", "0.5") or "0.5")

    print(f"模型: {model} | 维度: {dimensions} | 批大小: {batch_size} | 间隔: {delay}s")
    print(f"base_url: {base_url}")

    client = OpenAI(api_key=api_key, base_url=base_url)

    conn = sqlite3.connect(str(args.db.resolve()))
    try:
        db_init(conn)

        ck = load_checkpoint() if args.resume else {}
        last_row_id = int(ck.get("last_row_id", 0)) if ck.get("model") == model else 0
        index_has_header = bool(ck.get("index_has_header", False)) if ck.get("model") == model else False

        with inp.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise SystemExit("CSV 无表头")
            has_items_col = COL_ITEMS in reader.fieldnames

        out_index = args.index_out.resolve()
        index_mode = "a" if (args.resume and out_index.exists() and index_has_header) else "w"

        seen_hashes: set[str] = set()
        pending_hashes: list[str] = []
        pending_texts: list[str] = []
        embedded_new = 0

        def flush_pending(pbar_embed=None) -> None:
            nonlocal embedded_new
            if not pending_hashes:
                return
            uniq: dict[str, str] = {}
            for h, t in zip(pending_hashes, pending_texts):
                uniq.setdefault(h, t)
            hs = list(uniq.keys())
            got = db_get_many(conn, hs, model)
            todo_hs = [h for h in hs if h not in got]
            if todo_hs:
                todo_ts = [uniq[h] for h in todo_hs]
                vecs = embed_batch(client, model, todo_ts, dimensions)
                dim = len(vecs[0]) if vecs else 0
                db_rows = [(h, t, model, dim, v) for h, t, v in zip(todo_hs, todo_ts, vecs)]
                db_upsert_many(conn, db_rows)
                embedded_new += len(todo_hs)
                if pbar_embed is not None:
                    pbar_embed.update(len(todo_hs))
                time.sleep(delay)
            pending_hashes.clear()
            pending_texts.clear()

        row_id = 0
        item_total = 0
        with out_index.open(index_mode, encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            if index_mode == "w" or not index_has_header:
                w.writerow(["row_id", "item_idx", "text_hash", "text"])
                index_has_header = True

            pbar_scan = None
            if tqdm is not None:
                pbar_scan = tqdm(desc="scan rows", unit="row", initial=last_row_id)

            pbar_embed = None
            if tqdm is not None:
                pbar_embed = tqdm(desc="embed new items", unit="item")

            try:
                for row in iter_rows(inp):
                    row_id += 1
                    if row_id <= last_row_id:
                        if pbar_scan is not None:
                            pbar_scan.update(1)
                        continue

                    if has_items_col:
                        items_cell = (row.get(COL_ITEMS) or "").strip()
                        items = [x for x in items_cell.split("|") if x.strip()]
                    else:
                        items = split_items((row.get(COL_EXPERIENCE) or "").strip())

                    for j, t in enumerate(items):
                        h = sha1_text(t)
                        item_total += 1
                        w.writerow([row_id, j, h, t])
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            pending_hashes.append(h)
                            pending_texts.append(t)
                            if len(pending_hashes) >= batch_size:
                                flush_pending(pbar_embed)

                    ck2 = {"model": model, "last_row_id": row_id, "index_has_header": index_has_header}
                    save_checkpoint(ck2)
                    if pbar_scan is not None:
                        pbar_scan.update(1)

                flush_pending(pbar_embed)
            finally:
                if pbar_scan is not None:
                    pbar_scan.close()
                if pbar_embed is not None:
                    pbar_embed.close()

        print(f"完成。DB: {args.db.resolve()}")
        print(f"索引: {out_index}")
        print(f"扫描到 row_id={row_id}，写出条目数（含重复）={item_total}，本次新向量化={embedded_new}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
