# -*- coding: utf-8 -*-
"""
仅对 mapping 中 broad_category 为「其他/待人工细分」的专业，调用 DeepSeek 重新归类，
并与原表合并写出（默认 *_refined.csv）。

环境变量（必填）：
  DEEPSEEK_API_KEY   DeepSeek API 密钥

可选：
  DEEPSEEK_MODEL     默认 deepseek-chat
  DEEPSEEK_BASE_URL  默认 https://api.deepseek.com/v1
  BATCH_SIZE         每批条数，默认 40
  REQUEST_DELAY      每批间隔秒，默认 2
  REFINE_INPLACE     设为 1 时直接覆盖 major_mapping_acceptance.csv / major_mapping_undergrad.csv

依赖：pip install requests tqdm python-dotenv

说明：项目通过 ``env_loader`` 加载根目录 ``.env``（若存在）。

输入/输出文件与 ``major_mapping_apply.py`` 所用的 ``major_mapping_*.csv`` 同名；
精修后的表默认写入 ``*_refined.csv``，除非设置环境变量 ``REFINE_INPLACE=1``。

前置：须已生成 ``major_mapping_acceptance.csv`` / ``major_mapping_undergrad.csv``（由你的专业归类流程产出）。
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path

try:
    import requests
except ImportError as e:
    raise SystemExit("请先安装: pip install requests tqdm") from e

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

BASE = Path(__file__).resolve().parent

from env_loader import load_project_env

load_project_env()
SRC_ACCEPT = BASE / "major_mapping_acceptance.csv"
SRC_UNDER = BASE / "major_mapping_undergrad.csv"
OUT_ACCEPT = BASE / "major_mapping_acceptance_refined.csv"
OUT_UNDER = BASE / "major_mapping_undergrad_refined.csv"
STATS_ACCEPT = BASE / "major_category_stats_acceptance_refined.csv"
STATS_UNDER = BASE / "major_category_stats_undergrad_refined.csv"
CHECKPOINT = BASE / "deepseek_refine_checkpoint.json"

OTHER_BUCKET = "其他/待人工细分"

# 与 remap_majors_taxonomy.py 中 24 类口径一致
VALID_LABELS = [
    "管理",
    "商科",
    "金融",
    "会计",
    "语言文学",
    "教育",
    "计算机科学",
    "数据科学",
    "信息技术",
    "经济与贸易",
    "电子电气与通信",
    "机械与先进制造",
    "法学",
    "公共政策",
    "生物医学与健康",
    "数学与统计",
    "材料",
    "化学",
    "土木建筑与环境",
    "传媒设计与艺术",
    "理学基础学科",
    "社会科学与人文",
    "农林与食品",
    OTHER_BUCKET,
]

TAXONOMY = """
你必须为每个专业名称从下列标签中【恰好选一个】（标签文字必须与列表中完全一致，含空格与标点）：

管理 | 商科 | 金融 | 会计 | 语言文学 | 教育 | 计算机科学 | 数据科学 | 信息技术 | 经济与贸易 |
电子电气与通信 | 机械与先进制造 | 法学 | 公共政策 | 生物医学与健康 | 数学与统计 |
材料 | 化学 | 土木建筑与环境 | 传媒设计与艺术 | 理学基础学科 | 社会科学与人文 |
农林与食品 | 其他/待人工细分

说明要点：
- 法学类、各类法律/法务学位选「法学」；公共管理、行政管理、公共政策、国际关系、外交等选「公共政策」。
- 工商管理、信管、工程管理等选「管理」；市场营销、国际商务、旅游管理等偏商科的选「商科」或「管理」中以学位培养目标为准。
- 「材料」与「化学」分列；化工信条归入「化学」，材料成型等归入「材料」。
- 实在无法判断或强交叉学科，选「其他/待人工细分」。
"""


def get_api_key() -> str:
    k = os.environ.get("DEEPSEEK_API_KEY")
    if k:
        k = k.strip().strip('"').strip("'")
    if not k:
        hint = ""
        if not (BASE / ".env").is_file():
            hint = f"\n未找到 {BASE / '.env'}，或其中未定义 DEEPSEEK_API_KEY。"
        else:
            try:
                import dotenv  # noqa: F401
            except ImportError:
                hint = "\n已存在 .env 但未安装 python-dotenv，请执行：pip install python-dotenv"
            else:
                hint = (
                    "\n请确认 .env 中为：DEEPSEEK_API_KEY=sk-...（不要多余引号；变量名须完全一致）"
                )
        raise SystemExit(
            "未读取到 DEEPSEEK_API_KEY。"
            "\n可选方式：1) 在系统或终端设置环境变量 DEEPSEEK_API_KEY；"
            "\n2) 在脚本同目录 .env 中写入 DEEPSEEK_API_KEY=sk-... 并 pip install python-dotenv"
            + hint
        )
    return k


def clean_json_text(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def load_mapping_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "major": (row.get("major") or "").strip(),
                    "count": (row.get("count") or "0").strip(),
                    "broad_category": (row.get("broad_category") or "").strip(),
                }
            )
    return rows


def write_mapping(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["major", "count", "broad_category"])
        for row in rows:
            w.writerow([row["major"], row["count"], row["broad_category"]])


def write_stats_from_rows(rows: list[dict[str, str]], out: Path) -> None:
    from collections import Counter

    by_cat: Counter[str] = Counter()
    uniq: dict[str, set[str]] = {}
    for row in rows:
        cat = row["broad_category"] or OTHER_BUCKET
        maj = row["major"]
        by_cat[cat] += int(row["count"] or 0)
        uniq.setdefault(cat, set()).add(maj)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["broad_category", "total_records", "unique_majors"])
        for cat, n in by_cat.most_common():
            w.writerow([cat, n, len(uniq.get(cat, set()))])


def build_prompt(batch: list[str], kind_zh: str) -> str:
    majors_json = json.dumps(batch, ensure_ascii=False)
    return f"""{TAXONOMY}

任务：为下列【{kind_zh}】名称各选一个标签。

专业列表（JSON 数组，共 {len(batch)} 条）：
{majors_json}

要求：
1. 只输出一个 JSON 对象：键为专业名称（必须与列表中完全一致），值为上述标签之一。
2. 不要 Markdown、不要解释。
3. 必须覆盖列表中每一个专业，不可遗漏。"""


def normalize_label(v: str | None) -> str:
    if not v:
        return OTHER_BUCKET
    v = v.strip()
    if v in VALID_LABELS:
        return v
    for lab in VALID_LABELS:
        if lab in v or v in lab:
            return lab
    return OTHER_BUCKET


def deepseek_classify_batch(
    api_key: str,
    base_url: str,
    model: str,
    batch: list[str],
    kind_zh: str,
    retries: int = 5,
) -> dict[str, str]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    prompt = build_prompt(batch, kind_zh)
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是中国高等教育专业分类助手，只输出合法 JSON 对象。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=120)
            r.raise_for_status()
            data = r.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if not content or not str(content).strip():
                time.sleep(2 + attempt)
                continue
            cleaned = clean_json_text(str(content))
            obj = json.loads(cleaned)
            if not isinstance(obj, dict):
                continue
            out: dict[str, str] = {}
            for k, v in obj.items():
                out[str(k).strip()] = normalize_label(v if isinstance(v, str) else "")
            return out
        except json.JSONDecodeError:
            time.sleep(2 + attempt * 2)
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code in (429, 503):
                wait = 20 + attempt * 10
                print(f"\nHTTP {code}，等待 {wait}s …")
                time.sleep(wait)
            else:
                print(f"\nHTTP 错误: {e}")
                time.sleep(5)
        except Exception as e:
            print(f"\n请求异常: {e}")
            time.sleep(5)
    return {m: OTHER_BUCKET for m in batch}


def merge_refined(
    rows: list[dict[str, str]],
    refined: dict[str, str],
) -> list[dict[str, str]]:
    out = []
    for row in rows:
        m = row["major"]
        if row["broad_category"] == OTHER_BUCKET and m in refined:
            new_row = dict(row)
            new_row["broad_category"] = refined[m]
            out.append(new_row)
        else:
            out.append(dict(row))
    return out


def run_file(
    src: Path,
    kind_zh: str,
    ck_key: str,
    state: dict,
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int,
    delay: float,
) -> list[dict[str, str]]:
    rows = load_mapping_rows(src)
    to_refine = [
        r["major"]
        for r in rows
        if r["broad_category"] == OTHER_BUCKET and r["major"]
    ]
    done: dict[str, str] = dict(state.get(ck_key, {}))

    need = [m for m in to_refine if m not in done]
    if not need:
        print(f"[{kind_zh}] 无待处理项或已全部在 checkpoint。")
    else:
        it = range(0, len(need), batch_size)
        if tqdm:
            it = tqdm(list(it), desc=kind_zh)
        for start in it:
            batch = need[start : start + batch_size]
            partial = deepseek_classify_batch(api_key, base_url, model, batch, kind_zh)
            done.update(partial)
            state[ck_key] = done
            with CHECKPOINT.open("w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=0)
            time.sleep(delay)

    refined_full = {**done}
    return merge_refined(rows, refined_full)


def main() -> None:
    """加载 checkpoint，分批调用 DeepSeek 精修「其他/待人工细分」后写出 mapping 与统计。"""
    api_key = get_api_key()
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    batch_size = int(os.environ.get("BATCH_SIZE", "40"))
    delay = float(os.environ.get("REQUEST_DELAY", "2"))
    inplace = os.environ.get("REFINE_INPLACE", "").strip() in ("1", "true", "yes")

    state: dict = {}
    if CHECKPOINT.exists():
        try:
            with CHECKPOINT.open("r", encoding="utf-8") as f:
                state = json.load(f)
        except json.JSONDecodeError:
            state = {}

    if not SRC_ACCEPT.exists() or not SRC_UNDER.exists():
        raise SystemExit(f"请先存在 {SRC_ACCEPT.name} 与 {SRC_UNDER.name}")

    print(f"DeepSeek 模型: {model} | 每批 {batch_size} | 间隔 {delay}s")

    acc_rows = run_file(
        SRC_ACCEPT, "录取专业", "acceptance", state, api_key, base_url, model, batch_size, delay
    )
    und_rows = run_file(
        SRC_UNDER, "本科专业", "undergrad", state, api_key, base_url, model, batch_size, delay
    )

    out_a = SRC_ACCEPT if inplace else OUT_ACCEPT
    out_u = SRC_UNDER if inplace else OUT_UNDER
    write_mapping(out_a, acc_rows)
    write_mapping(out_u, und_rows)
    write_stats_from_rows(acc_rows, STATS_ACCEPT)
    write_stats_from_rows(und_rows, STATS_UNDER)

    print("完成。")
    print(f"  录取专业 -> {out_a}")
    print(f"  本科专业 -> {out_u}")
    print(f"  统计 -> {STATS_ACCEPT} , {STATS_UNDER}")
    print(f"  Checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()
