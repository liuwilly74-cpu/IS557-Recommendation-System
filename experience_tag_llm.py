# -*- coding: utf-8 -*-
"""
DeepSeek 兜底：仅对规则抽取后标签不足的 ``experience`` 相关行补全标签。

前置：先运行 ``experience_tag_rules.py`` 生成列：

- experience_items, experience_tags, experience_tags_en

本脚本会：

- 识别待兜底行：``experience_tags`` 中标签数 < ``--min-tags``（默认 2）
- 调用 DeepSeek（OpenAI 兼容 ``/chat/completions``）补标签，标签必须来自
  ``experience_tag_rules.TAG_RULES`` 的中文集合
- 写回 ``experience_tags`` / ``experience_tags_en``，并新增：

  - experience_llm_used — ``0`` / ``1``
  - experience_llm_added — 仅 LLM 新增的中文标签，``;`` 分隔

断点：同目录 ``experience_llm_checkpoint.json``（已处理的 ``row_*`` id）。

用法：

  python experience_tag_llm.py
  python experience_tag_llm.py --input offers_processed.csv --output offers_processed_llm.csv

可选环境变量：
  DEEPSEEK_MODEL     默认 deepseek-chat
  DEEPSEEK_BASE_URL  默认 https://api.deepseek.com/v1
  BATCH_SIZE         默认 25（每次请求批内行数）
  REQUEST_DELAY      默认 1.5（秒）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

import experience_tag_rules as et


BASE = Path(__file__).resolve().parent

DEFAULT_INPUT = BASE / "offers_processed.csv"
DEFAULT_OUTPUT = BASE / "offers_processed_llm.csv"

from env_loader import load_project_env

load_project_env()

COL_ITEMS = et.COL_ITEMS
COL_TAGS_ZH = et.COL_TAGS_ZH
COL_TAGS_EN = et.COL_TAGS_EN

COL_LLM_USED = "experience_llm_used"
COL_LLM_ADDED = "experience_llm_added"

CHECKPOINT = BASE / "experience_llm_checkpoint.json"


def get_api_key() -> str:
    """从环境变量读取 DeepSeek Key（需与 ``.env`` / shell 一致）。"""
    k = os.environ.get("DEEPSEEK_API_KEY")
    if k:
        k = k.strip().strip('"').strip("'")
    if not k:
        raise SystemExit(
            "未读取到 DEEPSEEK_API_KEY。请设置环境变量，例如 PowerShell：\n"
            '  $env:DEEPSEEK_API_KEY="sk-..."'
        )
    return k


def clean_json_text(text: str) -> str:
    """去掉模型输出外层 ```json 代码块标记，便于 ``json.loads``。"""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


ALLOWED_TAGS: list[str] = [r.tag_zh for r in et.TAG_RULES]
ALLOWED_TAGS_SET = set(ALLOWED_TAGS)


def normalize_tags(x: Any) -> list[str]:
    """将模型返回的标签列表约束到允许集合，去重保序。"""
    if not isinstance(x, list):
        return []
    out: list[str] = []
    for v in x:
        if not isinstance(v, str):
            continue
        s = v.strip()
        if s in ALLOWED_TAGS_SET and s not in out:
            out.append(s)
        else:
            # 宽松：模型偶发输出带解释
            for t in ALLOWED_TAGS:
                if t in s:
                    if t not in out:
                        out.append(t)
                    break
    return out


def build_prompt(batch: list[dict[str, Any]]) -> str:
    """
    batch: [{"id": "...", "items": [...], "existing_tags": [...]}]
    返回 JSON: {"results": [{"id":..., "tags":[...]}...]}
    """
    payload = json.dumps(batch, ensure_ascii=False)
    allowed = "、".join(ALLOWED_TAGS)
    return f"""你是经历标签抽取器。只能从给定标签表中选择标签（可多选），不要创造新标签，不要输出解释。

标签表（中文，必须原样使用）：{allowed}

输入是一个 JSON 数组，每个元素包含：
- id: 行标识
- items: 清洗后的经历条目数组
- existing_tags: 已有标签（可能为空）

请只输出一个 JSON 对象，格式如下：
{{"results":[{{"id":"...","tags":["标签1","标签2"]}}...]}}

规则：
1) tags 必须来自标签表；若无法判断，返回 ["其他/待人工细分"]。
2) 对于明显重复或泛化的标签可省略，但不要遗漏关键产出/经历类型/岗位。
3) 一般返回 2-8 个标签即可。

输入：{payload}"""


def deepseek_call(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    retries: int = 5,
) -> dict[str, list[str]] | None:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "只输出合法 JSON。"},
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
                data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            )
            obj = json.loads(clean_json_text(content))
            if not isinstance(obj, dict):
                raise ValueError("response not dict")
            results = obj.get("results")
            if not isinstance(results, list):
                raise ValueError("missing results")
            out: dict[str, list[str]] = {}
            for item in results:
                if not isinstance(item, dict):
                    continue
                _id = str(item.get("id", "")).strip()
                tags = normalize_tags(item.get("tags"))
                if _id:
                    out[_id] = tags
            return out
        except json.JSONDecodeError:
            time.sleep(2 + attempt * 2)
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code in (429, 503):
                wait = 15 + attempt * 10
                print(f"HTTP {code}，等待 {wait}s…")
                time.sleep(wait)
            else:
                print(f"HTTP 错误: {e}")
                time.sleep(5)
        except Exception as e:
            print(f"请求异常: {e}")
            time.sleep(5)
    return None


def parse_tags_cell(cell: str) -> list[str]:
    s = (cell or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(";") if p.strip()]
    # 去重保序
    out: list[str] = []
    for p in parts:
        if p not in out:
            out.append(p)
    return out


def merge_fieldnames(fieldnames: list[str]) -> list[str]:
    drop = {COL_LLM_USED, COL_LLM_ADDED}
    base = [f for f in fieldnames if f not in drop]
    return base + [COL_LLM_USED, COL_LLM_ADDED]


def load_checkpoint() -> dict[str, Any]:
    if not CHECKPOINT.exists():
        return {}
    try:
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_checkpoint(state: dict[str, Any]) -> None:
    CHECKPOINT.write_text(json.dumps(state, ensure_ascii=False, indent=0), encoding="utf-8")


def main() -> None:
    """流式读入 CSV，对需兜底的行分批请求 LLM 后写出新 CSV。"""
    ap = argparse.ArgumentParser(description="DeepSeek 兜底补全 experience 标签")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--min-tags", type=int, default=2)
    ap.add_argument("--max-tags", type=int, default=12, help="合并后最多保留多少标签（防爆）")
    args = ap.parse_args()

    inp = args.input.resolve()
    out = args.output.resolve()
    if not inp.exists():
        raise SystemExit(f"找不到输入文件: {inp}")

    api_key = get_api_key()
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    batch_size = int(os.environ.get("BATCH_SIZE", "25"))
    delay = float(os.environ.get("REQUEST_DELAY", "1.5"))

    state = load_checkpoint()
    done: set[str] = set(state.get("done_ids", []))

    tmp_path: str | None = None
    total = 0
    eligible = 0
    called = 0

    try:
        with inp.open("r", encoding="utf-8-sig", newline="") as rf:
            reader = csv.DictReader(rf)
            if not reader.fieldnames:
                raise SystemExit("CSV 无表头")
            for need in (COL_ITEMS, COL_TAGS_ZH, COL_TAGS_EN):
                if need not in reader.fieldnames:
                    raise SystemExit(f"缺少列 {need}。请先运行 experience_tag_rules.py")

            fieldnames = merge_fieldnames(list(reader.fieldnames))

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8-sig",
                newline="",
                delete=False,
                suffix=".csv",
                prefix="compass_exp_llm_",
                dir=out.parent,
            ) as wf:
                tmp_path = wf.name
                writer = csv.DictWriter(wf, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()

                pending_rows: list[dict[str, str]] = []
                pending_payload: list[dict[str, Any]] = []

                def flush() -> None:
                    nonlocal called
                    if not pending_rows:
                        return
                    prompt = build_prompt(pending_payload)
                    resp = deepseek_call(api_key, base_url, model, prompt)
                    called += 1
                    if resp is None:
                        resp = {}

                    for row, payload in zip(pending_rows, pending_payload):
                        _id = payload["id"]
                        existing = payload["existing_tags"]
                        llm_tags = resp.get(_id, [])
                        # 合并去重保序：先 existing，再 llm_tags
                        merged: list[str] = []
                        for t in list(existing) + list(llm_tags):
                            if t in ALLOWED_TAGS_SET and t not in merged:
                                merged.append(t)
                        if args.max_tags and len(merged) > args.max_tags:
                            merged = merged[: args.max_tags]

                        added = [t for t in merged if t not in existing]
                        row[COL_TAGS_ZH] = ";".join(merged)
                        row[COL_TAGS_EN] = ";".join(et.tags_to_en(merged))
                        row[COL_LLM_USED] = "1"
                        row[COL_LLM_ADDED] = ";".join(added)
                        writer.writerow(row)

                        done.add(_id)

                    pending_rows.clear()
                    pending_payload.clear()
                    state["done_ids"] = sorted(done)
                    save_checkpoint(state)
                    time.sleep(delay)

                for row in reader:
                    total += 1
                    items_cell = (row.get(COL_ITEMS) or "").strip()
                    tags_cell = (row.get(COL_TAGS_ZH) or "").strip()
                    existing = parse_tags_cell(tags_cell)
                    tags_n = len(existing)

                    # 需要兜底：items非空且标签不足
                    if items_cell and tags_n < args.min_tags:
                        eligible += 1
                        _id = f"row_{total}"
                        if _id in done:
                            # 已处理过，直接写出（保持已有列值）
                            row.setdefault(COL_LLM_USED, "1")
                            row.setdefault(COL_LLM_ADDED, "")
                            writer.writerow(row)
                            continue

                        items = [x for x in items_cell.split("|") if x.strip()]
                        pending_rows.append(row)
                        pending_payload.append(
                            {"id": _id, "items": items[:25], "existing_tags": existing}
                        )
                        if len(pending_rows) >= batch_size:
                            flush()
                    else:
                        # 不走 LLM
                        row[COL_LLM_USED] = row.get(COL_LLM_USED, "0") or "0"
                        row[COL_LLM_ADDED] = row.get(COL_LLM_ADDED, "") or ""
                        writer.writerow(row)

                flush()

        if tmp_path:
            et.replace_or_fallback(tmp_path, out)
            tmp_path = None
    except Exception:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        raise

    print(f"完成 -> {out}")
    print(f"总行数: {total} | 待兜底: {eligible} | API调用次数: {called}")


if __name__ == "__main__":
    main()

