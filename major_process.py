# -*- coding: utf-8 -*-
"""
（可选路径）使用 Google Gemini 对指定中文列做批量英译，并写回 ``*_en`` 列。

与 ``Baidu_Text_transAPI.py`` 二选一即可；本脚本默认：

- 输入：``raw_offers.csv``
- 输出：``offers_translated_gemini.csv``（避免覆盖百度管线生成的 ``offers_translated.csv``）

环境变量（必填）：

- GOOGLE_API_KEY 或 GEMINI_API_KEY（任一即可，用于 ``genai.configure``）

依赖：google-generativeai、pandas、tqdm

用法：

  python major_process.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import google.generativeai as genai
import pandas as pd
from tqdm import tqdm

BASE = Path(__file__).resolve().parent

from env_loader import load_project_env

load_project_env()

INPUT_FILE = BASE / "raw_offers.csv"
OUTPUT_FILE = BASE / "offers_translated_gemini.csv"
TRANSLATE_COLS = ["录取学校", "录取专业", "毕业学校", "本科专业"]
REQUEST_DELAY = 12


def _configure_gemini() -> None:
    key = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
    if not key:
        raise SystemExit(
            "未设置 GOOGLE_API_KEY 或 GEMINI_API_KEY。\n"
            "请在 .env 或环境中配置，勿将密钥硬编码进脚本。"
        )
    genai.configure(api_key=key)


def gemini_request_with_retry(model: genai.GenerativeModel, prompt: str) -> str | None:
    """带重试的生成；遇 429 稍作等待。"""
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            return (
                (response.text or "")
                .strip()
                .replace("```json", "")
                .replace("```", "")
            )
        except Exception as e:
            if "429" in str(e):
                print(f"\n触发频率限制，等待 60 秒后重试... (尝试 {attempt + 1}/3)")
                time.sleep(60)
            else:
                print(f"请求出错: {e}")
                return None
    return None


def main() -> None:
    _configure_gemini()
    model = genai.GenerativeModel("gemini-2.5-flash")

    if not INPUT_FILE.is_file():
        raise SystemExit(f"找不到输入文件: {INPUT_FILE}")

    print("正在加载文件...")
    try:
        df = pd.read_csv(INPUT_FILE, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, encoding="gbk")

    for col in TRANSLATE_COLS:
        if col not in df.columns:
            print(f"跳过列: {col}（表中不存在）")
            continue

        unique_vals = [str(x) for x in df[col].dropna().unique() if str(x).strip() != ""]
        print(f"\n正在翻译 [{col}] (共 {len(unique_vals)} 个唯一词条)")

        trans_map: dict[str, str] = {}
        batch_size = 500

        for i in tqdm(range(0, len(unique_vals), batch_size)):
            batch = unique_vals[i : i + batch_size]
            prompt = (
                "Translate the following Chinese academic names into English. "
                "Return ONLY a JSON object where key is Chinese and value is English: "
                f"{batch}"
            )
            result_json = gemini_request_with_retry(model, prompt)
            if result_json:
                try:
                    trans_map.update(json.loads(result_json))
                except json.JSONDecodeError:
                    print("解析失败，跳过该批次")
            time.sleep(REQUEST_DELAY)

        df[f"{col}_en"] = df[col].map(trans_map)
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    df.drop(columns=["raw_status"], inplace=True, errors="ignore")
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n全部任务完成。输出文件：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
