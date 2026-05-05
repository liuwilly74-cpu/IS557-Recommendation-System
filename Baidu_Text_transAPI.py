# -*- coding: utf-8 -*-
"""
使用百度翻译通用 API 对主表中指定中文列做「唯一条目去重后的批量翻译」，并写回 ``*_en`` 列。

数据流（与本仓库文件名一致）：

- 默认读取同目录 ``raw_offers.csv``
- 默认写出 ``offers_translated.csv``（utf-8-sig，便于 Excel）

环境变量（必填，勿提交到版本库）：

- BAIDU_TRANSLATE_APP_ID
- BAIDU_TRANSLATE_APP_KEY

可在项目根目录 ``.env`` 中配置（由 ``env_loader`` 加载）；需 ``pip install python-dotenv``。

可选：修改 ``TARGET_COLS`` 以匹配你的表头。

百度接口说明：单次 ``q`` 长度有限，脚本按 batch 拼接换行发送；免费版注意 QPS。
"""

from __future__ import annotations

import json
import os
import random
import time
from hashlib import md5
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

BASE = Path(__file__).resolve().parent

from env_loader import load_project_env

load_project_env()

# ================= 路径与列配置（与仓库中 CSV 命名一致）=================
INPUT_FILE = BASE / "raw_offers.csv"
OUTPUT_FILE = BASE / "offers_translated.csv"
TARGET_COLS = ["录取学校", "录取专业", "毕业学校", "本科专业"]

# 百度 API
ENDPOINT = "http://api.fanyi.baidu.com"
PATH = "/api/trans/vip/translate"
URL = ENDPOINT + PATH


def _credentials() -> tuple[str, str]:
    app_id = (os.environ.get("BAIDU_TRANSLATE_APP_ID") or "").strip().strip('"').strip("'")
    app_key = (os.environ.get("BAIDU_TRANSLATE_APP_KEY") or "").strip().strip('"').strip("'")
    return app_id, app_key


def make_md5(s: str, encoding: str = "utf-8") -> str:
    """百度 sign 用 MD5（小写十六进制）。"""
    return md5(s.encode(encoding)).hexdigest()


def baidu_translate_batch(
    text_list: list[str], app_id: str, app_key: str, from_lang: str = "zh", to_lang: str = "en"
) -> dict[str, str]:
    """将多条文本用换行拼成一次请求，返回 src -> dst 映射。"""
    if not text_list:
        return {}

    query = "\n".join(text_list)
    salt = random.randint(32768, 65536)
    sign = make_md5(app_id + query + str(salt) + app_key)
    payload = {
        "appid": app_id,
        "q": query,
        "from": from_lang,
        "to": to_lang,
        "salt": salt,
        "sign": sign,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    try:
        r = requests.post(URL, params=payload, headers=headers, timeout=60)
        result = r.json()
        mapping: dict[str, str] = {}
        if "trans_result" in result:
            for item in result["trans_result"]:
                mapping[item["src"]] = item["dst"]
            return mapping
        print(f"API 报错: {json.dumps(result, ensure_ascii=False)}")
        return {}
    except Exception as e:
        print(f"网络请求出错: {e}")
        return {}


def safe_read_csv(file_path: Path) -> pd.DataFrame:
    """依次尝试常见中文 CSV 编码。"""
    encodings = ["gbk", "gb18030", "utf-8-sig", "utf-8"]
    for enc in encodings:
        try:
            print(f"尝试使用 {enc} 编码读取...")
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeError("所有常用编码均无法读取文件，请确认文件是否完整。")


def process_csv() -> None:
    """逐列对唯一条目批译，映射回 DataFrame 后保存。"""
    app_id, app_key = _credentials()
    if not app_id or not app_key:
        raise SystemExit(
            "未配置 BAIDU_TRANSLATE_APP_ID / BAIDU_TRANSLATE_APP_KEY。\n"
            "请在 .env 或环境中设置，勿将密钥写入脚本。"
        )

    print(f"正在读取文件: {INPUT_FILE}...")
    if not INPUT_FILE.is_file():
        raise SystemExit(f"找不到输入文件: {INPUT_FILE}")
    df = safe_read_csv(INPUT_FILE)

    for col in TARGET_COLS:
        if col not in df.columns:
            print(f"跳过列: {col} (未在文件中找到)")
            continue

        print(f"\n正在处理列: {col}...")
        unique_texts = df[col].dropna().unique().tolist()
        unique_texts = [str(x) for x in unique_texts if str(x).strip() != ""]
        print(f"唯一条目数: {len(unique_texts)}")

        translations_dict: dict[str, str] = {}
        batch_size = 50
        for i in tqdm(range(0, len(unique_texts), batch_size)):
            batch = unique_texts[i : i + batch_size]
            res_map = baidu_translate_batch(batch, app_id, app_key)
            translations_dict.update(res_map)
            time.sleep(1.0)

        new_col_name = f"{col}_en"
        df[new_col_name] = df[col].map(translations_dict)
        print(f"列 {col} 翻译完成并存入 {new_col_name}。")

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n处理完成。保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    process_csv()
