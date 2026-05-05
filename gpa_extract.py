# -*- coding: utf-8 -*-
"""
从主表「基本背景」列中解析结构化字段（申请状态、GPA、雅思/托福/四六级等），并写回 CSV。

默认数据流：

- 输入：``offers_translated.csv``（由 ``Baidu_Text_transAPI.py`` 或含相同表头的文件提供）
- 输出：``offers_processed.csv``（供 ``major_mapping_apply.py``、经历标签/向量化等脚本使用）

依赖：pandas

用法：

  python gpa_extract.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent

INPUT_FILE = BASE / "offers_translated.csv"
OUTPUT_FILE = BASE / "offers_processed.csv"

_COL_BG = "基本背景"


def extract_background(text) -> pd.Series:
    """
    解析「基本背景」字符串。

    返回六列：申请状态、GPA、雅思成绩、托福成绩、四级成绩、六级成绩。
    无法匹配时对应位置为 None。
    """
    if pd.isna(text) or str(text).strip() == "":
        return pd.Series([None] * 6)

    t = str(text).replace("，", ",").replace("、", ",").replace("；", ",").replace(";", ",")
    status = t.split(",")[0].strip()

    def pick(pattern: str) -> str | None:
        m = re.search(pattern, t, re.I)
        return m.group(1) if m else None

    gpa = pick(r"GPA\s*[:：]?\s*([\d.]+)")
    ielts = pick(r"雅思\s*[:：]?\s*([\d.]+)")
    toefl = pick(r"托福\s*[:：]?\s*([\d.]+)")
    cet4 = pick(r"四级\s*[:：]?\s*([\d.]+)")
    cet6 = pick(r"六级\s*[:：]?\s*([\d.]+)")

    return pd.Series([status, gpa, ielts, toefl, cet4, cet6])


def main() -> None:
    if not INPUT_FILE.is_file():
        raise SystemExit(
            f"找不到输入文件: {INPUT_FILE}\n"
            "请先运行 Baidu_Text_transAPI.py 生成 offers_translated.csv，或设置正确路径。"
        )

    print("正在加载文件...")
    try:
        df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, encoding="gbk")

    if _COL_BG not in df.columns:
        raise SystemExit(f"未找到列: {_COL_BG}")

    print(f"成功读取 {len(df)} 行数据。开始解析 {_COL_BG}...")
    new_cols = ["申请状态", "GPA", "雅思成绩", "托福成绩", "四级成绩", "六级成绩"]
    df[new_cols] = df[_COL_BG].apply(extract_background)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print("-" * 30)
    print("处理完成。")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"新增列: {', '.join(new_cols)}")
    print("-" * 30)
    print("\n处理结果预览：")
    print(df[[_COL_BG] + new_cols].head())


if __name__ == "__main__":
    main()
