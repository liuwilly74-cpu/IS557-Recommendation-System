# -*- coding: utf-8 -*-
"""
将专业大类映射表合并进主数据 CSV。

输入映射（与本脚本同目录、默认文件名）：

- major_mapping_acceptance.csv — 列 major, count, broad_category（录取专业）
- major_mapping_undergrad.csv — 同上（本科专业）

主表默认 ``offers_processed.csv``，需包含列「录取专业」「本科专业」。

写出四列（若已存在则先删列名再追加到表尾，顺序为中-英-中-英）：

- 录取专业大类 / acceptance_category
- 本科专业大类 / undergrad_category

用法：

  python major_mapping_apply.py
  python major_mapping_apply.py --output offers_with_categories.csv

默认覆盖写入 ``--input`` 指定文件；先写临时文件再 ``os.replace``，降低损坏风险。
"""
from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEFAULT_DATA = BASE / "offers_processed.csv"
DEFAULT_ACC = BASE / "major_mapping_acceptance.csv"
DEFAULT_UG = BASE / "major_mapping_undergrad.csv"

COL_ACCEPT = "录取专业大类"
COL_ACCEPT_EN = "acceptance_category"
COL_UNDER = "本科专业大类"
COL_UNDER_EN = "undergrad_category"

# 与 remap_majors_taxonomy 口径一致；未知中文大类留空
CATEGORY_ZH_EN: dict[str, str] = {
    "管理": "Management",
    "商科": "Business",
    "金融": "Finance",
    "会计": "Accounting",
    "语言文学": "Language & Literature",
    "教育": "Education",
    "计算机科学": "Computer Science",
    "数据科学": "Data Science",
    "信息技术": "Information Technology",
    "经济与贸易": "Economics & Trade",
    "电子电气与通信": "Electronics, Electrical & Communication",
    "机械与先进制造": "Mechanical & Advanced Manufacturing",
    "法学": "Law",
    "公共政策": "Public Policy",
    "生物医学与健康": "Biomedical & Health",
    "数学与统计": "Mathematics & Statistics",
    "材料": "Materials",
    "化学": "Chemistry",
    "土木建筑与环境": "Civil Engineering, Architecture & Environment",
    "传媒设计与艺术": "Media, Design & Arts",
    "理学基础学科": "Basic Sciences",
    "社会科学与人文": "Social Sciences & Humanities",
    "农林与食品": "Agriculture, Forestry & Food",
    "其他/待人工细分": "Other",
}


def zh_to_en(cat_zh: str) -> str:
    z = (cat_zh or "").strip()
    return CATEGORY_ZH_EN.get(z, "")


def merge_fieldnames(fieldnames: list[str]) -> list[str]:
    """去掉旧四类列后，在末尾按 中-英-中-英 顺序接回。"""
    drop = {COL_ACCEPT, COL_ACCEPT_EN, COL_UNDER, COL_UNDER_EN}
    base = [f for f in fieldnames if f not in drop]
    block = [COL_ACCEPT, COL_ACCEPT_EN, COL_UNDER, COL_UNDER_EN]
    return base + block


def replace_or_fallback(tmp_path: str, out: Path) -> None:
    """Windows 下用 os.replace 覆盖已存在目标；若目标被占用则写到备用文件。"""
    try:
        os.replace(tmp_path, str(out))
        return
    except OSError as e:
        # 常见：PermissionError，目标被 Cursor/Excel 等占用
        alt = out.parent / f"{out.stem}_new{out.suffix}"
        try:
            os.replace(tmp_path, str(alt))
        except OSError as e2:
            Path(tmp_path).unlink(missing_ok=True)
            raise SystemExit(
                f"无法写入目标或备用文件：{out} / {alt}\n"
                f"原因：{e2}\n"
                "请关闭占用该 CSV 的程序（编辑器、Excel）后重试，或使用：\n"
                "  python major_mapping_apply.py --output 其他路径.csv"
            ) from e2
        raise SystemExit(
            f"无法覆盖已占用文件：{out}\n"
            f"（原因：{e}）\n"
            f"结果已写入备用文件：{alt}\n"
            "关闭占用后删除或重命名原文件，再将备用文件改名为目标文件名。"
        )


def load_mapping(path: Path) -> dict[str, str]:
    """读取 mapping CSV，返回 major 原文 -> broad_category（后者可为空字符串）。"""
    m: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            major = (row.get("major") or "").strip()
            cat = (row.get("broad_category") or "").strip()
            if major:
                m[major] = cat
    return m


def main() -> None:
    """将两张映射表 join 到主表每行，并写出带中英大类列的 CSV。"""
    ap = argparse.ArgumentParser(description="将专业大类映射写回主表 CSV")
    ap.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATA,
        help=f"主数据 CSV（默认 {DEFAULT_DATA.name}）",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出路径；默认与 --input 相同（覆盖原文件）",
    )
    ap.add_argument(
        "--acceptance-map",
        type=Path,
        default=DEFAULT_ACC,
        help="录取专业 mapping",
    )
    ap.add_argument(
        "--undergrad-map",
        type=Path,
        default=DEFAULT_UG,
        help="本科专业 mapping",
    )
    args = ap.parse_args()
    inp = args.input.resolve()
    out = (args.output or args.input).resolve()
    map_acc = args.acceptance_map.resolve()
    map_ug = args.undergrad_map.resolve()

    for p, name in [(inp, "input"), (map_acc, "acceptance map"), (map_ug, "undergrad map")]:
        if not p.exists():
            raise SystemExit(f"找不到文件 ({name}): {p}")

    acc = load_mapping(map_acc)
    ug = load_mapping(map_ug)

    tmp_path: str | None = None
    try:
        with inp.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise SystemExit("CSV 无表头")
            fieldnames = merge_fieldnames(list(reader.fieldnames))

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8-sig",
                newline="",
                delete=False,
                suffix=".csv",
                prefix="compass_mapped_",
                dir=out.parent,
            ) as wf:
                tmp_path = wf.name
                writer = csv.DictWriter(wf, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in reader:
                    a = (row.get("录取专业") or "").strip()
                    b = (row.get("本科专业") or "").strip()
                    row[COL_ACCEPT] = acc.get(a, "")
                    row[COL_UNDER] = ug.get(b, "")
                    row[COL_ACCEPT_EN] = zh_to_en(row[COL_ACCEPT])
                    row[COL_UNDER_EN] = zh_to_en(row[COL_UNDER])
                    writer.writerow(row)

        if tmp_path:
            replace_or_fallback(tmp_path, out)
            tmp_path = None
    except Exception:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        raise

    print(f"完成 -> {out}")


if __name__ == "__main__":
    main()
