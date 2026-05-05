# -*- coding: utf-8 -*-
"""
对主表 CSV（默认 ``offers_processed.csv``）的 ``experience`` 列做清洗与受控标签抽取。

输出新增列（默认覆盖 ``--input``；Windows 下目标被占用时由 ``replace_or_fallback`` 写到 ``*_new.csv``）：

- experience_items — 清洗后条目，``|`` 连接
- experience_tags — 中文标签，``;`` 分隔（规则取自 ``TAG_RULES``）
- experience_tags_en — 对应英文标签，``;`` 分隔

下游：可再运行 ``experience_tag_llm.py`` 对标签过少的行做 DeepSeek 兜底。

用法：

  python experience_tag_rules.py
  python experience_tag_rules.py --input offers_processed.csv --output offers_with_exp.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASE = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE / "offers_processed.csv"

COL_SOURCE = "experience"
COL_ITEMS = "experience_items"
COL_TAGS_ZH = "experience_tags"
COL_TAGS_EN = "experience_tags_en"


def replace_or_fallback(tmp_path: str, out: Path) -> None:
    """Windows 下优先 ``os.replace`` 原子覆盖；目标被占用则写入 ``*_new.csv`` 并提示。"""
    # 与 major_mapping_apply 中逻辑一致
    """Windows 下用 os.replace 覆盖已存在目标；若目标被占用则写到备用文件。"""
    try:
        os.replace(tmp_path, str(out))
        return
    except OSError as e:
        alt = out.parent / f"{out.stem}_new{out.suffix}"
        try:
            os.replace(tmp_path, str(alt))
        except OSError as e2:
            Path(tmp_path).unlink(missing_ok=True)
            raise SystemExit(
                f"无法写入目标或备用文件：{out} / {alt}\n"
                f"原因：{e2}\n"
                "请关闭占用该 CSV 的程序（编辑器、Excel）后重试，或使用 --output 写到其他路径。"
            ) from e2
        raise SystemExit(
            f"无法覆盖已占用文件：{out}\n"
            f"（原因：{e}）\n"
            f"结果已写入备用文件：{alt}\n"
            "关闭占用后删除或重命名原文件，再将备用文件改名为目标文件名。"
        ) from e


def norm_text(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    # 常见全角标点归一
    s = s.replace("；", "\n").replace(";", "\n")
    s = s.replace("。", "；").replace("，", "，")
    return s


_RE_BULLET = re.compile(r"^\s*(?:[-*]|(?:\d+)[\.\、\)]|[①②③④⑤⑥⑦⑧⑨⑩])\s*")
_RE_SPACE = re.compile(r"[ \t\u3000]+")


def split_items(experience: str) -> list[str]:
    """将一格 experience 切为条目列表。"""
    txt = norm_text(experience)
    parts = []
    for raw in txt.split("\n"):
        line = raw.strip().strip('"').strip()
        if not line:
            continue
        line = _RE_BULLET.sub("", line).strip()
        line = _RE_SPACE.sub(" ", line).strip("；;，, ").strip()
        if not line:
            continue
        parts.append(line)
    return parts


@dataclass(frozen=True)
class TagRule:
    tag_zh: str
    tag_en: str
    patterns: tuple[re.Pattern[str], ...]


def _rx(p: str) -> re.Pattern[str]:
    return re.compile(p, re.I)


# 受控标签（≈200目标）：先覆盖常见“类型/产出/角色/动作/场景”，后续可以继续追加规则。
TAG_RULES: list[TagRule] = [
    # ===== 产出 / 成果 =====
    TagRule("专利", "Patent", (_rx(r"\bpatent\b|专利"),)),
    TagRule("研究报告/策略", "Research Report/Strategy", (_rx(r"研究报告|策略报告|策略研究|投资策略|资产配置"),)),
    TagRule("软件著作权", "Software Copyright", (_rx(r"软著|软件著作权"),)),
    TagRule("论文发表", "Publication", (_rx(r"发表|期刊|会议论文|\bpublication\b"),)),
    TagRule("SCI论文", "SCI Paper", (_rx(r"\bSCI\b"),)),
    TagRule("EI论文", "EI Paper", (_rx(r"\bEI\b"),)),
    TagRule("核心期刊", "Core Journal", (_rx(r"核心期刊|北大核心|CSSCI"),)),
    TagRule("竞赛参赛", "Competition Participation", (_rx(r"竞赛|比赛|challenge|hackathon|Kaggle|\bACM\b"),)),
    TagRule("竞赛获奖", "Competition Award", (_rx(r"获奖|一等奖|二等奖|三等奖|冠军|亚军|季军"),)),
    TagRule("证书", "Certificate", (_rx(r"证书|资格证|认证"),)),
    TagRule("作品集", "Portfolio", (_rx(r"作品集|portfolio"),)),
    TagRule("数据集构建", "Dataset Construction", (_rx(r"数据集|dataset"),)),
    TagRule("开源贡献", "Open-source Contribution", (_rx(r"开源|open[- ]source|github"),)),
    TagRule("尽调/审阅", "Due Diligence/Review", (_rx(r"尽调|due diligence|DD|材料审阅|条款审阅"),)),
    TagRule("投标/标书", "Bidding/Tender", (_rx(r"投标|标书|tender"),)),
    TagRule("专利/论文检索", "Search (Patent/Paper)", (_rx(r"检索|专利检索|文献检索|prior art"),)),

    # ===== 经历类型 =====
    TagRule("毕业设计", "Capstone/Thesis", (_rx(r"毕设|毕业设计|毕业论文|thesis"),)),
    TagRule("课程项目", "Course Project", (_rx(r"课程设计|课程项目|course project"),)),
    TagRule("科研项目", "Research Project", (_rx(r"科研|课题|实验室|研究|research project"),)),
    TagRule("项目经历", "Project Experience", (_rx(r"项目|系统|平台|方案|产品|prototype|demo"),)),
    TagRule("大创/创新创业", "Innovation/Entrepreneurship", (_rx(r"大创|创新创业|互联网\+|挑战杯"),)),
    TagRule("实习", "Internship", (_rx(r"实习|intern"),)),
    TagRule("全职工作", "Full-time Work", (_rx(r"全职|在职|工作经历|任职"),)),
    TagRule("校招/管培", "Graduate Program", (_rx(r"管培|management trainee|校招"),)),
    TagRule("培训/训练营", "Training/Bootcamp", (_rx(r"训练营|bootcamp|培训|实训"),)),
    TagRule("志愿者", "Volunteer", (_rx(r"志愿者|志愿服务|volunteer"),)),
    TagRule("社团/学生组织", "Student Organization", (_rx(r"学生会|社团|协会|组织部|部长|主席"),)),
    TagRule("海外交流/交换", "Exchange Program", (_rx(r"交换|交流|访学|summer school|暑校"),)),
    TagRule("学术会议参与", "Conference Participation", (_rx(r"学术会议|conference|workshop|poster"),)),

    # ===== 角色 / 岗位（抽象到“岗位族”级别，避免过细失控）=====
    TagRule("券商/投行实习", "Securities/IB Internship", (_rx(r"券商|证券公司|投行|投资银行|IBD|投行部|投融资|承做|承揽|承销"), _rx(r"实习|intern"))),
    TagRule("律所实习", "Law Firm Internship", (_rx(r"律所|律师事务所|law firm"), _rx(r"实习|intern"))),
    TagRule("咨询实习", "Consulting Internship", (_rx(r"咨询公司|管理咨询|战略咨询|咨询岗|consulting"), _rx(r"实习|intern"))),
    TagRule("四大经历", "Big 4 Experience", (_rx(r"四大|德勤|毕马威|安永|普华永道|deloitte|kpmg|ey|pwc"),)),
    TagRule("投研/行研", "Investment/Equity Research", (_rx(r"投研|行业研究|行研|research analyst|equity research"),)),
    TagRule("风控", "Risk Management", (_rx(r"风控|风险控制|credit risk|market risk|var"),)),
    TagRule("资产配置", "Asset Allocation", (_rx(r"资产配置|allocation"),)),
    TagRule("宏观研究", "Macro Research", (_rx(r"宏观|宏观经济|macro"),)),
    TagRule("基金/资管", "Fund/Asset Management", (_rx(r"基金|资管|资产管理|私募|公募|PE|VC|创投"),)),
    TagRule("银行", "Banking", (_rx(r"银行|banking"),)),
    TagRule("保险", "Insurance", (_rx(r"保险|insurance"),)),
    TagRule("会计/审计", "Accounting/Audit", (_rx(r"会计|审计|accounting|audit"),)),
    TagRule("税务", "Tax", (_rx(r"税务|tax"),)),
    TagRule("咨询顾问", "Consultant", (_rx(r"顾问|consultant"),)),
    TagRule("战略/运营", "Strategy/Operations", (_rx(r"战略|strategy|运营|operations"),)),
    TagRule("市场/公关", "Marketing/PR", (_rx(r"市场|营销|marketing|公关|PR|内容运营|新媒体运营"),)),
    TagRule("人力资源", "Human Resources", (_rx(r"人力资源|HR|招聘|recruit"),)),
    TagRule("供应链/物流", "Supply Chain/Logistics", (_rx(r"供应链|物流|仓储|配送|采购|supply chain"),)),
    TagRule("销售/BD", "Sales/BD", (_rx(r"销售|商务拓展|BD|business development"),)),
    TagRule("研发工程师", "R&D Engineer", (_rx(r"研发.*工程师|\bR&D\b.*engineer|开发工程师|技术工程师"),)),
    TagRule("算法工程师", "Algorithm Engineer", (_rx(r"算法工程师|machine learning engineer|ml engineer"),)),
    TagRule("数据分析", "Data Analyst", (_rx(r"数据分析|data analyst|商业分析岗|分析岗"),)),
    TagRule("软件工程师", "Software Engineer", (_rx(r"软件工程师|software engineer|后端|前端|全栈"),)),
    TagRule("数据工程", "Data Engineer", (_rx(r"数据工程|data engineer|ETL|数据仓库|数仓"),)),
    TagRule("BI/可视化", "BI/Visualization", (_rx(r"\bBI\b|PowerBI|Tableau|可视化|dashboard"),)),
    TagRule("测试开发", "Test Development", (_rx(r"测试开发|自动化测试"),)),
    TagRule("信息安全", "Cybersecurity", (_rx(r"信息安全|网络安全|渗透|攻防|security"),)),
    TagRule("嵌入式开发", "Embedded Developer", (_rx(r"嵌入式|firmware"),)),
    TagRule("硬件开发", "Hardware Engineer", (_rx(r"硬件|hardware"),)),
    TagRule("电路设计", "Circuit Design", (_rx(r"电路|PCB|原理图|schematic"),)),
    TagRule("FPGA", "FPGA", (_rx(r"\bFPGA\b"),)),
    TagRule("产品经理", "Product Manager", (_rx(r"产品经理|product manager"),)),
    TagRule("产品/工业设计", "Product/Industrial Design", (_rx(r"产品设计|工业设计|交互设计|UI|UX"),)),
    TagRule("UI/UX设计", "UI/UX Design", (_rx(r"UI|UX|交互设计|用户体验"),)),
    TagRule("内容/新媒体", "Content/New Media", (_rx(r"新媒体|内容运营|短视频|公众号|社媒"),)),
    TagRule("视频/剪辑", "Video Editing", (_rx(r"视频剪辑|剪辑|pr|premiere|after effects|AE"),)),
    TagRule("平面/视觉", "Graphic/Visual Design", (_rx(r"平面设计|视觉设计|海报|品牌视觉"),)),
    TagRule("翻译", "Translation", (_rx(r"翻译|translation"),)),
    TagRule("教师/授课", "Teaching", (_rx(r"教师|老师|授课|讲师|班主任|任课|授课老师"),)),
    TagRule("教研/课程研发", "Curriculum Development", (_rx(r"教研|课程研发|课程设计|题库|备课|教案"),)),
    TagRule("教育机构/学校", "School/Education Institution", (_rx(r"学校|中学|小学|幼儿园|培训机构|教育机构|教培"),)),
    TagRule("英语教学/TESOL", "English Teaching/TESOL", (_rx(r"对外英语教学|TESOL|英语教学"),)),
    TagRule("对外汉语", "Teaching Chinese", (_rx(r"对外汉语|汉语国际教育"),)),
    TagRule("记者/采编", "Journalism/Reporting", (_rx(r"记者|新闻采编|采编|采访|写稿|通讯员|新闻报道|采写"),)),
    TagRule("编辑/出版", "Editing/Publishing", (_rx(r"编辑|编辑出版|出版|审稿|校对|排版|稿件|图书编辑|期刊编辑"),)),
    TagRule("文案/策划", "Copywriting/Planning", (_rx(r"文案|策划|活动策划|内容策划|选题|脚本"),)),
    TagRule("运营（泛）", "Operations (General)", (_rx(r"运营岗|运营专员|用户运营|内容运营|活动运营|社区运营|增长运营"),)),
    TagRule("公关/传播", "PR/Communications", (_rx(r"公关|传播|品牌传播|对外传播|媒体关系|PR\b"),)),
    TagRule("助理（泛）", "Assistant (General)", (_rx(r"(?:行政|项目|产品|运营|市场|研究|商务|法务|审计|会计)?助理"),)),
    TagRule("机械工程师", "Mechanical Engineer", (_rx(r"机械工程师|结构工程师|机械设计"),)),
    TagRule("工艺/制造工程", "Process/Manufacturing", (_rx(r"工艺工程师|制造工程师|生产工艺|工艺改进"),)),
    TagRule("质量/体系", "Quality/Compliance", (_rx(r"质量工程|质量体系|ISO|质量管理"),)),
    TagRule("电气工程师", "Electrical Engineer", (_rx(r"电气工程师"),)),
    TagRule("通信工程师", "Communication Engineer", (_rx(r"通信工程师|通信|5G|4G|LTE"),)),
    TagRule("自动化/控制", "Automation/Control", (_rx(r"自动化|PLC|控制工程|控制系统"),)),
    TagRule("土木/建筑", "Civil/Architecture", (_rx(r"土木|建筑设计师|建筑设计"),)),
    TagRule("测绘/GIS", "Surveying/GIS", (_rx(r"测绘|GIS|地理信息|遥感|RS\b"),)),
    TagRule("环境工程", "Environmental Engineering", (_rx(r"环境工程|环境科学|碳排|双碳"),)),
    TagRule("测试/QA", "QA/Test", (_rx(r"测试|QA|质量保证"),)),
    TagRule("运维/DevOps", "DevOps", (_rx(r"运维|devops|SRE"),)),
    TagRule("研究助理", "Research Assistant", (_rx(r"研究助理|\bRA\b"),)),
    TagRule("助教", "Teaching Assistant", (_rx(r"助教|\bTA\b"),)),
    TagRule("法务/合规", "Legal/Compliance", (_rx(r"法务|合规|合同审查|诉讼|仲裁"),)),
    TagRule("政策研究", "Policy Research", (_rx(r"政策研究|公共政策|政策分析|政府部门|智库"),)),

    # ===== 项目动作 / 职责 =====
    TagRule("结构/机械设计", "Mechanical/Structural Design", (_rx(r"结构设计|装置设计|机构设计|机械设计"),)),
    TagRule("建模仿真", "Modeling & Simulation", (_rx(r"建模|仿真|simulation|有限元|FEA"),)),
    TagRule("控制系统设计", "Control System Design", (_rx(r"控制系统|姿态约束|PID|控制算法"),)),
    TagRule("数据采集", "Data Collection", (_rx(r"数据采集|采样|传感器数据"),)),
    TagRule("数据清洗", "Data Cleaning", (_rx(r"数据清洗|cleaning"),)),
    TagRule("数据标注", "Data Labeling", (_rx(r"标注|labeling"),)),
    TagRule("特征工程", "Feature Engineering", (_rx(r"特征工程|feature engineering"),)),
    TagRule("模型训练", "Model Training", (_rx(r"训练模型|model training|训练"),)),
    TagRule("模型优化", "Model Optimization", (_rx(r"优化研究|调参|hyperparameter|优化"),)),
    TagRule("用户研究", "User Research", (_rx(r"用户研究|访谈|问卷|可用性测试"),)),
    TagRule("需求分析", "Requirements Analysis", (_rx(r"需求分析|PRD|需求文档"),)),
    TagRule("数据分析建模", "Analytics/Modeling", (_rx(r"回归|分类|聚类|时间序列|预测|forecast"),)),
    TagRule("项目管理", "Project Management", (_rx(r"项目管理|里程碑|甘特|scrum|敏捷"),)),
    TagRule("系统实现", "System Implementation", (_rx(r"实现|开发|搭建|实现；|design与实现"),)),
    TagRule("测试验证", "Testing & Validation", (_rx(r"测试|验证|实验验证|对比实验"),)),
    TagRule("文档/报告撰写", "Documentation", (_rx(r"文档|报告|撰写|write-up"),)),
    TagRule("专利撰写", "Patent Writing", (_rx(r"专利.*撰写|撰写专利"),)),

    # ===== 场景 / 方向（不强制归 domain，可作为细粒度背景标签）=====
    TagRule("搜索/推荐", "Search/Recommendation", (_rx(r"搜索引擎|推荐系统|ranking|retrieval"),)),
    TagRule("无人机", "UAV", (_rx(r"无人机|UAV|drone"),)),
    TagRule("证券/金融机构", "Financial Institution", (_rx(r"证券|券商|基金|私募|公募|银行|信托|保险|资管|期货|投行"),)),
    TagRule("法律服务", "Legal Services", (_rx(r"律所|律师事务所|法务|合规|诉讼|仲裁|合同审查"),)),
    TagRule("咨询项目", "Consulting Project", (_rx(r"咨询项目|management consulting|战略咨询|尽调|DD|due diligence"),)),
    TagRule("医疗健康", "Healthcare", (_rx(r"医院|医疗|健康|药|临床|护理"),)),
    TagRule("教育培训", "Education & Training", (_rx(r"教育|培训|教培|课程|教学"),)),
    TagRule("媒体内容", "Media/Content", (_rx(r"媒体|内容|传播|广告|新媒体"),)),
    TagRule("机器人", "Robotics", (_rx(r"机器人|robot"),)),
    TagRule("智能制造", "Smart Manufacturing", (_rx(r"智能制造|自动生产线|产线"),)),
]

def extract_tags(items: Iterable[str]) -> list[str]:
    """拼接条目全文，按 ``TAG_RULES`` 顺序输出命中的中文标签（去重保序）。"""
    text = "\n".join(items)
    tags: set[str] = set()

    for rule in TAG_RULES:
        for pat in rule.patterns:
            if pat.search(text):
                tags.add(rule.tag_zh)
                break

    # 稳定输出顺序：按 TAG_RULES 的出现顺序
    ordered_tags = [r.tag_zh for r in TAG_RULES if r.tag_zh in tags]
    return ordered_tags


TAG_ZH_EN = {r.tag_zh: r.tag_en for r in TAG_RULES}


def tags_to_en(tags_zh: list[str]) -> list[str]:
    out = []
    for t in tags_zh:
        out.append(TAG_ZH_EN.get(t, ""))
    return [x for x in out if x]


def merge_fieldnames(fieldnames: list[str]) -> list[str]:
    drop = {COL_ITEMS, COL_TAGS_ZH, COL_TAGS_EN}
    base = [f for f in fieldnames if f not in drop]
    return base + [COL_ITEMS, COL_TAGS_ZH, COL_TAGS_EN]


def main() -> None:
    """CLI：流式读入 CSV，为每行写入 experience_items / experience_tags / experience_tags_en。"""
    ap = argparse.ArgumentParser(description="experience 列清洗与标签抽取")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=None, help="默认覆盖 input（占用则写 *_new.csv）")
    args = ap.parse_args()

    inp = args.input.resolve()
    out = (args.output or args.input).resolve()
    if not inp.exists():
        raise SystemExit(f"找不到输入文件: {inp}")

    tmp_path: str | None = None
    total = 0
    non_empty = 0
    tag_non_empty = 0

    try:
        with inp.open("r", encoding="utf-8-sig", newline="") as rf:
            reader = csv.DictReader(rf)
            if not reader.fieldnames:
                raise SystemExit("CSV 无表头")
            if COL_SOURCE not in reader.fieldnames:
                raise SystemExit(f"未找到列: {COL_SOURCE}")

            fieldnames = merge_fieldnames(list(reader.fieldnames))

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8-sig",
                newline="",
                delete=False,
                suffix=".csv",
                prefix="compass_exp_",
                dir=out.parent,
            ) as wf:
                tmp_path = wf.name
                writer = csv.DictWriter(wf, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()

                for row in reader:
                    total += 1
                    exp = (row.get(COL_SOURCE) or "").strip()
                    items = split_items(exp)
                    if items:
                        non_empty += 1
                    tags = extract_tags(items)
                    if tags:
                        tag_non_empty += 1

                    row[COL_ITEMS] = "|".join(items)
                    row[COL_TAGS_ZH] = ";".join(tags)
                    row[COL_TAGS_EN] = ";".join(tags_to_en(tags))
                    writer.writerow(row)

        if tmp_path:
            replace_or_fallback(tmp_path, out)
            tmp_path = None
    except Exception:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
        raise

    print(f"完成 -> {out}")
    print(f"总行数: {total} | experience非空: {non_empty} | 命中标签: {tag_non_empty}")


if __name__ == "__main__":
    main()

