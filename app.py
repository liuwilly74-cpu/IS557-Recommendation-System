# -*- coding: utf-8 -*-
"""
留学选校推荐系统 — Streamlit 前端

启动：
  pip install streamlit pandas numpy openai python-dotenv
  streamlit run app.py
  & D:/anaconda3/python.exe -m streamlit run "D:\IS 557\app.py"
"""

from __future__ import annotations

from env_loader import load_project_env

load_project_env()

import streamlit as st
import pandas as pd
import numpy as np

from recommendation_engine import (
    load_offers_csv,
    category_options,
    tag_options_from_data,
    recommend_top_k,
    find_supporting_case,
    generate_reason_stream,
    EmbeddingStore,
    DEFAULT_FEATURE_WEIGHTS,
    COL_SCHOOL,
    COL_PROG,
    COL_TARGET_CAT_ZH,
    COL_UNDER_CAT_ZH,
    COL_GPA,
    COL_IELTS,
    COL_TOEFL,
    COL_TAGS,
)

# ======================== 页面基础配置 ========================

st.set_page_config(
    page_title="留学选校推荐系统",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================== 全局缓存 ========================

@st.cache_resource(show_spinner="正在加载历史录取数据（仅首次）…")
def _load_data():
    df = load_offers_csv()
    target_opts, undergrad_opts = category_options(df)
    tag_pool = tag_options_from_data(df)
    return df, target_opts, undergrad_opts, tag_pool


@st.cache_resource(show_spinner="正在加载经历向量库…")
def _load_emb_store():
    store = EmbeddingStore()
    return store


df, target_opts, undergrad_opts, tag_pool = _load_data()
emb_store = _load_emb_store()

# ======================== 侧边栏：权重调节 ========================

st.sidebar.header("⚙️ 权重调节")
st.sidebar.caption("拖动滑块微调各维度在总分中的占比（实时归一化）")

w_raw: dict[str, float] = {}
label_map = {
    "undergrad_major": "本科专业匹配",
    "gpa": "GPA",
    "language": "语言成绩",
    "experience_tags": "经历标签 (Jaccard)",
    "experience_embed": "经历语义向量",
    "target_major": "目标专业匹配",
}
for key, default_val in DEFAULT_FEATURE_WEIGHTS.items():
    w_raw[key] = st.sidebar.slider(
        label_map.get(key, key),
        min_value=0.0,
        max_value=1.0,
        value=default_val,
        step=0.05,
        key=f"w_{key}",
    )

w_total = sum(w_raw.values()) or 1.0
weights = {k: v / w_total for k, v in w_raw.items()}

st.sidebar.divider()
st.sidebar.markdown("**归一化后权重**")
for k, v in weights.items():
    st.sidebar.text(f"  {label_map.get(k, k)}: {v:.0%}")

# ======================== 主区域标题 ========================

st.title("🎓 留学选校推荐系统")
st.markdown(
    "基于 **13 万+** 真实录取案例，通过 GPA、语言成绩、本科专业、经历标签 "
    "及 **经历语义向量** 六维加权匹配，为你找到最相似的成功录取案例。"
)
st.divider()

# ======================== 输入区 ========================

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📝 填写你的背景")

    target_major = st.selectbox(
        "目标录取专业大类",
        options=["（不限）"] + target_opts,
        index=0,
        help="选择你希望申请的专业大方向",
    )

    undergrad_major = st.selectbox(
        "本科专业大类",
        options=["（不限）"] + undergrad_opts,
        index=0,
        help="你的本科专业所属大类",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        gpa_input = st.text_input("GPA", placeholder="如 3.6 或 85", help="4 分制或百分制均可")
    with c2:
        ielts_input = st.text_input("IELTS", placeholder="如 7.0")
    with c3:
        toefl_input = st.text_input("TOEFL", placeholder="如 105")

    selected_tags = st.multiselect(
        "经历标签（可多选）",
        options=tag_pool,
        default=[],
        help="从已有标签中选择与你背景相关的",
    )

    exp_text = st.text_area(
        "个人经历描述（自由文本，用于语义向量匹配）",
        height=140,
        placeholder="例如：某券商投行部暑期实习，参与 IPO 项目尽调；校级科研课题，NLP 情感分析方向……",
        help="系统会将此段文字实时向量化，与历史案例做语义余弦相似度匹配",
    )

    top_k = st.slider("返回结果数量", min_value=5, max_value=50, value=15, step=5)

    run_btn = st.button("🚀 开始匹配", type="primary", use_container_width=True)

# ======================== 运行与结果展示 ========================

def _safe_float(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None

if run_btn:
    user_profile: dict = {
        "target_major_cat": target_major if target_major != "（不限）" else "",
        "undergrad_major_cat": undergrad_major if undergrad_major != "（不限）" else "",
        "gpa": _safe_float(gpa_input),
        "ielts": _safe_float(ielts_input),
        "toefl": _safe_float(toefl_input),
        "tags": set(selected_tags),
    }

    user_vec = None
    if exp_text.strip():
        with st.spinner("正在向量化你的经历文本…"):
            user_vec = emb_store.embed_user_text(exp_text)
        if user_vec is not None:
            st.success("经历文本向量化完成 ✓")
        else:
            st.warning("经历向量化未成功（API key 未配或网络异常），将跳过该维度")

    with st.spinner("正在匹配 13 万+ 案例…"):
        result, df_scored = recommend_top_k(
            df,
            user_profile,
            k=top_k,
            weights=weights,
            emb_store=emb_store,
            user_vec=user_vec,
        )

    if result.empty:
        st.warning("未找到匹配结果，请尝试放宽条件。")
    else:
        st.session_state["zh_result"] = result
        st.session_state["zh_df_scored"] = df_scored
        st.session_state["zh_user_profile"] = user_profile
        st.session_state["zh_exp_text"] = exp_text

with col_right:
    st.subheader("📊 匹配结果")

    if "zh_result" not in st.session_state:
        st.info("👈 填写信息后点击「开始匹配」")
        st.stop()

result = st.session_state["zh_result"]
df_scored = st.session_state["zh_df_scored"]

with col_right:
    rename_map = {
        "_match_score": "匹配度",
        COL_SCHOOL: "录取学校",
        COL_PROG: "录取专业",
        COL_TARGET_CAT_ZH: "录取大类",
        COL_UNDER_CAT_ZH: "本科大类",
        COL_GPA: "GPA",
        COL_IELTS: "IELTS",
        COL_TOEFL: "TOEFL",
        COL_TAGS: "经历标签",
        "_sub_target_major": "目标专业分",
        "_sub_undergrad_major": "本科专业分",
        "_sub_gpa": "GPA分",
        "_sub_language": "语言分",
        "_sub_experience_tags": "标签分",
        "_sub_experience_embed": "语义向量分",
    }

    orig_idx = result.index
    display = result.reset_index(drop=True).rename(
        columns={k: v for k, v in rename_map.items() if k in result.columns}
    )

    # 将个人经历列嵌入表格
    if "experience" in df.columns:
        exp_series = df.loc[orig_idx, "experience"].values
        display["个人经历"] = [
            str(e).strip().replace("\\n", "\n") if pd.notna(e) else ""
            for e in exp_series
        ]

    sub_cols = [v for k, v in rename_map.items() if k.startswith("_sub_") and v in display.columns]
    for c in sub_cols:
        display[c] = display[c].replace(-1.0, "—")

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "匹配度": st.column_config.ProgressColumn(
                "匹配度",
                format="%.1f",
                min_value=0,
                max_value=100,
            ),
            "个人经历": st.column_config.TextColumn(
                "个人经历",
                width="medium",
                help="点击单元格可查看完整内容",
            ),
        },
    )

    with st.expander("📐 各维度子分数说明"):
        st.markdown("""
| 维度 | 算法 | 满分条件 |
|------|------|----------|
| 目标专业分 | 大类完全一致 | 100 |
| 本科专业分 | 大类完全一致 | 100 |
| GPA分 | 归一化绝对差线性衰减 | 差值=0 |
| 语言分 | 等效托福分差线性衰减 | 差值=0 |
| 标签分 | Jaccard 集合相似度 × 100 | 完全一致 |
| 语义向量分 | 用户经历 embedding 与历史行条目平均向量的余弦相似度 × 100 | 方向完全一致 |

子分数为 **—** 表示该维度数据缺失（用户未填或历史行无数据），权重自动重分配到其他维度。
""")

# ======================== LLM 推荐理由 ========================

st.divider()
st.subheader("💡 为什么选这个项目？")

case_options = []
for rank, (idx, row) in enumerate(result.iterrows(), start=1):
    school = row.get(COL_SCHOOL, "")
    prog = row.get(COL_PROG, "")
    score = row.get("_match_score", 0)
    case_options.append(f"#{rank}  {school} — {prog}  ({score}分)")

selected_case_label = st.selectbox(
    "选择一个案例查看推荐理由",
    options=case_options,
    index=0,
    help="从上方匹配结果中选取一条，我们将解释为何该项目适合你",
)

gen_btn = st.button("✨ 总结推荐理由", type="secondary", use_container_width=True)

if gen_btn and selected_case_label:
    sel_rank = int(selected_case_label.split("#")[1].split()[0]) - 1
    primary_idx = result.index[sel_rank]
    primary_row = df_scored.loc[primary_idx]

    supporting = find_supporting_case(df_scored, primary_idx, threshold=15.0)
    if supporting is not None:
        sup_school = supporting.get(COL_SCHOOL, "")
        sup_prog = supporting.get(COL_PROG, "")
        sup_score = supporting.get("_match_score", 0)
        st.info(f"🔍 找到同项目辅助案例（{sup_score}分），将一并纳入分析")
        with st.expander(f"辅助案例详情：{sup_school} — {sup_prog}"):
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("GPA", supporting.get(COL_GPA, "N/A"))
            sc2.metric("IELTS", supporting.get(COL_IELTS, "N/A"))
            sc3.metric("TOEFL", supporting.get(COL_TOEFL, "N/A"))
            sup_tags = supporting.get(COL_TAGS, "")
            if sup_tags:
                st.markdown(f"**经历标签：** {sup_tags}")
            sup_idx = supporting.name if hasattr(supporting, "name") else None
            if sup_idx is not None and "experience" in df.columns:
                sup_exp = df.loc[sup_idx, "experience"]
                if pd.notna(sup_exp) and str(sup_exp).strip():
                    st.markdown(f"**个人经历：**\n\n{str(sup_exp).strip().replace(chr(92)+'n', chr(10))}")

    user_for_reason = st.session_state.get("zh_user_profile", {})
    exp_for_reason = st.session_state.get("zh_exp_text", "")

    try:
        stream = generate_reason_stream(
            user=user_for_reason,
            primary_row=primary_row,
            supporting_row=supporting,
            experience_text=exp_for_reason,
            lang="zh",
        )
        st.write_stream(stream)
    except Exception as e:
        st.error(f"总结失败：{e}")
