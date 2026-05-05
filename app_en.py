# -*- coding: utf-8 -*-
"""
Study Abroad School Recommender — Streamlit UI (English)

Launch:
  pip install streamlit pandas numpy openai python-dotenv
  streamlit run app_en.py
  & D:/anaconda3/python.exe -m streamlit run "D:\IS 557\app_en.py"
"""

from __future__ import annotations

from env_loader import load_project_env

load_project_env()

import streamlit as st
import pandas as pd
import numpy as np

from recommendation_engine import (
    load_offers_csv,
    tag_options_from_data,
    recommend_top_k,
    find_supporting_case,
    generate_reason_stream,
    EmbeddingStore,
    DEFAULT_FEATURE_WEIGHTS,
    parse_tag_set,
    COL_SCHOOL,
    COL_PROG,
    COL_TARGET_CAT_ZH,
    COL_UNDER_CAT_ZH,
    COL_GPA,
    COL_IELTS,
    COL_TOEFL,
    COL_TAGS,
)

# English column names in the CSV
COL_TARGET_CAT_EN = "acceptance_category"
COL_UNDER_CAT_EN = "undergrad_category"
COL_TAGS_EN = "experience_tags_en"
COL_SCHOOL_EN = "acceptance"
COL_PROG_EN = "acceptance_program"

# ======================== Page config ========================

st.set_page_config(
    page_title="School Recommender",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================== Cached data ========================

@st.cache_resource(show_spinner="Loading admission data (first time only)…")
def _load_data():
    df = load_offers_csv()

    # --- English category options (from EN columns) ---
    target_en = sorted({
        str(x).strip() for x in df[COL_TARGET_CAT_EN].dropna().unique()
        if str(x).strip() and str(x).strip() != "nan"
    }) if COL_TARGET_CAT_EN in df.columns else []

    undergrad_en = sorted({
        str(x).strip() for x in df[COL_UNDER_CAT_EN].dropna().unique()
        if str(x).strip() and str(x).strip() != "nan"
    }) if COL_UNDER_CAT_EN in df.columns else []

    # --- EN -> ZH reverse mapping (built from data, row-level pairing) ---
    en2zh_target: dict[str, str] = {}
    en2zh_undergrad: dict[str, str] = {}
    if COL_TARGET_CAT_EN in df.columns and COL_TARGET_CAT_ZH in df.columns:
        for _, row in df[[COL_TARGET_CAT_EN, COL_TARGET_CAT_ZH]].drop_duplicates().iterrows():
            en = str(row[COL_TARGET_CAT_EN]).strip()
            zh = str(row[COL_TARGET_CAT_ZH]).strip()
            if en and zh and en != "nan" and zh != "nan":
                en2zh_target[en] = zh
    if COL_UNDER_CAT_EN in df.columns and COL_UNDER_CAT_ZH in df.columns:
        for _, row in df[[COL_UNDER_CAT_EN, COL_UNDER_CAT_ZH]].drop_duplicates().iterrows():
            en = str(row[COL_UNDER_CAT_EN]).strip()
            zh = str(row[COL_UNDER_CAT_ZH]).strip()
            if en and zh and en != "nan" and zh != "nan":
                en2zh_undergrad[en] = zh

    # --- English tag options (from experience_tags_en) ---
    tag_en_pool: list[str] = []
    en2zh_tags: dict[str, str] = {}
    if COL_TAGS_EN in df.columns and COL_TAGS in df.columns:
        freq: dict[str, int] = {}
        sample = min(len(df), 80000)
        for zh_cell, en_cell in zip(
            df[COL_TAGS].head(sample).values,
            df[COL_TAGS_EN].head(sample).values,
        ):
            zh_list = [t.strip() for t in str(zh_cell).split(";") if t.strip() and t.strip() != "nan"]
            en_list = [t.strip() for t in str(en_cell).split(";") if t.strip() and t.strip() != "nan"]
            for et in en_list:
                freq[et] = freq.get(et, 0) + 1
            if len(zh_list) == len(en_list):
                for zt, et in zip(zh_list, en_list):
                    en2zh_tags.setdefault(et, zt)
        top = sorted(freq.keys(), key=lambda x: -freq[x])[:200]
        tag_en_pool = sorted(top)

    return df, target_en, undergrad_en, tag_en_pool, en2zh_target, en2zh_undergrad, en2zh_tags


@st.cache_resource(show_spinner="Loading experience embedding store…")
def _load_emb_store():
    return EmbeddingStore()


df, target_en_opts, undergrad_en_opts, tag_en_pool, en2zh_target, en2zh_undergrad, en2zh_tags = _load_data()
emb_store = _load_emb_store()

# ======================== Sidebar: weight tuning ========================

st.sidebar.header("⚙️ Weight Tuning")
st.sidebar.caption("Drag sliders to adjust dimension weights (auto-normalized)")

w_raw: dict[str, float] = {}
label_map = {
    "undergrad_major": "Undergrad Major",
    "gpa": "GPA",
    "language": "Language Score",
    "experience_tags": "Experience Tags (Jaccard)",
    "experience_embed": "Experience Embedding",
    "target_major": "Target Major",
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
st.sidebar.markdown("**Normalized Weights**")
for k, v in weights.items():
    st.sidebar.text(f"  {label_map.get(k, k)}: {v:.0%}")

# ======================== Title ========================

st.title("🎓 Study Abroad School Recommender")
st.markdown(
    "Powered by **130,000+** real admission records. The system matches your profile "
    "against historical admits across **6 dimensions**: GPA, language scores, "
    "undergraduate major, experience tags, **semantic experience embeddings**, "
    "and target major preference."
)
st.divider()

# ======================== Input ========================

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📝 Your Profile")

    target_major_en = st.selectbox(
        "Target Admission Field",
        options=["(Any)"] + target_en_opts,
        index=0,
        help="The broad field you intend to apply for",
    )

    undergrad_major_en = st.selectbox(
        "Undergraduate Major Field",
        options=["(Any)"] + undergrad_en_opts,
        index=0,
        help="The broad category of your undergraduate major",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        gpa_input = st.text_input("GPA", placeholder="e.g. 3.6 or 85", help="4.0 scale or 100-point scale")
    with c2:
        ielts_input = st.text_input("IELTS", placeholder="e.g. 7.0")
    with c3:
        toefl_input = st.text_input("TOEFL", placeholder="e.g. 105")

    selected_tags_en = st.multiselect(
        "Experience Tags (multi-select)",
        options=tag_en_pool,
        default=[],
        help="Pick tags that match your background from the tag pool",
    )

    exp_text = st.text_area(
        "Personal Experience (free text, for semantic embedding matching)",
        height=140,
        placeholder="e.g. Summer intern at an investment bank, participated in IPO due diligence; "
                    "undergraduate research in NLP sentiment analysis…",
        help="This text will be embedded in real-time via DashScope API and matched "
             "against historical case vectors using cosine similarity",
    )

    top_k = st.slider("Number of results", min_value=5, max_value=50, value=15, step=5)

    run_btn = st.button("🚀 Find Matches", type="primary", use_container_width=True)

# ======================== Results ========================

def _safe_float(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None

if run_btn:
    target_zh = en2zh_target.get(target_major_en, "") if target_major_en != "(Any)" else ""
    undergrad_zh = en2zh_undergrad.get(undergrad_major_en, "") if undergrad_major_en != "(Any)" else ""
    tags_zh = {en2zh_tags[t] for t in selected_tags_en if t in en2zh_tags}

    user_profile: dict = {
        "target_major_cat": target_zh,
        "undergrad_major_cat": undergrad_zh,
        "gpa": _safe_float(gpa_input),
        "ielts": _safe_float(ielts_input),
        "toefl": _safe_float(toefl_input),
        "tags": tags_zh,
    }

    user_vec = None
    if exp_text.strip():
        with st.spinner("Embedding your experience text…"):
            user_vec = emb_store.embed_user_text(exp_text)
        if user_vec is not None:
            st.success("Experience text embedded successfully ✓")
        else:
            st.warning("Embedding failed (API key missing or network error). This dimension will be skipped.")

    with st.spinner("Matching against 130,000+ records…"):
        result, df_scored = recommend_top_k(
            df,
            user_profile,
            k=top_k,
            weights=weights,
            emb_store=emb_store,
            user_vec=user_vec,
        )

    if result.empty:
        st.warning("No matches found. Try relaxing your criteria.")
    else:
        st.session_state["en_result"] = result
        st.session_state["en_df_scored"] = df_scored
        st.session_state["en_user_profile"] = user_profile
        st.session_state["en_exp_text"] = exp_text

with col_right:
    st.subheader("📊 Matching Results")

    if "en_result" not in st.session_state:
        st.info("👈 Fill in your profile and click **Find Matches**")
        st.stop()

result = st.session_state["en_result"]
df_scored = st.session_state["en_df_scored"]

with col_right:
    orig_idx = result.index
    show = result.copy()

    if COL_SCHOOL_EN in df.columns:
        show["Admitted School"] = df.loc[orig_idx, COL_SCHOOL_EN].values
    if COL_PROG_EN in df.columns:
        show["Admitted Program"] = df.loc[orig_idx, COL_PROG_EN].values

    if COL_TARGET_CAT_EN in df.columns:
        show["Target Field"] = df.loc[orig_idx, COL_TARGET_CAT_EN].values
        show.drop(columns=[COL_TARGET_CAT_ZH], inplace=True, errors="ignore")
    if COL_UNDER_CAT_EN in df.columns:
        show["Undergrad Field"] = df.loc[orig_idx, COL_UNDER_CAT_EN].values
        show.drop(columns=[COL_UNDER_CAT_ZH], inplace=True, errors="ignore")

    if COL_TAGS_EN in df.columns and COL_TAGS in show.columns:
        show["Experience Tags"] = df.loc[orig_idx, COL_TAGS_EN].values
        show.drop(columns=[COL_TAGS], inplace=True, errors="ignore")

    # 将个人经历列嵌入表格
    if "experience" in df.columns:
        exp_series = df.loc[orig_idx, "experience"].values
        show["Experience"] = [
            str(e).strip().replace("\\n", "\n") if pd.notna(e) else ""
            for e in exp_series
        ]

    show = show.reset_index(drop=True)

    rename_map = {
        "_match_score": "Match %",
        COL_TARGET_CAT_ZH: "Target Field",
        COL_UNDER_CAT_ZH: "Undergrad Field",
        COL_GPA: "GPA",
        COL_IELTS: "IELTS",
        COL_TOEFL: "TOEFL",
        COL_TAGS: "Experience Tags",
        COL_SCHOOL: "School (CN)",
        COL_PROG: "Program (CN)",
        "_sub_target_major": "Sub: Target",
        "_sub_undergrad_major": "Sub: Undergrad",
        "_sub_gpa": "Sub: GPA",
        "_sub_language": "Sub: Language",
        "_sub_experience_tags": "Sub: Tags",
        "_sub_experience_embed": "Sub: Embedding",
    }

    display = show.rename(columns={k: v for k, v in rename_map.items() if k in show.columns})

    for drop_col in ["School (CN)", "Program (CN)"]:
        if drop_col in display.columns and "Admitted School" in display.columns:
            display.drop(columns=[drop_col], inplace=True, errors="ignore")

    sub_cols = [c for c in display.columns if c.startswith("Sub:")]
    for c in sub_cols:
        display[c] = display[c].replace(-1.0, "—")

    preferred_order = [
        "Match %", "Admitted School", "Admitted Program",
        "Target Field", "Undergrad Field",
        "GPA", "IELTS", "TOEFL", "Experience Tags", "Experience",
        "Sub: Target", "Sub: Undergrad", "Sub: GPA",
        "Sub: Language", "Sub: Tags", "Sub: Embedding",
    ]
    ordered = [c for c in preferred_order if c in display.columns]
    remaining = [c for c in display.columns if c not in ordered]
    display = display[ordered + remaining]

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Match %": st.column_config.ProgressColumn(
                "Match %",
                format="%.1f",
                min_value=0,
                max_value=100,
            ),
            "Experience": st.column_config.TextColumn(
                "Experience",
                width="medium",
                help="Click a cell to view full content",
            ),
        },
    )

    with st.expander("📐 Sub-score Explanation"):
        st.markdown("""
| Dimension | Algorithm | Full Score When |
|-----------|-----------|-----------------|
| Target Major | Exact category match | Categories identical |
| Undergrad Major | Exact category match | Categories identical |
| GPA | Normalized absolute difference, linear decay | Difference = 0 |
| Language | TOEFL-equivalent difference, linear decay | Difference = 0 |
| Tags (Jaccard) | Jaccard index × 100 | Tag sets identical |
| Embedding (cos) | Cosine similarity between user text embedding and historical row mean embedding × 100 | Vectors perfectly aligned |

A sub-score of **—** means the dimension had missing data (user left it blank or the
historical record lacks it). Its weight is automatically redistributed to other dimensions.
""")

# ======================== LLM Recommendation Reason ========================

st.divider()
st.subheader("💡 Why This Program?")

case_options_en = []
for rank, (idx, row) in enumerate(result.iterrows(), start=1):
    school_en = df.loc[idx, COL_SCHOOL_EN] if COL_SCHOOL_EN in df.columns else row.get(COL_SCHOOL, "")
    prog_en = df.loc[idx, COL_PROG_EN] if COL_PROG_EN in df.columns else row.get(COL_PROG, "")
    score = row.get("_match_score", 0)
    case_options_en.append(f"#{rank}  {school_en} — {prog_en}  ({score})")

selected_case_en = st.selectbox(
    "Select a case to see the recommendation reason",
    options=case_options_en,
    index=0,
    help="Pick a matched case above — we will explain why this program fits you",
)

gen_btn_en = st.button("✨ Summarize Recommendation Reason", type="secondary", use_container_width=True)

if gen_btn_en and selected_case_en:
    sel_rank = int(selected_case_en.split("#")[1].split()[0]) - 1
    primary_idx = result.index[sel_rank]
    primary_row = df_scored.loc[primary_idx]

    supporting = find_supporting_case(df_scored, primary_idx, threshold=15.0)
    if supporting is not None:
        sup_score = supporting.get("_match_score", 0)
        sup_idx = supporting.name if hasattr(supporting, "name") else None
        sup_school_en = df.loc[sup_idx, COL_SCHOOL_EN] if (sup_idx is not None and COL_SCHOOL_EN in df.columns) else supporting.get(COL_SCHOOL, "")
        sup_prog_en = df.loc[sup_idx, COL_PROG_EN] if (sup_idx is not None and COL_PROG_EN in df.columns) else supporting.get(COL_PROG, "")
        st.info(f"🔍 Found a supporting case ({sup_score} pts) — included in analysis")
        with st.expander(f"Supporting case: {sup_school_en} — {sup_prog_en}"):
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("GPA", supporting.get(COL_GPA, "N/A"))
            sc2.metric("IELTS", supporting.get(COL_IELTS, "N/A"))
            sc3.metric("TOEFL", supporting.get(COL_TOEFL, "N/A"))
            sup_tags = ""
            if sup_idx is not None and COL_TAGS_EN in df.columns:
                sup_tags = df.loc[sup_idx, COL_TAGS_EN]
            if not sup_tags or (isinstance(sup_tags, float) and pd.isna(sup_tags)):
                sup_tags = supporting.get(COL_TAGS, "")
            if sup_tags:
                st.markdown(f"**Experience Tags:** {sup_tags}")
            if sup_idx is not None and "experience" in df.columns:
                sup_exp = df.loc[sup_idx, "experience"]
                if pd.notna(sup_exp) and str(sup_exp).strip():
                    st.markdown(f"**Experience:**\n\n{str(sup_exp).strip().replace(chr(92)+'n', chr(10))}")

    user_for_reason = st.session_state.get("en_user_profile", {})
    exp_for_reason = st.session_state.get("en_exp_text", "")

    try:
        stream = generate_reason_stream(
            user=user_for_reason,
            primary_row=primary_row,
            supporting_row=supporting,
            experience_text=exp_for_reason,
            lang="en",
        )
        st.write_stream(stream)
    except Exception as e:
        st.error(f"Summary failed: {e}")
