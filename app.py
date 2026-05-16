"""
app.py — AI Business Intelligence Platform
Main Streamlit application with 6 tabs covering all exam requirements.
"""

from __future__ import annotations

import io
import json
import os
import warnings
import time
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    APP_TITLE, APP_VERSION, DAGSHUB_USERNAME, DAGSHUB_REPO_NAME,
    MLFLOW_TRACKING_URI, DATASETS,
)
from logger import log_action, read_logs
from ml_pipeline import (
    load_and_preprocess, train_all_models,
    load_champion_model, predict_single, predict_batch,
    get_dataset_meta,
)
from gemini_ai import (
    analyse_dataset, interpret_prediction,
    summarise_batch_predictions, chat_about_data,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — dark industrial theme ───────────────────────────────────────
st.markdown("""
<style>
  /* Import fonts */
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* Root variables */
  :root {
    --bg:       #0d0f14;
    --surface:  #161920;
    --card:     #1c2030;
    --border:   #2a2f42;
    --accent:   #00d4ff;
    --accent2:  #7c3aed;
    --success:  #10b981;
    --warning:  #f59e0b;
    --danger:   #ef4444;
    --text:     #e2e8f0;
    --muted:    #64748b;
  }

  /* Global */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Space Mono', monospace;
    color: var(--accent) !important;
  }

  /* Main content blocks */
  .block-container { padding: 1.5rem 2rem !important; }

  /* Metric cards */
  [data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
  }
  [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .05em; }
  [data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Space Mono', monospace; font-size: 1.6rem !important; }

  /* Tabs */
  [data-testid="stTabs"] button {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    color: var(--muted) !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: transparent !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.03em;
    padding: 0.5rem 1.2rem !important;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }

  /* Inputs */
  .stTextInput > div > div > input,
  .stNumberInput > div > div > input,
  .stSelectbox > div > div,
  .stTextArea > div > textarea {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
  }

  /* DataFrames */
  [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }

  /* Expanders */
  .streamlit-expanderHeader {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
  }

  /* Hero header */
  .hero-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b27 50%, #0f172a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(0,212,255,0.06) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
  }
  .hero-sub {
    color: var(--muted);
    font-size: 0.9rem;
    margin-top: 0.4rem;
  }

  /* Section dividers */
  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    color: var(--accent);
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem;
  }

  /* Tag pills */
  .tag {
    display: inline-block;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    color: var(--accent);
    margin: 2px;
  }

  /* Info/success boxes */
  .info-box {
    background: rgba(0,212,255,0.05);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
  }
  .success-box {
    background: rgba(16,185,129,0.07);
    border-left: 3px solid var(--success);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
  }
  .warning-box {
    background: rgba(245,158,11,0.07);
    border-left: 3px solid var(--warning);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.6rem 0;
    font-size: 0.88rem;
  }

  /* Gemini response */
  .gemini-response {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-size: 0.9rem;
    line-height: 1.65;
    white-space: pre-wrap;
  }

  /* Progress bar color */
  [data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, var(--accent2), var(--accent)) !important;
  }

  /* Hide Streamlit branding */
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def ds_selector(key_prefix: str = "") -> str:
    """Sidebar dataset selector; returns dataset_key."""
    return st.sidebar.selectbox(
        "📂 Dataset",
        options=list(DATASETS.keys()),
        format_func=lambda k: DATASETS[k]["label"],
        key=f"ds_select_{key_prefix}",
    )


def hero(title: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="hero-header">'
        f'<p class="hero-title">{title}</p>'
        f'<p class="hero-sub">{sub}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section(label: str) -> None:
    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)


def info(msg: str) -> None:
    st.markdown(f'<div class="info-box">{msg}</div>', unsafe_allow_html=True)


def ok(msg: str) -> None:
    st.markdown(f'<div class="success-box">{msg}</div>', unsafe_allow_html=True)


def warn(msg: str) -> None:
    st.markdown(f'<div class="warning-box">{msg}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _cached_df_meta(dataset_key: str):
    return get_dataset_meta(dataset_key)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    st.sidebar.markdown(
        '<p style="font-family:Space Mono,monospace;font-size:1.1rem;'
        'background:linear-gradient(90deg,#00d4ff,#7c3aed);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
        '🧠 AI BIZ INTEL</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"v{APP_VERSION} · MLOps Platform")
    st.sidebar.divider()

    dagshub_url = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}"
    mlflow_url = f"{MLFLOW_TRACKING_URI}"
    st.sidebar.markdown("**🔗 Quick Links**")
    st.sidebar.markdown(f"[🐙 DagsHub Repo]({dagshub_url})")
    st.sidebar.markdown(f"[📊 MLflow Tracking]({dagshub_url}.mlflow)")
    st.sidebar.divider()
    st.sidebar.markdown("**📦 Datasets**")
    for k, v in DATASETS.items():
        task_icon = "🔵" if v["task"] == "classification" else "🟡"
        st.sidebar.caption(f"{task_icon} {v['label']}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Project presentation
# ══════════════════════════════════════════════════════════════════════════════

def tab_presentation() -> None:
    hero("AI Business Intelligence Platform",
         "End-to-end MLOps pipeline · Predictive analytics · Generative AI")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Datasets", len(DATASETS))
    col2.metric("ML Tasks", "2 (Clf + Reg)")
    col3.metric("MLOps Tool", "MLflow + DagsHub")
    col4.metric("Gen-AI", "Gemini 1.5 Flash")

    section("Problem Statement")
    st.markdown("""
Enterprises need a **unified intelligence layer** that transforms raw tabular data into actionable decisions.
This platform provides:

- 🎯 **Automated ML pipelines** — data cleaning, encoding, training, evaluation
- 🏆 **Champion model management** — best model auto-selected and registered
- 🤖 **Generative AI analysis** — natural language insights powered by Gemini
- 📊 **Interactive dashboards** — visual exploration of data and model performance
- 🚀 **Cloud deployment** — fully deployable on Streamlit Community Cloud
""")

    section("Architecture")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Data Layer**
- 4 curated datasets (2 classification, 2 regression)
- Auto cleaning: dedup, imputation, label encoding

**ML Layer**
- Random Forest · Logistic Regression · Linear Regression
- XGBoost · Gradient Boosting
- MLflow experiment tracking on DagsHub
        """)
    with col_b:
        st.markdown("""
**App Layer**
- Streamlit multi-tab interface (6 tabs)
- Single & batch CSV prediction
- JSONL action logging

**AI Layer**
- Google Gemini 1.5 Flash
- Dataset analysis, prediction explanations, Q&A

**DevOps Layer**
- Git/GitHub versioning
- Streamlit Community Cloud deployment
- `.env` / `st.secrets` secret management
        """)

    section("Datasets")
    for k, v in DATASETS.items():
        task_badge = "🔵 Classification" if v["task"] == "classification" else "🟡 Regression"
        with st.expander(f"{v['label']} — {task_badge}"):
            st.markdown(f"""
**Target:** `{v['target']}` &nbsp;|&nbsp; **Task:** {v['task']}

{v['description']}
            """)
            try:
                df_prev = pd.read_csv(v["path"], nrows=5)
                st.dataframe(df_prev, use_container_width=True, height=180)
            except Exception:
                warn("Dataset file not found. Please copy CSVs to the `data/` folder.")

    section("Technology Stack")
    tags = ["Python 3.11", "Streamlit", "MLflow", "DagsHub", "scikit-learn",
            "XGBoost", "Google Gemini", "Plotly", "Pandas", "Git/GitHub"]
    st.markdown(" ".join(f'<span class="tag">{t}</span>' for t in tags), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Training & MLflow tracking
# ══════════════════════════════════════════════════════════════════════════════

def tab_training() -> None:
    hero("ML Training & Experiment Tracking",
         "Train models, compare results, track with MLflow on DagsHub")

    dataset_key = ds_selector("train")
    cfg = DATASETS[dataset_key]

    col1, col2 = st.columns([2, 1])
    with col1:
        info(f"<b>{cfg['label']}</b> — {cfg['description']} · Task: <b>{cfg['task']}</b>")
    with col2:
        dagshub_url = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"
        st.markdown(f"[📊 Open MLflow UI ↗]({dagshub_url})", unsafe_allow_html=False)

    section("Launch Training")
    if st.button("🚀 Train & Track All Models", key="btn_train"):
        results_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_cb(frac: float, msg: str) -> None:
            progress_bar.progress(frac)
            status_text.markdown(f"⏳ {msg}")

        with st.spinner("Running experiments…"):
            try:
                results = train_all_models(dataset_key, progress_callback=progress_cb)
                st.session_state[f"results_{dataset_key}"] = results
                progress_bar.progress(1.0)
                status_text.markdown("✅ Training complete!")
            except Exception as e:
                st.error(f"Training error: {e}")
                return

    # Display results if available
    if f"results_{dataset_key}" in st.session_state:
        results: List[Dict] = st.session_state[f"results_{dataset_key}"]
        section("Experiment Results")

        rows = []
        for r in results:
            row = {"Model": r["model_name"], "Run ID": r["run_id"][:8] + "…"}
            row.update({k.upper(): round(v, 4) for k, v in r["metrics"].items()})
            rows.append(row)

        df_res = pd.DataFrame(rows)
        st.dataframe(df_res, use_container_width=True, hide_index=True)

        # Champion highlight
        task = cfg["task"]
        champion_metric = "ROC_AUC" if task == "classification" else "R2"
        if champion_metric in df_res.columns:
            best_idx = df_res[champion_metric].idxmax()
            best_model = df_res.loc[best_idx, "Model"]
            best_score = df_res.loc[best_idx, champion_metric]
            ok(f"🏆 <b>Champion:</b> {best_model} &nbsp;|&nbsp; {champion_metric}: {best_score:.4f} — registered in MLflow Model Registry")

        # Bar chart
        metric_cols = [c for c in df_res.columns if c not in ("Model", "Run ID")]
        if metric_cols:
            fig = go.Figure()
            for mc in metric_cols:
                fig.add_trace(go.Bar(
                    name=mc, x=df_res["Model"], y=df_res[mc],
                    marker_color=["#00d4ff", "#7c3aed", "#10b981", "#f59e0b"][metric_cols.index(mc) % 4],
                ))
            fig.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                legend_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#2a2f42"),
                yaxis=dict(gridcolor="#2a2f42"),
                title="Model Comparison",
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        warn("No training results yet. Click 'Train & Track All Models' to start.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Single prediction
# ══════════════════════════════════════════════════════════════════════════════

def tab_single_prediction() -> None:
    hero("Single Record Prediction",
         "Fill in feature values and get an instant AI-interpreted prediction")

    dataset_key = ds_selector("single")
    cfg = DATASETS[dataset_key]

    try:
        df, meta = _cached_df_meta(dataset_key)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    section("Input Features")
    st.caption(f"Target: `{meta['target']}` · Task: `{meta['task']}`")

    features = meta["features"]
    cat_cols = set(meta["cat_cols"])

    # Auto-generate form
    cols = st.columns(3)
    input_vals: Dict[str, Any] = {}
    for i, feat in enumerate(features):
        col = cols[i % 3]
        if feat in cat_cols:
            unique_vals = df[feat].dropna().unique().tolist()
            input_vals[feat] = col.selectbox(feat, options=unique_vals, key=f"inp_{feat}")
        else:
            lo = float(df[feat].min())
            hi = float(df[feat].max())
            med = float(df[feat].median())
            input_vals[feat] = col.number_input(
                feat, min_value=lo, max_value=hi, value=med,
                step=(hi - lo) / 100 if hi != lo else 1.0,
                key=f"inp_{feat}",
            )

    st.divider()
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        predict_clicked = st.button("⚡ Predict", key="btn_single_predict")

    if predict_clicked:
        with st.spinner("Loading champion model…"):
            model = load_champion_model(dataset_key)

        if model is None:
            warn("Champion model not found in MLflow Registry. Please train models first (Tab 2).")
            return

        with st.spinner("Computing prediction…"):
            result = predict_single(dataset_key, input_vals, model, meta)

        section("Prediction Result")
        pred = result["prediction"]
        if meta["task"] == "classification":
            label_map = {0: "✅ Negative", 1: "⚠️ Positive"}
            pred_label = label_map.get(int(pred), str(int(pred)))
            col_r1, col_r2 = st.columns(2)
            col_r1.metric("Prediction", pred_label)
            if "probability" in result:
                col_r2.metric("Confidence", f"{result['probability']:.1%}")
        else:
            st.metric(f"Predicted {meta['target']}", f"{pred:,.2f}")

        # Gemini interpretation
        section("🤖 Gemini Interpretation")
        with st.spinner("Asking Gemini…"):
            gemini_text = interpret_prediction(
                dataset_label=cfg["label"],
                task=meta["task"],
                target=meta["target"],
                input_data=input_vals,
                prediction_result=result,
            )
        st.markdown(f'<div class="gemini-response">{gemini_text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Batch CSV prediction
# ══════════════════════════════════════════════════════════════════════════════

def tab_batch_prediction() -> None:
    hero("Batch CSV Prediction",
         "Upload a CSV, run bulk predictions, download results with Gemini summary")

    dataset_key = ds_selector("batch")
    cfg = DATASETS[dataset_key]

    try:
        _, meta = _cached_df_meta(dataset_key)
    except Exception as e:
        st.error(f"Error loading meta: {e}")
        return

    section("Upload CSV")
    st.caption(f"Upload a CSV with the same columns as **{cfg['label']}** (target column optional).")

    uploaded = st.file_uploader("Choose CSV file", type=["csv"], key="batch_upload")

    if uploaded is not None:
        log_action("csv_upload", {"filename": uploaded.name, "dataset": dataset_key})
        df_upload = pd.read_csv(uploaded)
        st.dataframe(df_upload.head(10), use_container_width=True, height=250)
        info(f"📄 {len(df_upload)} rows · {df_upload.shape[1]} columns")

        if st.button("⚡ Run Batch Prediction", key="btn_batch_predict"):
            with st.spinner("Loading champion model…"):
                model = load_champion_model(dataset_key)

            if model is None:
                warn("Champion model not found. Please train models first.")
                return

            with st.spinner(f"Predicting {len(df_upload)} records…"):
                df_results = predict_batch(dataset_key, df_upload, model, meta)

            section("Prediction Results")
            st.dataframe(df_results, use_container_width=True, height=300)

            # Download button
            csv_bytes = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results CSV",
                data=csv_bytes,
                file_name=f"{dataset_key}_predictions.csv",
                mime="text/csv",
            )

            # Quick stats
            preds = df_results["prediction"]
            col1, col2, col3 = st.columns(3)
            if cfg["task"] == "classification":
                n_pos = int((preds == 1).sum())
                col1.metric("Total Records", len(df_results))
                col2.metric("Positive Predictions", n_pos)
                col3.metric("Rate", f"{n_pos / len(df_results):.1%}")
            else:
                col1.metric("Total Records", len(df_results))
                col2.metric("Mean Prediction", f"{preds.mean():,.2f}")
                col3.metric("Std Dev", f"{preds.std():,.2f}")

            # Gemini summary
            section("🤖 Gemini Batch Summary")
            with st.spinner("Generating AI summary…"):
                summary = summarise_batch_predictions(
                    cfg["label"], cfg["task"], meta["target"], df_results
                )
            st.markdown(f'<div class="gemini-response">{summary}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Gemini AI Analysis
# ══════════════════════════════════════════════════════════════════════════════

def tab_gemini() -> None:
    hero("Gemini AI Analysis",
         "Ask questions about your data in natural language · Powered by Google Gemini 1.5 Flash")

    dataset_key = ds_selector("gemini")
    cfg = DATASETS[dataset_key]

    try:
        df, meta = _cached_df_meta(dataset_key)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    # Auto analysis
    section("📊 Automated Dataset Analysis")
    if st.button("🔍 Generate AI Analysis", key="btn_auto_analysis"):
        with st.spinner("Gemini is analysing your dataset…"):
            analysis = analyse_dataset(
                df, cfg["label"], meta["target"], meta["task"]
            )
        st.session_state[f"analysis_{dataset_key}"] = analysis

    if f"analysis_{dataset_key}" in st.session_state:
        st.markdown(
            f'<div class="gemini-response">{st.session_state[f"analysis_{dataset_key}"]}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Free-form Q&A chat
    section("💬 Ask Anything About Your Data")
    chat_key = f"chat_{dataset_key}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    # Display history
    for turn in st.session_state[chat_key]:
        role = turn["role"]
        icon = "🧑" if role == "user" else "🤖"
        bg = "#1c2030" if role == "user" else "#161920"
        st.markdown(
            f'<div style="background:{bg};border:1px solid #2a2f42;'
            f'border-radius:10px;padding:0.8rem 1rem;margin:0.4rem 0;font-size:0.88rem;">'
            f'<b style="color:#00d4ff">{icon} {role.capitalize()}</b><br>{turn["content"]}</div>',
            unsafe_allow_html=True,
        )

    user_q = st.text_input(
        "Your question…",
        placeholder=f"e.g. 'What are the main drivers of {meta['target']}?'",
        key="gemini_chat_input",
    )
    col_send, col_clear = st.columns([1, 4])
    with col_send:
        send = st.button("Send ➤", key="btn_gemini_send")
    with col_clear:
        if st.button("🗑 Clear Chat", key="btn_gemini_clear"):
            st.session_state[chat_key] = []
            st.rerun()

    if send and user_q.strip():
        st.session_state[chat_key].append({"role": "user", "content": user_q})
        with st.spinner("Gemini is thinking…"):
            reply = chat_about_data(
                cfg["label"], df, meta["target"], meta["task"],
                st.session_state[chat_key], user_q,
            )
        st.session_state[chat_key].append({"role": "assistant", "content": reply})
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def tab_dashboard() -> None:
    hero("Dashboard & Visualisation",
         "Data distributions · Correlation heatmap · Feature explorer · Action logs")

    dataset_key = ds_selector("dash")
    cfg = DATASETS[dataset_key]

    try:
        df, meta = _cached_df_meta(dataset_key)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    PLOTLY_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        legend_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#2a2f42"),
        yaxis=dict(gridcolor="#2a2f42"),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    # ── Dataset overview metrics ───────────────────────────────────────────────
    section("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features", meta["n_features"])
    c3.metric("Target", meta["target"])
    c4.metric("Task", meta["task"].capitalize())

    st.dataframe(df.describe().round(2), use_container_width=True, height=200)

    # ── Target distribution ───────────────────────────────────────────────────
    section("Target Distribution")
    target = meta["target"]
    if cfg["task"] == "classification":
        vc = df[target].value_counts().reset_index()
        vc.columns = ["Class", "Count"]
        fig_target = px.pie(
            vc, values="Count", names="Class",
            color_discrete_sequence=["#00d4ff", "#7c3aed", "#10b981"],
        )
        fig_target.update_layout(**PLOTLY_LAYOUT)
    else:
        fig_target = px.histogram(
            df, x=target, nbins=40,
            color_discrete_sequence=["#00d4ff"],
        )
        fig_target.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_target, use_container_width=True)

    # ── Correlation heatmap ────────────────────────────────────────────────────
    section("Correlation Heatmap")
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr().round(2)
    fig_corr = px.imshow(
        corr,
        color_continuous_scale=[[0, "#7c3aed"], [0.5, "#161920"], [1, "#00d4ff"]],
        text_auto=True,
        aspect="auto",
    )
    fig_corr.update_layout(**PLOTLY_LAYOUT, title="Feature Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Feature distributions ─────────────────────────────────────────────────
    section("Feature Explorer")
    col_left, col_right = st.columns(2)
    with col_left:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feat_x = st.selectbox("X-axis feature", num_cols, key="feat_x")
    with col_right:
        feat_y = st.selectbox("Y-axis feature", [c for c in num_cols if c != feat_x], key="feat_y")

    color_col = target if df[target].nunique() < 10 else None
    fig_scatter = px.scatter(
        df, x=feat_x, y=feat_y, color=color_col,
        opacity=0.7,
        color_discrete_sequence=["#00d4ff", "#7c3aed", "#10b981", "#f59e0b"],
    )
    fig_scatter.update_layout(**PLOTLY_LAYOUT, title=f"{feat_x} vs {feat_y}")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Action Logs ────────────────────────────────────────────────────────────
    section("📋 Action Logs")
    logs = read_logs()
    if logs:
        df_logs = pd.DataFrame(logs)

        # Summary
        action_counts = df_logs["action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]
        fig_logs = px.bar(
            action_counts, x="Action", y="Count",
            color_discrete_sequence=["#7c3aed"],
        )
        fig_logs.update_layout(**PLOTLY_LAYOUT, title="Actions by Type")
        st.plotly_chart(fig_logs, use_container_width=True)

        # Table
        st.dataframe(
            df_logs.sort_values("timestamp", ascending=False).head(50),
            use_container_width=True,
            height=300,
            hide_index=True,
        )

        # Export
        csv_logs = df_logs.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Logs CSV",
            data=csv_logs,
            file_name="action_logs.csv",
            mime="text/csv",
        )
    else:
        warn("No actions logged yet. Start training models, making predictions, or using Gemini.")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    render_sidebar()

    tabs = st.tabs([
        "🏠 Présentation",
        "🧪 Training & MLflow",
        "⚡ Prédiction Unitaire",
        "📦 Prédiction Batch",
        "🤖 Gemini AI",
        "📊 Dashboard",
    ])

    with tabs[0]:
        tab_presentation()
    with tabs[1]:
        tab_training()
    with tabs[2]:
        tab_single_prediction()
    with tabs[3]:
        tab_batch_prediction()
    with tabs[4]:
        tab_gemini()
    with tabs[5]:
        tab_dashboard()


if __name__ == "__main__":
    main()
