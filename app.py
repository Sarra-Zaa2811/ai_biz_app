"""
app.py — Plateforme Intelligence Affaires IA
Application Streamlit multi-onglets avec MLflow, Gemini AI et tableaux de bord.
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

# ── Configuration de page ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plateforme IA Business Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé — thème premium bleu/violet ──────────────────────────────
st.markdown("""
<style>
  /* Import fonts */
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

  /* Variables CSS */
  :root {
    --bg:       #0a0f1f;
    --surface:  #111827;
    --card:     #1a2847;
    --border:   #2d3f5f;
    --accent:   #00d9ff;
    --accent2:  #7c3aed;
    --success:  #10b981;
    --warning:  #f59e0b;
    --danger:   #ef4444;
    --text:     #f0f5ff;
    --muted:    #8b92a9;
  }

  /* Global */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--surface) 0%, #0f1729 100%) !important;
    border-right: 1px solid var(--border);
  }
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Space Mono', monospace;
    color: var(--accent) !important;
  }

  /* Conteneur principal */
  .block-container { 
    padding: 2rem 2.5rem !important; 
    max-width: 1400px;
  }

  /* Cartes de métrique */
  [data-testid="stMetric"] {
    background: linear-gradient(135deg, var(--card), #1f3557);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 4px 15px rgba(0, 217, 255, 0.05);
    transition: all 0.3s ease;
  }
  [data-testid="stMetric"]:hover {
    border-color: var(--accent);
    box-shadow: 0 6px 25px rgba(0, 217, 255, 0.1);
  }
  [data-testid="stMetricLabel"] { 
    color: var(--muted) !important; 
    font-size: 0.75rem; 
    text-transform: uppercase; 
    letter-spacing: .08em;
    font-weight: 600;
  }
  [data-testid="stMetricValue"] { 
    color: var(--accent) !important; 
    font-family: 'Space Mono', monospace; 
    font-size: 1.8rem !important;
    font-weight: 700;
  }

  /* Onglets */
  [data-testid="stTabs"] button {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    color: var(--muted) !important;
    border-radius: 8px 8px 0 0 !important;
    border-bottom: 3px solid transparent !important;
    padding: 0.8rem 1.3rem !important;
    transition: all 0.3s ease;
    margin-right: 0.5rem;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 3px solid var(--accent) !important;
    background: linear-gradient(180deg, var(--card) 0%, transparent 100%) !important;
  }

  /* Boutons */
  .stButton > button {
    background: linear-gradient(135deg, var(--accent2) 0%, var(--accent) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600;
    letter-spacing: 0.02em;
    padding: 0.65rem 1.4rem !important;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
  }
  .stButton > button:hover { 
    opacity: 0.9;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4);
    transform: translateY(-2px);
  }

  /* Entrées texte */
  .stTextInput > div > div > input,
  .stNumberInput > div > div > input,
  .stSelectbox > div > div,
  .stTextArea > div > textarea {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-size: 0.9rem;
    transition: all 0.3s ease;
  }
  .stTextInput > div > div > input:focus,
  .stNumberInput > div > div > input:focus,
  .stSelectbox > div > div:focus,
  .stTextArea > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 12px rgba(0, 217, 255, 0.2);
  }

  /* DataFrames */
  [data-testid="stDataFrame"] { 
    border: 1px solid var(--border); 
    border-radius: 12px; 
    overflow: hidden;
  }

  /* Expanders */
  .streamlit-expanderHeader {
    background: linear-gradient(90deg, var(--card), #1f3557) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--accent) !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    transition: all 0.3s ease;
  }
  .streamlit-expanderHeader:hover {
    border-color: var(--accent) !important;
    background: linear-gradient(90deg, #1f3557, var(--card)) !important;
  }

  /* En-tête héro */
  .hero-header {
    background: linear-gradient(135deg, #1a2847 0%, #1f3557 50%, #2d3f5f 100%);
    border: 1.5px solid var(--border);
    border-radius: 18px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0, 217, 255, 0.08);
  }
  .hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(0,217,255,0.08) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .hero-sub {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.6rem;
    font-weight: 300;
  }

  /* En-têtes de section */
  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: var(--accent);
    text-transform: uppercase;
    border-bottom: 2px solid var(--border);
    padding-bottom: 0.6rem;
    margin: 2rem 0 1.2rem;
    font-weight: 700;
  }

  /* Tags */
  .tag {
    display: inline-block;
    background: linear-gradient(135deg, var(--card), #1f3557);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.4rem 0.9rem;
    font-size: 0.75rem;
    font-family: 'DM Sans', sans-serif;
    color: var(--accent);
    margin: 0.4rem;
    font-weight: 500;
    transition: all 0.3s ease;
  }
  .tag:hover {
    border-color: var(--accent);
    box-shadow: 0 0 12px rgba(0, 217, 255, 0.2);
  }

  /* Boîtes info/succès/avertissement */
  .info-box {
    background: linear-gradient(90deg, rgba(0,217,255,0.08), rgba(0,217,255,0.03));
    border-left: 4px solid var(--accent);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.3rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    border-right: 1px solid var(--border);
  }
  .success-box {
    background: linear-gradient(90deg, rgba(16,185,129,0.08), rgba(16,185,129,0.03));
    border-left: 4px solid var(--success);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.3rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    border-right: 1px solid var(--border);
  }
  .warning-box {
    background: linear-gradient(90deg, rgba(245,158,11,0.08), rgba(245,158,11,0.03));
    border-left: 4px solid var(--warning);
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.3rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    border-right: 1px solid var(--border);
  }

  /* Réponse Gemini */
  .gemini-response {
    background: linear-gradient(135deg, var(--card), #1f3557);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.7rem;
    font-size: 0.92rem;
    line-height: 1.75;
    white-space: pre-wrap;
    word-wrap: break-word;
    box-shadow: 0 4px 15px rgba(0, 217, 255, 0.05);
  }

  /* Barre de progression */
  [data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, var(--accent2), var(--accent)) !important;
  }

  /* Dividers */
  .stDivider {
    border-color: var(--border) !important;
  }

  /* Masquer branding Streamlit */
  #MainMenu, footer { visibility: hidden; }

  /* Spinner personnalisé */
  .stSpinner { color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ── Fonctions helpers ─────────────────────────────────────────────────────────

def ds_selector(key_prefix: str = "") -> str:
    """Sélecteur de dataset dans la barre latérale."""
    return st.sidebar.selectbox(
        "Ensemble de données",
        options=list(DATASETS.keys()),
        format_func=lambda k: DATASETS[k]["label"],
        key=f"ds_select_{key_prefix}",
    )


def hero(title: str, sub: str = "") -> None:
    """Affiche un en-tête héro."""
    st.markdown(
        f'<div class="hero-header">'
        f'<p class="hero-title">{title}</p>'
        f'<p class="hero-sub">{sub}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section(label: str) -> None:
    """Affiche un titre de section."""
    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)


def info(msg: str) -> None:
    """Affiche une boîte d'info."""
    st.markdown(f'<div class="info-box">{msg}</div>', unsafe_allow_html=True)


def ok(msg: str) -> None:
    """Affiche une boîte de succès."""
    st.markdown(f'<div class="success-box">{msg}</div>', unsafe_allow_html=True)


def warn(msg: str) -> None:
    """Affiche une boîte d'avertissement."""
    st.markdown(f'<div class="warning-box">{msg}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _cached_df_meta(dataset_key: str):
    """Cache les métadonnées du dataset."""
    return get_dataset_meta(dataset_key)


# ── Barre latérale ────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    """Rend la barre latérale avec logo et liens."""
    st.sidebar.markdown(
        '<p style="font-family:Space Mono,monospace;font-size:1.2rem;'
        'background:linear-gradient(90deg,#00d9ff,#7c3aed);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">'
        '🧠 Plateforme IA Business Intelligence</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"v{APP_VERSION} · Plateforme MLOps")
    st.sidebar.divider()

    dagshub_url = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}"
    mlflow_url = f"{MLFLOW_TRACKING_URI}"
    st.sidebar.markdown("**Liens rapides**")
    st.sidebar.markdown(f"[📊 DagsHub Repo]({dagshub_url})")
    st.sidebar.markdown(f"[📈 Suivi MLflow]({dagshub_url}.mlflow)")
    st.sidebar.divider()
    st.sidebar.markdown("**Ensembles de données**")
    for k, v in DATASETS.items():
        task_icon = "🔵" if v["task"] == "classification" else "🟡"
        st.sidebar.caption(f"{task_icon} {v['label']}")


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 1 — Entraînement & MLflow
# ══════════════════════════════════════════════════════════════════════════════

def tab_training() -> None:
    """Onglet d'entraînement des modèles et suivi MLflow."""
    hero("Entraînement ML & Suivi d'expériences",
         "Entraînez des modèles, comparez les résultats, tracez avec MLflow sur DagsHub")

    dataset_key = ds_selector("train")
    cfg = DATASETS[dataset_key]

    col1, col2 = st.columns([2, 1])
    with col1:
        info(f"<b>{cfg['label']}</b> — {cfg['description']} · Tâche: <b>{cfg['task']}</b>")
    with col2:
        dagshub_url = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"
        st.markdown(f"[🔗 Ouvrir MLflow]({dagshub_url})", unsafe_allow_html=False)

    section("Lancer l'entraînement")
    if st.button("🚀 Entraîner & Suivre tous les modèles", key="btn_train"):
        results_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_cb(frac: float, msg: str) -> None:
            progress_bar.progress(frac)
            status_text.markdown(f"**{msg}**")

        with st.spinner("Exécution des expériences…"):
            try:
                results = train_all_models(dataset_key, progress_callback=progress_cb)
                st.session_state[f"results_{dataset_key}"] = results
                progress_bar.progress(1.0)
                status_text.markdown("✅ **Entraînement terminé!**")
                time.sleep(1)
            except Exception as e:
                st.error(f"❌ Erreur d'entraînement: {e}")
                return

    # Afficher les résultats si disponibles
    if f"results_{dataset_key}" in st.session_state:
        results: List[Dict] = st.session_state[f"results_{dataset_key}"]
        section("Résultats des expériences")

        rows = []
        for r in results:
            row = {"Modèle": r["model_name"], "ID Exécution": r["run_id"][:8] + "…"}
            row.update({k.upper(): round(v, 4) for k, v in r["metrics"].items()})
            rows.append(row)

        df_res = pd.DataFrame(rows)
        st.dataframe(df_res, use_container_width=True, hide_index=True)

        # Mise en évidence du champion
        task = cfg["task"]
        champion_metric = "ROC_AUC" if task == "classification" else "R2"
        if champion_metric in df_res.columns:
            best_idx = df_res[champion_metric].idxmax()
            best_model = df_res.loc[best_idx, "Modèle"]
            best_score = df_res.loc[best_idx, champion_metric]
            ok(f"<b>🏆 Champion:</b> {best_model} &nbsp;|&nbsp; {champion_metric}: <b>{best_score:.4f}</b> — enregistré dans MLflow Model Registry")

        # Graphique comparatif
        metric_cols = [c for c in df_res.columns if c not in ("Modèle", "ID Exécution")]
        if metric_cols:
            fig = go.Figure()
            colors = ["#00d9ff", "#7c3aed", "#10b981", "#f59e0b"]
            for i, mc in enumerate(metric_cols):
                fig.add_trace(go.Bar(
                    name=mc, x=df_res["Modèle"], y=df_res[mc],
                    marker_color=colors[i % len(colors)],
                ))
            fig.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#f0f5ff",
                legend_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#2d3f5f"),
                yaxis=dict(gridcolor="#2d3f5f"),
                title="Comparaison des modèles",
                margin=dict(l=10, r=10, t=40, b=10),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        warn("Aucun résultat d'entraînement encore. Cliquez sur 'Entraîner & Suivre tous les modèles' pour commencer.")


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 2 — Prédiction unique
# ══════════════════════════════════════════════════════════════════════════════

def tab_single_prediction() -> None:
    """Onglet de prédiction sur un seul enregistrement."""
    hero("Prédiction d'enregistrement unique",
         "Remplissez les valeurs des caractéristiques et obtenez une prédiction instantanée interprétée par l'IA")

    dataset_key = ds_selector("single")
    cfg = DATASETS[dataset_key]

    try:
        df, meta = _cached_df_meta(dataset_key)
    except Exception as e:
        st.error(f"❌ Erreur de chargement du dataset: {e}")
        return

    section("Saisir les caractéristiques")
    st.caption(f"Cible: `{meta['target']}` · Tâche: `{meta['task']}`")

    features = meta["features"]
    cat_cols = set(meta["cat_cols"])

    # Génération auto du formulaire
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
        predict_clicked = st.button("🔮 Prédire", key="btn_single_predict")

    if predict_clicked:
        with st.spinner("Chargement du modèle champion…"):
            model = load_champion_model(dataset_key)

        if model is None:
            warn("Modèle champion introuvable dans MLflow Registry. Veuillez entraîner les modèles d'abord (Onglet 1).")
            return

        with st.spinner("Calcul de la prédiction…"):
            result = predict_single(dataset_key, input_vals, model, meta)

        section("Résultat de la prédiction")
        pred = result["prediction"]
        if meta["task"] == "classification":
            label_map = {0: "Négatif", 1: "Positif"}
            pred_label = label_map.get(int(pred), str(int(pred)))
            col_r1, col_r2 = st.columns(2)
            col_r1.metric("Prédiction", pred_label)
            if "probability" in result:
                col_r2.metric("Confiance", f"{result['probability']:.1%}")
        else:
            st.metric(f"Prédiction {meta['target']}", f"{pred:,.2f}")

        # Interprétation Gemini
        section("Interprétation Gemini")
        with st.spinner("Consultation de Gemini…"):
            gemini_text = interpret_prediction(
                dataset_label=cfg["label"],
                task=meta["task"],
                target=meta["target"],
                input_data=input_vals,
                prediction_result=result,
            )
        st.markdown(f'<div class="gemini-response">{gemini_text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 3 — Prédiction batch CSV
# ══════════════════════════════════════════════════════════════════════════════

def tab_batch_prediction() -> None:
    """Onglet de prédiction batch à partir d'un CSV."""
    hero("Prédiction batch CSV",
         "Téléchargez un CSV, lancez des prédictions en masse, téléchargez les résultats avec résumé Gemini")

    dataset_key = ds_selector("batch")
    cfg = DATASETS[dataset_key]

    try:
        _, meta = _cached_df_meta(dataset_key)
    except Exception as e:
        st.error(f"❌ Erreur de chargement des métadonnées: {e}")
        return

    section("Télécharger CSV")
    st.caption(f"Téléchargez un CSV avec les mêmes colonnes que **{cfg['label']}** (colonne cible optionnelle).")

    uploaded = st.file_uploader("Choisir fichier CSV", type=["csv"], key="batch_upload")

    if uploaded is not None:
        log_action("csv_upload", {"filename": uploaded.name, "dataset": dataset_key})
        df_upload = pd.read_csv(uploaded)
        st.dataframe(df_upload.head(10), use_container_width=True, height=250)
        info(f"📊 <b>{len(df_upload)}</b> lignes · <b>{df_upload.shape[1]}</b> colonnes")

        if st.button("🎯 Lancer prédiction batch", key="btn_batch_predict"):
            with st.spinner("Chargement du modèle champion…"):
                model = load_champion_model(dataset_key)

            if model is None:
                warn("Modèle champion introuvable. Veuillez entraîner les modèles d'abord.")
                return

            with st.spinner(f"Prédiction de {len(df_upload)} enregistrements…"):
                df_results = predict_batch(dataset_key, df_upload, model, meta)

            section("Résultats des prédictions")
            st.dataframe(df_results, use_container_width=True, height=300)

            # Bouton de téléchargement
            csv_bytes = df_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Télécharger résultats CSV",
                data=csv_bytes,
                file_name=f"{dataset_key}_predictions.csv",
                mime="text/csv",
            )

            # Statistiques rapides
            preds = df_results["prediction"]
            col1, col2, col3 = st.columns(3)
            if cfg["task"] == "classification":
                n_pos = int((preds == 1).sum())
                col1.metric("Enregistrements totaux", len(df_results))
                col2.metric("Prédictions positives", n_pos)
                col3.metric("Taux", f"{n_pos / len(df_results):.1%}")
            else:
                col1.metric("Enregistrements totaux", len(df_results))
                col2.metric("Prédiction moyenne", f"{preds.mean():,.2f}")
                col3.metric("Écart-type", f"{preds.std():,.2f}")

            # Résumé Gemini
            section("Résumé batch Gemini")
            with st.spinner("Génération du résumé IA…"):
                summary = summarise_batch_predictions(
                    cfg["label"], cfg["task"], meta["target"], df_results
                )
            st.markdown(f'<div class="gemini-response">{summary}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 4 — Analyse Gemini IA
# ══════════════════════════════════════════════════════════════════════════════

def tab_gemini() -> None:
    """Onglet d'analyse avec Gemini AI."""
    hero("Analyse Gemini IA",
         "Posez des questions sur vos données en langage naturel · Alimenté par Google Gemini 1.5 Flash")

    dataset_key = ds_selector("gemini")
    cfg = DATASETS[dataset_key]

    try:
        df, meta = _cached_df_meta(dataset_key)
    except Exception as e:
        st.error(f"❌ Erreur de chargement du dataset: {e}")
        return

    # Analyse automatique
    section("Analyse automatisée du dataset")
    if st.button("🤖 Générer analyse IA", key="btn_auto_analysis"):
        with st.spinner("Gemini analyse votre dataset…"):
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

    # Chat libre
    section("Posez vos questions sur vos données")
    chat_key = f"chat_{dataset_key}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    # Affichage historique
    for turn in st.session_state[chat_key]:
        role = turn["role"]
        icon = "👤" if role == "user" else "🤖"
        bg = "#1f3557" if role == "user" else "#1a2847"
        st.markdown(
            f'<div style="background:{bg};border:1px solid #2d3f5f;'
            f'border-radius:12px;padding:1rem 1.2rem;margin:0.6rem 0;font-size:0.9rem;">'
            f'<b style="color:#00d9ff">{icon} {role.capitalize()}</b><br><br>{turn["content"]}</div>',
            unsafe_allow_html=True,
        )

    user_q = st.text_input(
        "Votre question…",
        placeholder=f"Ex: 'Quels sont les principaux facteurs de {meta['target']}?'",
        key="gemini_chat_input",
    )
    col_send, col_clear = st.columns([1, 4])
    with col_send:
        send = st.button("📤 Envoyer", key="btn_gemini_send")
    with col_clear:
        if st.button("🗑 Effacer historique", key="btn_gemini_clear"):
            st.session_state[chat_key] = []
            st.rerun()

    if send and user_q.strip():
        st.session_state[chat_key].append({"role": "user", "content": user_q})
        with st.spinner("Gemini réfléchit…"):
            reply = chat_about_data(
                cfg["label"], df, meta["target"], meta["task"],
                st.session_state[chat_key], user_q,
            )
        st.session_state[chat_key].append({"role": "assistant", "content": reply})
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ONGLET 5 — Tableau de bord & Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def tab_dashboard() -> None:
    """Onglet de tableau de bord avec visualisations."""
    hero("Tableau de bord & Visualisations",
         "Distributions des données · Heatmap de corrélation · Explorateur de caractéristiques · Journaux d'actions")

    dataset_key = ds_selector("dash")
    cfg = DATASETS[dataset_key]

    try:
        df, meta = _cached_df_meta(dataset_key)
    except Exception as e:
        st.error(f"❌ Erreur de chargement du dataset: {e}")
        return

    PLOTLY_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f0f5ff",
        legend_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#2d3f5f"),
        yaxis=dict(gridcolor="#2d3f5f"),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
    )

    # Métriques de vue d'ensemble
    section("Vue d'ensemble du dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Lignes", f"{len(df):,}")
    c2.metric("📈 Caractéristiques", meta["n_features"])
    c3.metric("🎯 Cible", meta["target"])
    c4.metric("⚙️ Tâche", meta["task"].capitalize())

    st.dataframe(df.describe().round(2), use_container_width=True, height=200)

    # Distribution de la cible
    section("Distribution de la cible")
    target = meta["target"]
    if cfg["task"] == "classification":
        vc = df[target].value_counts().reset_index()
        vc.columns = ["Classe", "Nombre"]
        fig_target = px.pie(
            vc, values="Nombre", names="Classe",
            color_discrete_sequence=["#00d9ff", "#7c3aed", "#10b981"],
            title="Distribution des classes"
        )
        fig_target.update_layout(**PLOTLY_LAYOUT)
    else:
        fig_target = px.histogram(
            df, x=target, nbins=40,
            color_discrete_sequence=["#00d9ff"],
            title=f"Distribution de {target}"
        )
        fig_target.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_target, use_container_width=True)

    # Heatmap de corrélation
    section("Heatmap de corrélation")
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr().round(2)
    fig_corr = px.imshow(
        corr,
        color_continuous_scale=[[0, "#7c3aed"], [0.5, "#1a2847"], [1, "#00d9ff"]],
        text_auto=True,
        aspect="auto",
        title="Matrice de corrélation des caractéristiques"
    )
    fig_corr.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_corr, use_container_width=True)

    # Explorateur de caractéristiques
    section("Explorateur de caractéristiques")
    col_left, col_right = st.columns(2)
    with col_left:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feat_x = st.selectbox("Caractéristique X", num_cols, key="feat_x")
    with col_right:
        feat_y = st.selectbox("Caractéristique Y", [c for c in num_cols if c != feat_x], key="feat_y")

    color_col = target if df[target].nunique() < 10 else None
    fig_scatter = px.scatter(
        df, x=feat_x, y=feat_y, color=color_col,
        opacity=0.7,
        color_discrete_sequence=["#00d9ff", "#7c3aed", "#10b981", "#f59e0b"],
        title=f"{feat_x} vs {feat_y}"
    )
    fig_scatter.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Journaux d'actions
    section("Journaux d'actions")
    logs = read_logs()
    if logs:
        df_logs = pd.DataFrame(logs)

        # Résumé
        action_counts = df_logs["action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Nombre"]
        fig_logs = px.bar(
            action_counts, x="Action", y="Nombre",
            color_discrete_sequence=["#7c3aed"],
            title="Actions par type"
        )
        fig_logs.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_logs, use_container_width=True)

        # Tableau
        st.dataframe(
            df_logs.sort_values("timestamp", ascending=False).head(50),
            use_container_width=True,
            height=300,
            hide_index=True,
        )

        # Export
        csv_logs = df_logs.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Exporter journaux CSV",
            data=csv_logs,
            file_name="action_logs.csv",
            mime="text/csv",
        )
    else:
        warn("Aucune action enregistrée pour le moment. Commencez à entraîner des modèles, faire des prédictions ou utiliser Gemini.")



# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    render_sidebar()

    tabs = st.tabs([
        "Entraînement & MLflow",
        "Prédiction unique",
        "Prédiction batch",
        "Analyse Gemini",
        "Tableau de bord",
    ])

    with tabs[0]:
        tab_training()
    with tabs[1]:
        tab_single_prediction()
    with tabs[2]:
        tab_batch_prediction()
    with tabs[3]:
        tab_gemini()
    with tabs[4]:
        tab_dashboard()


if __name__ == "__main__":
    main()
