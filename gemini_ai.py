"""
gemini_ai.py — Intégration API Google Gemini pour plateforme IA Business Intelligence
Fournit analyse de datasets, interprétation de prédictions et résumés batch.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pandas as pd

from config import GOOGLE_API_KEY
from logger import log_action


def _get_client():
    """Retourne une instance GenerativeModel Gemini configurée."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        return genai.GenerativeModel("gemini-1.5-flash")
    except ImportError:
        raise RuntimeError(
            "google-generativeai n'est pas installé. "
            "Exécutez: pip install google-generativeai"
        )
    except Exception as e:
        raise RuntimeError(f"Échec d'initialisation de Gemini: {e}")


def _safe_generate(prompt: str, label: str = "gemini_query") -> str:
    """Appelle Gemini et retourne la réponse texte, avec gestion d'erreur."""
    try:
        model = _get_client()
        response = model.generate_content(prompt)
        text = response.text
        log_action("gemini_query", {"label": label, "prompt_length": len(prompt)})
        return text
    except Exception as e:
        log_action("gemini_query", {"label": label, "error": str(e)}, status="error")
        return f"⚠️ Erreur API Gemini: {e}"


# ── Analyse du dataset ────────────────────────────────────────────────────────

def analyse_dataset(
    df: pd.DataFrame,
    dataset_label: str,
    target: str,
    task: str,
    user_question: str = "",
) -> str:
    """
    Génère une analyse IA du dataset.
    Si user_question est fourni, répond à la question en contexte du dataset.
    """
    # Construire un résumé statistique compact
    n_rows, n_cols = df.shape
    numeric_stats = df.describe().round(2).to_string()
    null_counts = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_summary = {
        col: df[col].value_counts().head(5).to_dict() for col in cat_cols[:4]
    }
    target_summary = df[target].describe().round(2).to_dict() if task == "regression" else df[target].value_counts().to_dict()

    context = f"""
Vous êtes un scientifique des données senior analysant le dataset "{dataset_label}" pour une plateforme de business intelligence.

Vue d'ensemble du dataset:
- Dimensions: {n_rows} lignes × {n_cols} colonnes
- Type de tâche: {task}
- Variable cible: {target}
- Distribution cible: {json.dumps(target_summary, default=str)}
- Valeurs manquantes: {json.dumps(null_counts) if null_counts else "Aucune"}
- Colonnes catégoriques: {cat_cols}
- Exemples de valeurs catégoriques: {json.dumps(cat_summary, default=str)}

Statistiques numériques:
{numeric_stats}
""".strip()

    if user_question:
        prompt = f"""{context}

Question de l'utilisateur: {user_question}

Veuillez fournir une réponse claire, perspicace et orientée métier. Utilisez des puces où utile. Soyez concis mais complet (max 400 mots).
"""
    else:
        prompt = f"""{context}

Fournissez une analyse de business intelligence structurée couvrant:
1. **Vue d'ensemble du dataset** — ce que représentent les données et leur valeur métier
2. **Principaux motifs et insights** — distributions remarquables, corrélations ou anomalies
3. **Analyse de la variable cible** — équilibre des classes (classification) ou distribution (régression)
4. **Recommandations** — quels modèles ML pourraient fonctionner au mieux et pourquoi
5. **Implications métier** — quelles décisions pourraient être améliorées avec ce dataset

Gardez la réponse professionnelle, concise et actionnable (max 500 mots).
"""

    return _safe_generate(prompt, label=f"dataset_analysis_{dataset_label}")


# ── Interprétation de prédiction ──────────────────────────────────────────────

def interpret_prediction(
    dataset_label: str,
    task: str,
    target: str,
    input_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    model_name: str = "Modèle Champion",
) -> str:
    """Génère une explication en langage naturel d'une prédiction unique."""
    pred = prediction_result.get("prediction")
    prob = prediction_result.get("probability")
    class_probs = prediction_result.get("class_probabilities", {})

    if task == "classification":
        pred_line = f"Classe prédite: {int(pred)} (confiance: {prob:.1%})" if prob else f"Classe prédite: {int(pred)}"
        prob_detail = f"Probabilités des classes: {json.dumps({k: f'{v:.1%}' for k, v in class_probs.items()})}" if class_probs else ""
    else:
        pred_line = f"Prédiction {target}: {pred:,.2f}"
        prob_detail = ""

    prompt = f"""Vous êtes un assistant IA expliquant une prédiction machine learning à un utilisateur métier.

Dataset: {dataset_label}
Modèle utilisé: {model_name}
Tâche: {task}
Cible: {target}

Caractéristiques saisies:
{json.dumps(input_data, indent=2, default=str)}

Résultat de la prédiction:
- {pred_line}
{prob_detail}

Veuillez expliquer:
1. Ce que cette prédiction signifie en langage métier simple
2. Quelles caractéristiques d'entrée ont probablement eu le plus d'influence (basé sur leurs valeurs)
3. Quelle action le métier devrait envisager basé sur ce résultat
4. Toute mise en garde ou limitation à garder à l'esprit

Gardez-le concis et sans jargon (max 300 mots).
"""
    return _safe_generate(prompt, label=f"prediction_interpretation_{dataset_label}")


# ── Résumé batch ──────────────────────────────────────────────────────────────

def summarise_batch_predictions(
    dataset_label: str,
    task: str,
    target: str,
    df_results: pd.DataFrame,
) -> str:
    """Génère un résumé de haut niveau des résultats de prédiction batch."""
    n = len(df_results)
    preds = df_results["prediction"]

    if task == "classification":
        dist = preds.value_counts().to_dict()
        stats_str = f"Distribution des classes: {dist}"
        if "confidence" in df_results.columns:
            avg_conf = df_results["confidence"].mean()
            stats_str += f" | Confiance moyenne: {avg_conf:.1%}"
    else:
        stats_str = (
            f"Moyenne: {preds.mean():,.2f} | "
            f"Min: {preds.min():,.2f} | "
            f"Max: {preds.max():,.2f} | "
            f"Écart-type: {preds.std():,.2f}"
        )

    prompt = f"""Vous êtes un analyste de données résumant les prédictions batch ML pour un rapport métier.

Dataset: {dataset_label}
Tâche: {task}
Cible: {target}
Nombre total d'enregistrements traités: {n}
Statistiques de prédiction: {stats_str}

Veuillez fournir:
1. **Résumé exécutif** — ce que les résultats batch nous disent d'un coup d'œil
2. **Principales conclusions** — les chiffres les plus importants et leur signification
3. **Recommandations métier** — actions concrètes à entreprendre basées sur ces prédictions
4. **Zones à risque ou attention** — enregistrements ou motifs nécessitant un suivi

Gardez-le professionnel et actionnable (max 350 mots).
"""
    return _safe_generate(prompt, label=f"batch_summary_{dataset_label}")


# ── Chat libre sur le dataset ─────────────────────────────────────────────────

def chat_about_data(
    dataset_label: str,
    df: pd.DataFrame,
    target: str,
    task: str,
    conversation_history: list,
    user_message: str,
) -> str:
    """Répond à une question libre de l'utilisateur avec contexte du dataset."""
    sample = df.sample(min(5, len(df)), random_state=42).to_string(index=False)
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}

    history_text = ""
    for turn in conversation_history[-6:]:  # garder les 6 derniers tours pour le contexte
        role = turn.get("role", "user")
        content = turn.get("content", "")
        history_text += f"\n{role.upper()}: {content}"

    prompt = f"""Vous êtes un expert scientifique des données et analyste métier aidant un utilisateur à explorer le dataset "{dataset_label}".

Schéma du dataset: {json.dumps(schema)}
Cible: {target} ({task})
Exemples de lignes:
{sample}

Conversation jusqu'à présent:{history_text}

UTILISATEUR: {user_message}

Répondez utilement, en utilisant des exemples spécifiques au dataset le cas échéant. 
Si on vous demande de calculer des statistiques, raisonnez à partir de l'exemple et du schéma plutôt que d'inventer des nombres.
Soyez concis (max 350 mots).
"""
    return _safe_generate(prompt, label=f"chat_{dataset_label}")


# ── Dataset analysis ──────────────────────────────────────────────────────────

def analyse_dataset(
    df: pd.DataFrame,
    dataset_label: str,
    target: str,
    task: str,
    user_question: str = "",
) -> str:
    """
    Generate an AI analysis of a dataset.
    If user_question is provided, answer it in the context of the dataset.
    """
    # Build a compact statistical summary
    n_rows, n_cols = df.shape
    numeric_stats = df.describe().round(2).to_string()
    null_counts = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_summary = {
        col: df[col].value_counts().head(5).to_dict() for col in cat_cols[:4]
    }
    target_summary = df[target].describe().round(2).to_dict() if task == "regression" else df[target].value_counts().to_dict()

    context = f"""
You are a senior data scientist analysing the "{dataset_label}" dataset for a business intelligence platform.

Dataset overview:
- Shape: {n_rows} rows × {n_cols} columns
- Task type: {task}
- Target variable: {target}
- Target distribution: {json.dumps(target_summary, default=str)}
- Missing values: {json.dumps(null_counts) if null_counts else "None"}
- Categorical columns: {cat_cols}
- Categorical value samples: {json.dumps(cat_summary, default=str)}

Numeric statistics:
{numeric_stats}
""".strip()

    if user_question:
        prompt = f"""{context}

User question: {user_question}

Please provide a clear, insightful, and business-focused answer. Use bullet points where helpful. Be concise but complete (max 400 words).
"""
    else:
        prompt = f"""{context}

Provide a structured business intelligence analysis covering:
1. **Dataset overview** — what the data represents and its business value
2. **Key patterns & insights** — notable distributions, correlations, or anomalies
3. **Target variable analysis** — class balance (classification) or distribution (regression)
4. **Recommendations** — what ML models might work best and why
5. **Business implications** — what decisions can be improved with this dataset

Keep the response professional, concise, and actionable (max 500 words).
"""

    return _safe_generate(prompt, label=f"dataset_analysis_{dataset_label}")


# ── Prediction interpretation ─────────────────────────────────────────────────

def interpret_prediction(
    dataset_label: str,
    task: str,
    target: str,
    input_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    model_name: str = "Champion Model",
) -> str:
    """Generate a plain-language explanation of a single prediction."""
    pred = prediction_result.get("prediction")
    prob = prediction_result.get("probability")
    class_probs = prediction_result.get("class_probabilities", {})

    if task == "classification":
        pred_line = f"Predicted class: {int(pred)} (confidence: {prob:.1%})" if prob else f"Predicted class: {int(pred)}"
        prob_detail = f"Class probabilities: {json.dumps({k: f'{v:.1%}' for k, v in class_probs.items()})}" if class_probs else ""
    else:
        pred_line = f"Predicted {target}: {pred:,.2f}"
        prob_detail = ""

    prompt = f"""You are an AI assistant explaining a machine learning prediction to a business user.

Dataset: {dataset_label}
Model used: {model_name}
Task: {task}
Target: {target}

Input features provided:
{json.dumps(input_data, indent=2, default=str)}

Prediction result:
- {pred_line}
{prob_detail}

Please explain:
1. What this prediction means in plain business language
2. Which input features likely had the most influence (based on their values)
3. What action the business should consider based on this result
4. Any caveats or limitations to keep in mind

Keep it concise and jargon-free (max 300 words).
"""
    return _safe_generate(prompt, label=f"prediction_interpretation_{dataset_label}")


# ── Batch summary ─────────────────────────────────────────────────────────────

def summarise_batch_predictions(
    dataset_label: str,
    task: str,
    target: str,
    df_results: pd.DataFrame,
) -> str:
    """Generate a high-level summary of batch prediction results."""
    n = len(df_results)
    preds = df_results["prediction"]

    if task == "classification":
        dist = preds.value_counts().to_dict()
        stats_str = f"Class distribution: {dist}"
        if "confidence" in df_results.columns:
            avg_conf = df_results["confidence"].mean()
            stats_str += f" | Average confidence: {avg_conf:.1%}"
    else:
        stats_str = (
            f"Mean: {preds.mean():,.2f} | "
            f"Min: {preds.min():,.2f} | "
            f"Max: {preds.max():,.2f} | "
            f"Std: {preds.std():,.2f}"
        )

    prompt = f"""You are a data analyst summarising batch ML predictions for a business report.

Dataset: {dataset_label}
Task: {task}
Target: {target}
Total records processed: {n}
Prediction statistics: {stats_str}

Please provide:
1. **Executive summary** — what the batch results tell us at a glance
2. **Key findings** — the most important numbers and what they mean
3. **Business recommendations** — concrete actions to take based on these predictions
4. **Risk or attention areas** — records or patterns that need follow-up

Keep it professional and actionable (max 350 words).
"""
    return _safe_generate(prompt, label=f"batch_summary_{dataset_label}")


# ── Free-form chat about the dataset ─────────────────────────────────────────

def chat_about_data(
    dataset_label: str,
    df: pd.DataFrame,
    target: str,
    task: str,
    conversation_history: list,
    user_message: str,
) -> str:
    """Answer a free-form user question with dataset context."""
    sample = df.sample(min(5, len(df)), random_state=42).to_string(index=False)
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}

    history_text = ""
    for turn in conversation_history[-6:]:  # keep last 6 turns for context
        role = turn.get("role", "user")
        content = turn.get("content", "")
        history_text += f"\n{role.upper()}: {content}"

    prompt = f"""You are an expert data scientist and business analyst helping a user explore the "{dataset_label}" dataset.

Dataset schema: {json.dumps(schema)}
Target: {target} ({task})
Sample rows:
{sample}

Conversation so far:{history_text}

USER: {user_message}

Respond helpfully, using data-specific examples where relevant. 
If asked to compute statistics, reason from the sample and schema rather than making up numbers.
Be concise (max 350 words).
"""
    return _safe_generate(prompt, label=f"chat_{dataset_label}")
