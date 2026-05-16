"""
config.py — Central configuration for AI Business Intelligence App
Reads from .env locally, and st.secrets on cloud deployment.
"""

import os

def _get_secret(key: str, default: str = "") -> str:
    """Try st.secrets first, then os.environ, then default."""
    try:
        import streamlit as st
        val = st.secrets.get(key, None)
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(key, default)


def load_env_file():
    """Load .env file if it exists (local dev only)."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


# Load .env on import
load_env_file()


# ── DagsHub / MLflow ──────────────────────────────────────────────────────────
DAGSHUB_USERNAME: str   = _get_secret("DAGSHUB_USERNAME", "YOUR_USERNAME")
DAGSHUB_TOKEN: str      = _get_secret("DAGSHUB_TOKEN",   "YOUR_TOKEN")
DAGSHUB_REPO_NAME: str  = _get_secret("DAGSHUB_REPO_NAME", "YOUR_REPO_NAME")

MLFLOW_TRACKING_URI: str = (
    f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"
)

# ── Gemini ────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = _get_secret("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")

# ── App ───────────────────────────────────────────────────────────────────────
APP_TITLE:       str = "AI Business Intelligence Platform"
APP_VERSION:     str = "1.0.0"
LOG_FILE:        str = "action_logs.jsonl"
CHAMPION_ALIAS:  str = "champion"

# ── Datasets meta ─────────────────────────────────────────────────────────────
DATASETS = {
    "fraud_detection": {
        "path": "data/fraud_detection_classification_realistic.csv",
        "target": "fraud",
        "task": "classification",
        "label": "Fraud Detection",
        "description": "Predict whether a financial transaction is fraudulent.",
    },
    "streaming_subscription": {
        "path": "data/streaming_subscription_classification_realistic.csv",
        "target": "subscription_cancelled",
        "task": "classification",
        "label": "Streaming Churn",
        "description": "Predict whether a streaming subscriber will cancel.",
    },
    "building_energy": {
        "path": "data/building_energy_regression_realistic.csv",
        "target": "energy_consumption",
        "task": "regression",
        "label": "Building Energy",
        "description": "Predict energy consumption of a building (kWh).",
    },
    "restaurant_revenue": {
        "path": "data/restaurant_revenue_regression_realistic.csv",
        "target": "monthly_revenue",
        "task": "regression",
        "label": "Restaurant Revenue",
        "description": "Predict monthly revenue of a restaurant (€).",
    },
}
