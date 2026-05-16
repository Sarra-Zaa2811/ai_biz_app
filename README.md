# 🧠 AI Business Intelligence Platform

> End-to-end MLOps application — ML training, MLflow tracking, Gemini AI, cloud deployment.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org)
[![DagsHub](https://img.shields.io/badge/DagsHub-orange?style=flat)](https://dagshub.com)
[![Gemini](https://img.shields.io/badge/Gemini_AI-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev)

---

## 📋 Project Overview

This platform enables businesses to:
- Run **automated ML pipelines** on 4 real-world datasets
- Track every experiment with **MLflow on DagsHub**
- Auto-select and serve the **Champion model**
- Get **Gemini AI** insights on data and predictions
- Deploy publicly via **Streamlit Community Cloud**

---

## 📦 Datasets

| Dataset | Task | Target |
|---------|------|--------|
| Fraud Detection | Classification | `fraud` |
| Streaming Churn | Classification | `subscription_cancelled` |
| Building Energy | Regression | `energy_consumption` |
| Restaurant Revenue | Regression | `monthly_revenue` |

---

## 🚀 Quick Start (Local)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. Add your credentials
Edit `.env` with your real credentials:
```env
DAGSHUB_USERNAME=your_dagshub_username
DAGSHUB_TOKEN=your_dagshub_token
DAGSHUB_REPO_NAME=your_repo_name
GOOGLE_API_KEY=your_gemini_api_key
```

### 4. Copy datasets
```bash
mkdir -p data
cp /path/to/*.csv data/
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push this repo to GitHub (with `.env` and `*.pkl` in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your GitHub repo, branch `main`, file `app.py`
4. Under **Advanced settings → Secrets**, paste:
```toml
DAGSHUB_USERNAME  = "your_username"
DAGSHUB_TOKEN     = "your_token"
DAGSHUB_REPO_NAME = "your_repo"
GOOGLE_API_KEY    = "your_gemini_key"
```
5. Click **Deploy** ✅

---

## 🗂 Project Structure

```
├── app.py              # Main Streamlit application (6 tabs)
├── ml_pipeline.py      # ML training, MLflow tracking, champion management
├── gemini_ai.py        # Google Gemini API integration
├── config.py           # Central config (reads .env + st.secrets)
├── logger.py           # JSONL action logging
├── requirements.txt
├── .env                # Local secrets (⚠ never commit)
├── .gitignore
├── data/               # CSV datasets (copy here)
│   ├── fraud_detection_classification_realistic.csv
│   ├── streaming_subscription_classification_realistic.csv
│   ├── building_energy_regression_realistic.csv
│   └── restaurant_revenue_regression_realistic.csv
└── .streamlit/
    ├── config.toml     # Dark theme + server config
    └── secrets.toml    # Cloud secrets template (⚠ never commit)
```

---

## 🔬 MLOps Workflow

```
CSV data
  │
  ▼
load_and_preprocess()
  │  dedup · imputation · label encoding · 80/20 split
  ▼
train_all_models()
  │  RF · LR / LinReg · GBM · XGBoost
  │  → mlflow.log_params() + mlflow.log_metrics() + mlflow.sklearn.log_model()
  ▼
Champion selection (best ROC-AUC or R²)
  │  → mlflow.register_model() + set alias "champion"
  ▼
load_champion_model()   ← used by Tabs 3 & 4
```

---

## 🛠 Technologies

- **Python 3.11** — core language
- **Streamlit** — web interface
- **MLflow** — experiment tracking & model registry
- **DagsHub** — remote MLflow server & Git hosting
- **scikit-learn** — ML algorithms
- **XGBoost** — gradient boosting
- **Google Gemini 1.5 Flash** — generative AI
- **Plotly** — interactive charts
- **Pandas / NumPy** — data manipulation

---

## 📄 License

MIT License — for academic use.
