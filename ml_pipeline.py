"""
ml_pipeline.py — ML training, MLflow tracking, and Champion model management
Supports both classification and regression datasets.
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# ── scikit-learn ───────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── MLflow ────────────────────────────────────────────────────────────────────
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from config import (
    DAGSHUB_USERNAME, DAGSHUB_TOKEN, DAGSHUB_REPO_NAME,
    MLFLOW_TRACKING_URI, CHAMPION_ALIAS, DATASETS,
)
from logger import log_action


# ── MLflow auth setup ─────────────────────────────────────────────────────────

def setup_mlflow() -> None:
    """Configure MLflow to use DagsHub as remote tracking server."""
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ── Data loading & preprocessing ──────────────────────────────────────────────

def load_and_preprocess(
    dataset_key: str,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    pd.DataFrame, Dict[str, Any]
]:
    """
    Load CSV, clean it, encode categoricals, impute missing values,
    and return X_train, X_test, y_train, y_test plus the raw df and a meta dict.
    """
    cfg = DATASETS[dataset_key]
    df = pd.read_csv(cfg["path"])
    target = cfg["target"]
    task = cfg["task"]

    # ── Basic cleaning ─────────────────────────────────────────────────────────
    # Drop duplicates
    df = df.drop_duplicates()

    # Separate features and target
    X_raw = df.drop(columns=[target])
    y_raw = df[target].copy()

    # Identify column types
    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()

    # Encode categorical features
    encoders: Dict[str, LabelEncoder] = {}
    X_enc = X_raw.copy()
    for col in cat_cols:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col].astype(str))
        encoders[col] = le

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_enc)

    # Encode target if classification
    target_encoder: Optional[LabelEncoder] = None
    if task == "classification" and y_raw.dtype == object:
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw)
    else:
        y = y_raw.values

    # Train/test split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42,
        stratify=y if task == "classification" else None,
    )

    meta = {
        "task": task,
        "target": target,
        "features": list(X_enc.columns),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "encoders": encoders,
        "imputer": imputer,
        "target_encoder": target_encoder,
        "n_samples": len(df),
        "n_features": X_imputed.shape[1],
    }

    return X_train, X_test, y_train, y_test, df, meta


# ── Model catalogue ───────────────────────────────────────────────────────────

def _get_models(task: str) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    if task == "classification":
        models["Random Forest"] = RandomForestClassifier(n_estimators=100, random_state=42)
        models["Logistic Regression"] = LogisticRegression(max_iter=1000, random_state=42)
        models["Gradient Boosting"] = GradientBoostingClassifier(n_estimators=100, random_state=42)
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBClassifier(
                n_estimators=100, random_state=42,
                use_label_encoder=False, eval_metric="logloss", verbosity=0,
            )
    else:  # regression
        models["Random Forest"] = RandomForestRegressor(n_estimators=100, random_state=42)
        models["Linear Regression"] = LinearRegression()
        models["Gradient Boosting"] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    return models


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(task: str, y_true, y_pred, y_prob=None) -> Dict[str, float]:
    if task == "classification":
        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        if y_prob is not None:
            try:
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                else:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
            except Exception:
                pass
        return metrics
    else:
        mse = mean_squared_error(y_true, y_pred)
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_true, y_pred)),
        }


def _champion_metric(task: str) -> str:
    return "roc_auc" if task == "classification" else "r2"


def _higher_is_better(task: str) -> bool:
    return True  # both roc_auc and r2 → higher is better


# ── Training loop ─────────────────────────────────────────────────────────────

def train_all_models(
    dataset_key: str,
    progress_callback=None,
) -> List[Dict[str, Any]]:
    """
    Train all models for a dataset, track every run with MLflow,
    register the champion, and return a list of result dicts.
    """
    setup_mlflow()

    X_train, X_test, y_train, y_test, df, meta = load_and_preprocess(dataset_key)
    task = meta["task"]
    experiment_name = f"ai_biz_app_{dataset_key}"

    mlflow.set_experiment(experiment_name)

    models = _get_models(task)
    results: List[Dict[str, Any]] = []
    total = len(models)

    for idx, (model_name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(idx / total, f"Training {model_name}…")

        with mlflow.start_run(run_name=model_name):
            # ── Fit ──────────────────────────────────────────────────────────
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = None
            if task == "classification" and hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)
                except Exception:
                    pass

            metrics = _compute_metrics(task, y_test, y_pred, y_prob)

            # ── Log params ───────────────────────────────────────────────────
            params = {
                "model": model_name,
                "dataset": dataset_key,
                "task": task,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features": meta["n_features"],
            }
            if hasattr(model, "get_params"):
                raw_params = model.get_params()
                for k, v in raw_params.items():
                    if v is not None:
                        params[f"model_{k}"] = str(v)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            # ── Log model artifact ───────────────────────────────────────────
            mlflow.sklearn.log_model(model, artifact_path="model")
            run_id = mlflow.active_run().info.run_id

        result = {
            "model_name": model_name,
            "run_id": run_id,
            "metrics": metrics,
            "model_obj": model,
        }
        results.append(result)

    # ── Select champion ────────────────────────────────────────────────────────
    champ_metric = _champion_metric(task)
    best = max(results, key=lambda r: r["metrics"].get(champ_metric, float("-inf")))

    _register_champion(best, dataset_key, experiment_name)

    if progress_callback:
        progress_callback(1.0, f"Champion: {best['model_name']}")

    log_action(
        "training_run",
        {
            "dataset": dataset_key,
            "champion": best["model_name"],
            "champion_metric": champ_metric,
            "champion_value": best["metrics"].get(champ_metric),
            "models_trained": [r["model_name"] for r in results],
        },
    )

    return results


def _register_champion(
    best: Dict[str, Any],
    dataset_key: str,
    experiment_name: str,
) -> None:
    """Register the best run as 'champion' in the MLflow Model Registry."""
    client = MlflowClient()
    model_uri = f"runs:/{best['run_id']}/model"
    registered_name = f"{dataset_key}_champion"

    try:
        mlflow.register_model(model_uri=model_uri, name=registered_name)
        # Set alias to "champion" (MLflow ≥ 2.x)
        try:
            versions = client.get_latest_versions(registered_name)
            if versions:
                latest_version = versions[-1].version
                client.set_registered_model_alias(
                    name=registered_name,
                    alias=CHAMPION_ALIAS,
                    version=latest_version,
                )
        except Exception:
            pass
    except Exception as e:
        # Registry may not be supported on all DagsHub tiers — log but continue
        print(f"[ml_pipeline] Model registry note: {e}")


# ── Load champion for inference ───────────────────────────────────────────────

def load_champion_model(dataset_key: str) -> Optional[Any]:
    """
    Try to load the champion model from MLflow Model Registry.
    Falls back gracefully if unavailable.
    """
    setup_mlflow()
    registered_name = f"{dataset_key}_champion"
    try:
        model = mlflow.sklearn.load_model(f"models:/{registered_name}@{CHAMPION_ALIAS}")
        return model
    except Exception:
        pass
    # Fallback: latest version
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(registered_name)
        if versions:
            latest = versions[-1]
            model = mlflow.sklearn.load_model(f"runs:/{latest.run_id}/model")
            return model
    except Exception:
        pass
    return None


# ── Single & Batch prediction ─────────────────────────────────────────────────

def predict_single(
    dataset_key: str,
    input_dict: Dict[str, Any],
    model: Any,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Run inference on a single input dict, return prediction + proba."""
    features = meta["features"]
    row = []
    for col in features:
        val = input_dict.get(col, 0)
        if col in meta["cat_cols"]:
            le = meta["encoders"].get(col)
            if le is not None:
                try:
                    val = int(le.transform([str(val)])[0])
                except Exception:
                    val = 0
            else:
                val = 0
        row.append(float(val) if val != "" else 0.0)

    X = np.array(row).reshape(1, -1)
    X = meta["imputer"].transform(X)

    pred = model.predict(X)[0]
    result: Dict[str, Any] = {"prediction": float(pred)}

    if meta["task"] == "classification" and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
            result["probability"] = float(max(proba))
            result["class_probabilities"] = {
                str(cls): float(p) for cls, p in zip(model.classes_, proba)
            }
        except Exception:
            pass

    log_action(
        "single_prediction",
        {"dataset": dataset_key, "prediction": result.get("prediction")},
    )
    return result


def predict_batch(
    dataset_key: str,
    df_input: pd.DataFrame,
    model: Any,
    meta: Dict[str, Any],
) -> pd.DataFrame:
    """Run inference on a DataFrame, return it with predictions appended."""
    features = meta["features"]
    df_enc = df_input.copy()

    for col in meta["cat_cols"]:
        if col in df_enc.columns:
            le = meta["encoders"].get(col)
            if le is not None:
                df_enc[col] = df_enc[col].astype(str).apply(
                    lambda x: int(le.transform([x])[0]) if x in le.classes_ else 0
                )

    # Keep only known feature columns; fill missing with 0
    for col in features:
        if col not in df_enc.columns:
            df_enc[col] = 0

    X = df_enc[features].values.astype(float)
    X = meta["imputer"].transform(X)
    preds = model.predict(X)
    df_out = df_input.copy()
    df_out["prediction"] = preds

    if meta["task"] == "classification" and hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X)
            df_out["confidence"] = probas.max(axis=1)
        except Exception:
            pass

    log_action(
        "batch_prediction",
        {"dataset": dataset_key, "n_rows": len(df_out)},
    )
    return df_out


# ── Get meta without full training (for form generation) ─────────────────────

def get_dataset_meta(dataset_key: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return raw df and meta dict for a dataset (no training)."""
    _, _, _, _, df, meta = load_and_preprocess(dataset_key)
    return df, meta
