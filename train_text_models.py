import os
import pickle

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from logistic_model import evaluate_logistic, train_logistic


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "Text", "disaster_dataset.csv")
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def _resolve_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _safe_stratify(labels):
    counts = pd.Series(labels).value_counts()
    return labels if (counts >= 2).all() else None


def train_text_models(
    dataset_path=DEFAULT_DATASET_PATH,
    models_dir=DEFAULT_MODELS_DIR,
    max_features=3000,
    test_size=0.2,
    random_state=42,
):
    dataset_path = _resolve_path(dataset_path)
    models_dir = _resolve_path(models_dir)
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(dataset_path)
    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required_columns)}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.title()

    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df["text"])
    y = df["label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=_safe_stratify(y),
    )

    logistic_model = train_logistic(X_train, y_train)
    logistic_predictions = logistic_model.predict(X_test)
    logistic_accuracy = accuracy_score(y_test, logistic_predictions)
    evaluate_logistic(logistic_model, X_test, y_test)

    vectorizer_path = os.path.join(models_dir, "text_vectorizer.pkl")
    logistic_model_path = os.path.join(models_dir, "text_logistic_model.pkl")
    metrics_path = os.path.join(models_dir, "text_model_metrics.pkl")

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(logistic_model, logistic_model_path)
    joblib.dump(
        {
            "logistic_accuracy": float(logistic_accuracy),
            "dataset_path": dataset_path,
            "max_features": int(max_features),
            "test_size": float(test_size),
            "random_state": int(random_state),
        },
        metrics_path,
    )

    print("\n===== Saved Artifacts =====")
    print(f"Vectorizer: {vectorizer_path}")
    print(f"Logistic model: {logistic_model_path}")
    print(f"Metrics: {metrics_path}")

    return {
        "vectorizer_path": vectorizer_path,
        "logistic_model_path": logistic_model_path,
        "metrics_path": metrics_path,
        "logistic_accuracy": float(logistic_accuracy),
    }


if __name__ == "__main__":
    train_text_models()