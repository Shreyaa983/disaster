import io
import os
import pickle

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
import joblib

from model import CNN


app = Flask(__name__)
CORS(app)

CLASSES = ["Earthquake", "Fire", "Flood", "Normal"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_LOGISTIC_MODEL_PATH = os.environ.get(
    "TEXT_LOGISTIC_MODEL_PATH",
    os.environ.get("TEXT_MODEL_PATH", os.path.join("models", "text_logistic_model.pkl")),
)
TEXT_VECTORIZER_PATH = os.environ.get(
    "TEXT_VECTORIZER_PATH",
    os.path.join("models", "text_vectorizer.pkl"),
)
TEXT_METRICS_PATH = os.environ.get(
    "TEXT_METRICS_PATH",
    os.path.join("models", "text_model_metrics.pkl"),
)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

DISASTER_KEYWORDS = {
    "Earthquake": ["earthquake", "tremor", "shake", "quake", "seismic", "rupture", "fault", "magnitude"],
    "Fire": ["fire", "burn", "blaze", "flame", "smoke", "heat", "inferno", "wildfire", "burning"],
    "Flood": ["flood", "water", "inundation", "overflow", "rain", "wet", "submerged", "drown", "swamp"],
    "Normal": ["normal", "clear", "safe", "good", "fine", "ok", "okay", "nothing", "all"],
}


def resolve_existing_path(*candidate_paths):
    for candidate_path in candidate_paths:
        if candidate_path and os.path.exists(candidate_path):
            return candidate_path
    return candidate_paths[0] if candidate_paths else None


def load_cnn_model():
    model = CNN().to(device)
    model_path = os.path.join("models", "model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CNN model file not found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_resnet_model():
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    model = model.to(device)

    resnet_path = os.environ.get(
        "RESNET_MODEL_PATH",
        r"C:\Users\Shreya\OneDrive\Documents\Degree-Shreya\6th-Sem\New folder\disaster-detection\models\resnet18_disaster_best.pth",
    )
    if not os.path.exists(resnet_path):
        raise FileNotFoundError(f"ResNet model file not found at {resnet_path}")

    checkpoint = torch.load(resnet_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_text_assets():
    text_vectorizer_path = resolve_existing_path(
        TEXT_VECTORIZER_PATH,
        os.path.join("models", "text_vectorizer.pkl"),
        os.path.join("models", "vectorizer_tfidf.pkl"),
    )
    text_logistic_model_path = resolve_existing_path(
        TEXT_LOGISTIC_MODEL_PATH,
        os.path.join("models", "text_logistic_model.pkl"),
        os.path.join("models", "text_model_tfidf.pkl"),
    )
    text_metrics_path = resolve_existing_path(
        TEXT_METRICS_PATH,
        os.path.join("models", "text_model_metrics.pkl"),
    )

    if not os.path.exists(text_vectorizer_path):
        raise FileNotFoundError(f"Text vectorizer file not found at {text_vectorizer_path}")
    if not os.path.exists(text_logistic_model_path):
        raise FileNotFoundError(f"Text logistic model file not found at {text_logistic_model_path}")

    def load_serialized_object(file_path):
        try:
            return joblib.load(file_path)
        except Exception:
            with open(file_path, "rb") as file_handle:
                return pickle.load(file_handle)

    text_vectorizer = load_serialized_object(text_vectorizer_path)
    text_logistic_model = load_serialized_object(text_logistic_model_path)

    loaded_metrics = {}
    if text_metrics_path and os.path.exists(text_metrics_path):
        loaded_metrics = load_serialized_object(text_metrics_path)

    return {
        "vectorizer": text_vectorizer,
        "logistic_model": text_logistic_model,
        "metrics": loaded_metrics,
        "paths": {
            "vectorizer": text_vectorizer_path,
            "logistic_model": text_logistic_model_path,
            "metrics": text_metrics_path,
        },
    }


def normalize_text_label(label):
    return str(label).strip().title()


def classify_with_logistic_model(features, model):
    predicted_class = model.predict(features)[0]
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        predicted_class_index = list(model.classes_).index(predicted_class)
        confidence = float(probabilities[predicted_class_index])
    else:
        confidence = 1.0
    return normalize_text_label(predicted_class), confidence


def classify_text_with_tfidf(text):
    if text_vectorizer is None or text_logistic_model is None:
        raise RuntimeError("Text vectorizer and logistic model must be loaded")

    features = text_vectorizer.transform([text]).toarray()
    label, confidence = classify_with_logistic_model(features, text_logistic_model)
    candidates = {
        "logistic_regression": {
            "label": label,
            "confidence": confidence,
        }
    }
    return label, confidence, "logistic_regression", candidates


def ensemble_predict(image_tensor, cnn_weight=0.65, resnet_weight=0.35):
    with torch.no_grad():
        cnn_probs = torch.softmax(cnn_model(image_tensor), dim=1)
        resnet_probs = torch.softmax(resnet_model(image_tensor), dim=1)
        ensemble_probs = (cnn_weight * cnn_probs) + (resnet_weight * resnet_probs)

        confidence, predicted = torch.max(ensemble_probs, 1)
        cnn_confidence, cnn_predicted = torch.max(cnn_probs, 1)
        resnet_confidence, resnet_predicted = torch.max(resnet_probs, 1)

        return CLASSES[predicted.item()], confidence.item(), {
            "cnn_prediction": CLASSES[cnn_predicted.item()],
            "cnn_confidence": cnn_confidence.item(),
            "resnet_prediction": CLASSES[resnet_predicted.item()],
            "resnet_confidence": resnet_confidence.item(),
        }


def combine_predictions(image_pred, image_conf, text_pred):
    if image_pred is None or image_pred == "Not analyzed":
        return f"{text_pred} detected"
    if text_pred == "Normal" and image_pred != "Normal":
        return f"{image_pred} detected with high confidence"
    if image_pred == text_pred:
        return f"{image_pred} confirmed by both image and text analysis"
    if image_conf > 0.7:
        return f"{image_pred} detected (text analysis: {text_pred})"
    return f"Possible {image_pred} or {text_pred}"


def determine_priority(decision, user_level, image_conf=0):
    decision_lower = decision.lower()
    if any(word in decision_lower for word in ["earthquake", "fire", "flood"]):
        return "High" if image_conf > 0.8 or "confirmed" in decision_lower else "Medium"
    if user_level in ["Low", "Medium", "High"]:
        return user_level
    return "Medium"


try:
    cnn_model = load_cnn_model()
    print(f"CNN model loaded on {device}")
except Exception as error:
    cnn_model = None
    print(f"Error loading CNN model: {error}")

try:
    resnet_model = load_resnet_model()
    print(f"ResNet model loaded on {device}")
except Exception as error:
    resnet_model = None
    print(f"Error loading ResNet model: {error}")

try:
    text_assets = load_text_assets()
    text_vectorizer = text_assets["vectorizer"]
    text_logistic_model = text_assets["logistic_model"]
    text_metrics = text_assets["metrics"]
    text_model_paths = text_assets["paths"]
    print(
        "Text assets loaded: "
        f"vectorizer={text_model_paths['vectorizer']}, "
        f"logistic={text_model_paths['logistic_model']}"
    )
except Exception as error:
    text_vectorizer = None
    text_logistic_model = None
    text_metrics = {}
    text_model_paths = {
        "vectorizer": TEXT_VECTORIZER_PATH,
        "logistic_model": TEXT_LOGISTIC_MODEL_PATH,
        "metrics": TEXT_METRICS_PATH,
    }
    print(f"Error loading text model assets: {error}")

text_model_loaded = text_vectorizer is not None and text_logistic_model is not None


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if cnn_model is None or resnet_model is None:
            return jsonify({"error": "One or both models are not loaded"}), 500

        text = request.form.get("text", "").strip()
        location = request.form.get("location", "Not provided").strip()
        emergency_level = request.form.get("emergency_level", "").strip()
        image_file = request.files.get("image")

        if not text:
            return jsonify({"error": "Text description is required"}), 400

        image_prediction = "Not analyzed"
        image_confidence = 0.0
        model_details = {}

        if image_file and image_file.filename:
            image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
            image_prediction, image_confidence, model_details = ensemble_predict(image_tensor)

        text_classification, text_confidence, text_analysis_source, text_model_candidates = classify_text_with_tfidf(text)
        final_decision = combine_predictions(image_prediction, image_confidence, text_classification)
        priority_level = determine_priority(final_decision, emergency_level, image_confidence)

        response = {
            "image_prediction": image_prediction,
            "image_confidence": float(image_confidence),
            "text_keywords": [],
            "text_classification": text_classification,
            "text_confidence": float(text_confidence),
            "text_analysis_source": text_analysis_source,
            "text_model_candidates": text_model_candidates,
            "final_decision": final_decision,
            "priority_level": priority_level,
            "location": location,
            "text_model_loaded": text_model_loaded,
            "text_model_paths": text_model_paths,
            "text_model_metrics": text_metrics,
            "model_ensemble": model_details,
            "message": "Report processed successfully",
        }

        print(f"Prediction: {final_decision} | Priority: {priority_level}")
        return jsonify(response), 200
    except Exception as error:
        print(f"Error in predict endpoint: {error}")
        return jsonify({"error": f"Server error: {str(error)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "device": str(device),
        "cnn_loaded": cnn_model is not None,
        "resnet_loaded": resnet_model is not None,
        "text_model_loaded": text_model_loaded,
        "text_model_metrics": text_metrics,
    }), 200


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name": "Disaster Report System API",
        "version": "1.0",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
        },
    }), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    if cnn_model is None or resnet_model is None or not text_model_loaded:
        print("Failed to load one or more models. Cannot start server.")
        print(f"   CNN Model: {'Loaded' if cnn_model else 'Failed'}")
        print(f"   ResNet Model: {'Loaded' if resnet_model else 'Failed'}")
        print(f"   Text Vectorizer: {'Loaded' if text_vectorizer else 'Failed'}")
        print(f"   Logistic Text Model: {'Loaded' if text_logistic_model else 'Failed'}")
        raise SystemExit(1)

    print("\n" + "=" * 50)
    print("Disaster Report System - Backend API")
    print("=" * 50)
    print("Server running on: http://localhost:5000")
    print(f"Device: {device}")
    print(f"Classes: {', '.join(CLASSES)}")
    print("Models: CNN + ResNet18 (Ensemble)")
    print("Weights: 65% CNN + 35% ResNet18")
    print(f"Logistic text model: {text_model_paths['logistic_model']}")
    print(f"Text vectorizer: {text_model_paths['vectorizer']}")
    print("=" * 50 + "\n")

    app.run(debug=True, host="localhost", port=5000, use_reloader=False)
