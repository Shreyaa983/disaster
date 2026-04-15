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

TEXT_MODEL_PATH = os.environ.get(
    "TEXT_MODEL_PATH",
    os.path.join("models", "Disaster_tfidf.pkl"),
)
TEXT_VECTORIZER_PATH = os.environ.get(
    "TEXT_VECTORIZER_PATH",
    os.path.join("models", "Vectorizer_tfidf.pkl"),
)
TEXT_CLASSES_PATH = os.environ.get(
    "TEXT_CLASSES_PATH",
    os.path.join("models", "Text_claseses.pkl"),
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
    if not os.path.exists(TEXT_MODEL_PATH):
        raise FileNotFoundError(f"Text model file not found at {TEXT_MODEL_PATH}")
    if not os.path.exists(TEXT_VECTORIZER_PATH):
        raise FileNotFoundError(f"Text vectorizer file not found at {TEXT_VECTORIZER_PATH}")
    if not os.path.exists(TEXT_CLASSES_PATH):
        raise FileNotFoundError(f"Text classes file not found at {TEXT_CLASSES_PATH}")

    def load_serialized_object(file_path):
        try:
            return joblib.load(file_path)
        except Exception:
            with open(file_path, "rb") as file_handle:
                return pickle.load(file_handle)

    text_model = load_serialized_object(TEXT_MODEL_PATH)
    text_vectorizer = load_serialized_object(TEXT_VECTORIZER_PATH)
    loaded_classes = load_serialized_object(TEXT_CLASSES_PATH)

    if hasattr(loaded_classes, "classes_"):
        text_classes = list(loaded_classes.classes_)
    elif isinstance(loaded_classes, dict) and "classes" in loaded_classes:
        text_classes = list(loaded_classes["classes"])
    elif isinstance(loaded_classes, (list, tuple)):
        text_classes = list(loaded_classes)
    else:
        text_classes = CLASSES

    return text_model, text_vectorizer, text_classes


def get_estimator_feature_count(estimator):
    if estimator is None:
        return None
    if hasattr(estimator, "n_features_in_"):
        return int(estimator.n_features_in_)
    if hasattr(estimator, "coef_"):
        return int(estimator.coef_.shape[1])
    return None


def get_vectorizer_feature_count(vectorizer):
    if vectorizer is None:
        return None
    if hasattr(vectorizer, "vocabulary_"):
        return len(vectorizer.vocabulary_)
    if hasattr(vectorizer, "get_feature_names_out"):
        return len(vectorizer.get_feature_names_out())
    return None


def extract_keywords(text):
    text_lower = text.lower()
    found = set()
    for keywords in DISASTER_KEYWORDS.values():
        for keyword in keywords:
            if keyword in text_lower:
                found.add(keyword)
    return list(found)


def classify_text_with_tfidf(text):
    if not text_model_compatible:
        return classify_text_fallback(text)

    try:
        features = text_vectorizer.transform([text])

        if hasattr(text_model, "predict_proba"):
            probabilities = text_model.predict_proba(features)[0]
            predicted_index = int(probabilities.argmax())
            confidence = float(probabilities[predicted_index])
        else:
            predicted_index = int(text_model.predict(features)[0])
            confidence = 1.0

        predicted_class = text_classes[predicted_index] if predicted_index < len(text_classes) else CLASSES[predicted_index]
        return predicted_class, confidence
    except Exception as error:
        print(f"⚠️ Text model inference failed, using fallback keywords: {error}")
        return classify_text_fallback(text)


def classify_text_fallback(text):
    text_lower = text.lower()
    scores = {
        category: sum(1 for keyword in keywords if keyword in text_lower)
        for category, keywords in DISASTER_KEYWORDS.items()
    }
    best_category = max(scores, key=scores.get)
    if scores[best_category] == 0:
        return "Unknown", 0.0
    return best_category, float(scores[best_category]) / 5.0


def ensemble_predict(image_tensor, cnn_weight=0.35, resnet_weight=0.65):
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
    print(f"✅ CNN model loaded on {device}")
except Exception as error:
    cnn_model = None
    print(f"❌ Error loading CNN model: {error}")

try:
    resnet_model = load_resnet_model()
    print(f"✅ ResNet model loaded on {device}")
except Exception as error:
    resnet_model = None
    print(f"❌ Error loading ResNet model: {error}")

try:
    text_model, text_vectorizer, text_classes = load_text_assets()
    print(f"✅ Text model loaded from {TEXT_MODEL_PATH}")
except Exception as error:
    text_model = None
    text_vectorizer = None
    text_classes = CLASSES
    print(f"❌ Error loading text model assets: {error}")

text_model_expected_features = get_estimator_feature_count(text_model)
text_vectorizer_features = get_vectorizer_feature_count(text_vectorizer)
text_model_compatible = (
    text_model is not None
    and text_vectorizer is not None
    and text_model_expected_features is not None
    and text_vectorizer_features is not None
    and text_model_expected_features == text_vectorizer_features
)

if text_model is not None and text_vectorizer is not None and not text_model_compatible:
    print(
        "⚠️ Text model/vectorizer feature mismatch: "
        f"model expects {text_model_expected_features}, vectorizer produces {text_vectorizer_features}. "
        "Falling back to keyword-based text classification."
    )


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

        text_keywords = extract_keywords(text)
        text_classification, text_confidence = classify_text_with_tfidf(text)
        final_decision = combine_predictions(image_prediction, image_confidence, text_classification)
        priority_level = determine_priority(final_decision, emergency_level, image_confidence)

        response = {
            "image_prediction": image_prediction,
            "image_confidence": float(image_confidence),
            "text_keywords": text_keywords,
            "text_classification": text_classification,
            "text_confidence": float(text_confidence),
            "final_decision": final_decision,
            "priority_level": priority_level,
            "location": location,
            "text_model_loaded": text_model is not None,
            "text_model_compatible": text_model_compatible,
            "text_model_expected_features": text_model_expected_features,
            "text_vectorizer_features": text_vectorizer_features,
            "model_ensemble": model_details,
            "message": "Report processed successfully",
        }

        print(f"✅ Prediction: {final_decision} | Priority: {priority_level}")
        return jsonify(response), 200
    except Exception as error:
        print(f"❌ Error in predict endpoint: {error}")
        return jsonify({"error": f"Server error: {str(error)}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "device": str(device),
        "cnn_loaded": cnn_model is not None,
        "resnet_loaded": resnet_model is not None,
        "text_model_loaded": text_model is not None,
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
    if cnn_model is None or resnet_model is None or text_model is None or text_vectorizer is None:
        print("❌ Failed to load one or more models. Cannot start server.")
        print(f"   CNN Model: {'✅ Loaded' if cnn_model else '❌ Failed'}")
        print(f"   ResNet Model: {'✅ Loaded' if resnet_model else '❌ Failed'}")
        print(f"   Text Model: {'✅ Loaded' if text_model else '❌ Failed'}")
        print(f"   Text Vectorizer: {'✅ Loaded' if text_vectorizer else '❌ Failed'}")
        raise SystemExit(1)

    print("\n" + "=" * 50)
    print("🚨 Disaster Report System - Backend API")
    print("=" * 50)
    print("📍 Server running on: http://localhost:5000")
    print(f"🔧 Device: {device}")
    print(f"📦 Classes: {', '.join(CLASSES)}")
    print("🤖 Models: CNN + ResNet18 (Ensemble)")
    print("⚖️  Weights: 35% CNN + 65% ResNet18")
    print(f"📝 Text model: {TEXT_MODEL_PATH}")
    print(f"🧠 Vectorizer: {TEXT_VECTORIZER_PATH}")
    print("=" * 50 + "\n")

    app.run(debug=True, host="localhost", port=5000, use_reloader=False)
