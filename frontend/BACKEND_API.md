# Backend API Reference for Frontend Integration

This file documents the expected backend API structure that the frontend expects to communicate with.

## API Endpoint

**URL**: `http://localhost:5000/predict`  
**Method**: `POST`  
**Content-Type**: `multipart/form-data`

## Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | String | Yes | User's description of the disaster situation |
| `image` | File | No | Image file (PNG, JPG, GIF, WebP) up to 10MB |
| `location` | String | No | Location where the disaster occurred |
| `emergency_level` | String | No | User-assessed level: "Low", "Medium", or "High" |

## Backend Model Files

The backend now expects three text-classification assets in addition to the image models:

- `Disaster_tfidf.pkl`
- `Vectorizer_tfidf.pkl`
- `Text_claseses.pkl`

By default, the backend looks for these files in `models/`, but you can override the paths with environment variables:

- `TEXT_MODEL_PATH`
- `TEXT_VECTORIZER_PATH`
- `TEXT_CLASSES_PATH`

## Response Format

All responses should be valid JSON with the following structure:

```json
{
    "image_prediction": "Flood",
    "image_confidence": 0.91,
    "text_keywords": ["flood", "water", "houses"],
    "text_classification": "Flood emergency",
    "text_confidence": 0.94,
    "final_decision": "Flood detected — High Priority",
    "priority_level": "High"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `image_prediction` | String | Predicted disaster class (Earthquake, Fire, Flood, Normal) if image provided |
| `image_confidence` | Float | Confidence score 0-1 (will be converted to percentage) |
| `text_keywords` | Array[String] | Keywords extracted from text |
| `text_classification` | String | Classification based on text analysis |
| `final_decision` | String | Combined AI decision to display to user |
| `priority_level` | String | "High", "Medium", or "Low" |

## Example Implementation (Flask)

If you're using Flask to build the backend, here's a sample structure:

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load your models
# image_model, text_model = load_models()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        text = request.form.get('text', '')
        location = request.form.get('location', 'Not provided')
        emergency_level = request.form.get('emergency_level', 'Not specified')
        
        # Get image if provided
        image_file = request.files.get('image')
        
        image_prediction = None
        image_confidence = 0
        
        # Process image
        if image_file:
            image = Image.open(io.BytesIO(image_file.read()))
            image = image.convert('RGB')
            # Your image processing and model inference here
            # image_prediction, image_confidence = image_model.predict(image)
        
        # Process text
        text_keywords = extract_keywords(text)  # Your keyword extraction logic
        text_classification = classify_text(text)  # Your text classification logic
        
        # Combine results
        final_decision = combine_predictions(
            image_prediction, 
            image_confidence, 
            text_classification
        )
        
        priority_level = determine_priority(
            final_decision, 
            emergency_level
        )
        
        return jsonify({
            'image_prediction': image_prediction or 'Not analyzed',
            'image_confidence': image_confidence,
            'text_keywords': text_keywords,
            'text_classification': text_classification,
            'final_decision': final_decision,
            'priority_level': priority_level
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
```

## CORS Configuration

The frontend requests come from a different origin, so your backend needs CORS support:

```python
from flask_cors import CORS
CORS(app)
```

Or if using Express.js:

```javascript
const cors = require('cors');
app.use(cors());
```

## Error Handling

The frontend expects either:
1. **Success Response** (HTTP 200) with JSON data
2. **Error Response** (HTTP 4xx or 5xx) with optional message

Example error response:
```json
{
    "error": "Image processing failed"
}
```

## Testing the API

You can test your API using curl or Postman:

```bash
# With image
curl -X POST http://localhost:5000/predict \
  -F "text=Heavy flooding reported" \
  -F "image=@/path/to/image.jpg" \
  -F "location=Downtown" \
  -F "emergency_level=High"

# Without image
curl -X POST http://localhost:5000/predict \
  -F "text=Heavy flooding reported" \
  -F "location=Downtown" \
  -F "emergency_level=High"
```

## Performance Considerations

1. **Image Size**: Frontend limits to 10MB, but consider processing smaller images for faster inference
2. **Model Inference**: Cache models in memory to avoid reload on each request
3. **Timeout**: Consider implementing request timeout for long processing tasks
4. **Rate Limiting**: Implement rate limiting to prevent abuse

## Deployment Notes

1. Update `apiEndpoint` in `script.js` if deploying to production
2. Use environment variables for configuration
3. Implement authentication if needed
4. Add logging for debugging
5. Use HTTPS in production

## Integration with Your Models

Based on your project structure:

- **Image Model**: Use your trained CNN from `models/model.pth`
- **Text Model**: Use TF-IDF vectorizer and classifier
- **Preprocessing**: Image resize to 64x64, TF-IDF feature extraction

Refer to your existing `predict.py` and `train.py` scripts for model integration details.
