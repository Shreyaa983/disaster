# 🚨 Disaster Report System - Frontend

A modern, responsive web interface for reporting disasters with AI-powered analysis. Users can submit text descriptions and images to be processed by the backend CNN and text analysis models.

## 📋 Features

✨ **User-Friendly Interface**
- Clean, intuitive disaster report form
- Optional image upload with drag & drop support
- Location and emergency level inputs
- Real-time form validation

🤖 **AI Analysis Display**
- Image prediction with confidence percentage
- Text analysis with detected keywords
- Combined final decision and priority level
- Visual confidence bars and priority badges

📥 **Report Management**
- Download reports as JSON
- Submit multiple reports
- Clear error handling and feedback

## 🛠️ Setup & Usage

### Prerequisites
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Backend server running on `http://localhost:5000`

### Files Structure
```
frontend/
├── index.html       # Main HTML structure
├── style.css        # Styling and responsive design
├── script.js        # Frontend logic and API communication
└── README.md        # This file
```

### Running the Frontend

**Option 1: Simple HTTP Server**
```bash
# Using Python 3
python -m http.server 8080

# Using Python 2
python -m SimpleHTTPServer 8080
```
Then open `http://localhost:8080` in your browser.

**Option 2: Node.js http-server**
```bash
npm install -g http-server
http-server
```

**Option 3: Live Server (VS Code)**
- Install Live Server extension
- Right-click on `index.html` and select "Open with Live Server"

## 🔌 Backend API Integration

The frontend expects the backend API at: `http://localhost:5000/predict`

### API Request Format (POST)
```
Content-Type: multipart/form-data

Parameters:
- text: String (description of the disaster)
- image: File (optional, image of the disaster)
- location: String (optional, location information)
- emergency_level: String (optional, one of: Low, Medium, High)
```

### API Response Format (JSON)
```json
{
    "image_prediction": "Flood",
    "image_confidence": 0.91,
    "text_keywords": ["flood", "water", "houses"],
    "text_classification": "Flood emergency",
    "final_decision": "Flood detected — High Priority",
    "priority_level": "High"
}
```

## 📱 Data Flow

```
User Input
    ↓
Form Validation
    ↓
Send to Backend (multipart/form-data)
    ├─ Text Description
    ├─ Image (if provided)
    ├─ Location
    └─ Emergency Level
    ↓
Backend Processing
    ├─ Image → CNN Model
    ├─ Text → TF-IDF Analysis
    └─ Combine Results
    ↓
Frontend Receives JSON Response
    ↓
Display Results Section
    ├─ Image Prediction
    ├─ Text Analysis
    └─ Final Decision
```

## 🎨 UI Components

### Main Form Section
- **Text Description**: Large textarea for detailed disaster information
- **Image Upload**: Drag & drop or click to upload (up to 10MB)
- **Location**: Optional location information
- **Emergency Level**: Dropdown (Low, Medium, High)
- **Submit Button**: Triggers API call to backend

### Results Section
- **Image Prediction Card**: Shows disaster type and confidence percentage
- **Text Analysis Card**: Displays keywords and classification
- **Final Result Card**: Combined AI decision with priority level
- **Action Buttons**: Download report or submit new report

## 🔐 Features & Validations

✅ **Input Validation**
- Text description is required
- Image size limited to 10MB
- Supported image formats: PNG, JPG, GIF, WebP

✅ **Error Handling**
- Displays error messages if backend is unavailable
- Handles network errors gracefully
- Validates form inputs before submission

✅ **User Feedback**
- Loading spinner during API call
- Success animation for results
- Clear error messages with troubleshooting hints

## 🎯 Responsive Design

- **Desktop**: Full-width responsive layout
- **Tablet**: Optimized for smaller screens
- **Mobile**: Touch-friendly interface with stacked components

## 📊 Priority Levels

- 🚨 **High**: Critical disaster requiring immediate emergency response
- ⚠️ **Medium**: Significant event needing prompt attention
- ✅ **Low**: Minor incident or false alarm

## 🐛 Troubleshooting

### "Failed to process report: ... Make sure the backend server is running"
**Solution**: Ensure your backend server is running on `http://localhost:5000` and the `/predict` endpoint is available.

### Image upload not working
**Solution**: Check if the browser supports the File API and Fetch API (all modern browsers do).

### Results not displaying
**Solution**: Open browser DevTools (F12) and check the Network tab to see the API response format.

## 🚀 Future Enhancements

- Real-time notifications for authorities
- Map integration showing disaster locations
- Historical report dashboard
- Multi-language support
- Mobile app version
- WebSocket for live updates

## 📄 License

This project is part of the Calamity AI System.
