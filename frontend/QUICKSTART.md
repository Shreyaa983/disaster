# 🚀 Quick Start Guide - Disaster Report System

Get your disaster reporting system up and running in 5 minutes!

## 📋 Prerequisites

- Your backend API running on `http://localhost:5000` with `/predict` endpoint
- A terminal/command prompt
- A modern web browser

## 🎯 Step 1: Start Your Backend

Before running the frontend, make sure your backend server is running:

```bash
cd ../
python predict.py
# or
python train.py  # if you need to start the Flask app
```

Your backend should be listening on `http://localhost:5000`

## 🌐 Step 2: Start the Frontend

Navigate to the frontend directory:

```bash
cd frontend
```

**Choose one method:**

### Option A: Python HTTP Server (Recommended for simplicity)
```bash
# Python 3
python -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000
```
Then open: `http://localhost:8000`

### Option B: Node.js http-server
```bash
npx http-server
```

### Option C: Live Server (VS Code)
- Install Live Server extension
- Right-click `index.html`
- Select "Open with Live Server"

## ✅ Step 3: Test the System

1. Open the frontend in your browser
2. Enter a disaster description (e.g., "Heavy flooding near downtown")
3. (Optional) Upload an image
4. Click "Submit Report"
5. Wait for AI analysis results

## 🔧 Common Issues & Solutions

### ❌ "Failed to process report: ... Make sure the backend server is running"

**Solution**: 
- Check if backend is running on `http://localhost:5000`
- Try accessing `http://localhost:5000` directly in browser
- Check the console for more details (F12 → Network tab)

### ❌ CORS Error

**Solution**:
- Backend needs CORS enabled
- Add to your Flask app: `from flask_cors import CORS; CORS(app)`
- Or add to Express: `app.use(cors());`

### ❌ Image upload not working

**Solution**:
- Check browser console (F12) for errors
- Ensure image is under 10MB
- Try a different image format

### ❌ Results not showing after submission

**Solution**:
- Open browser DevTools (F12)
- Check Network tab to see API response
- Verify backend is returning correct JSON format

## 📊 Expected API Response Format

Your backend should return JSON like this:

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

See `BACKEND_API.md` for complete API documentation.

## 🗂️ File Structure

```
frontend/
├── index.html           # Main HTML page
├── style.css            # Styling (responsive design)
├── script.js            # Frontend logic
├── README.md            # Full documentation
├── BACKEND_API.md       # Backend integration guide
└── QUICKSTART.md        # This file
```

## 🎨 Features Overview

✨ **User Interface**
- Clean, modern design
- Mobile responsive
- Drag & drop image upload
- Real-time validation

🤖 **AI Integration**
- Image prediction from CNN model
- Text analysis with keyword extraction
- Combined decision-making
- Priority level classification

📥 **Report Management**
- Download reports as JSON
- Submit multiple reports
- Error handling and feedback

## 🔒 Security Notes

- Frontend runs entirely in browser (no sensitive data stored)
- All data sent to backend for processing
- Images processed server-side only
- No data persisted in frontend

## 💡 Tips for Better Results

1. **Clear Text Description**: More detailed descriptions help text analysis
2. **Quality Images**: Clear, well-lit images improve CNN predictions
3. **Location Info**: Helps authorities locate and respond faster
4. **Emergency Level**: Set appropriately for priority routing

## 🚀 Next Steps

1. **Customize the Brand**: Edit colors and text in `index.html` and `style.css`
2. **Add Backend Integration**: Follow `BACKEND_API.md` to connect your models
3. **Deploy**: Host on your server when ready
4. **Monitor Results**: Track prediction accuracy and improve models

## 📞 Support

For issues:
1. Check this guide for common problems
2. Review `BACKEND_API.md` for integration help
3. Check browser console (F12) for error messages
4. Verify backend is running correctly

## 📝 Configuration

### Change Backend URL
Edit `script.js` line 5:
```javascript
this.apiEndpoint = 'http://localhost:5000/predict';
```

### Adjust Timeout
Edit `script.js` if API calls are timing out:
```javascript
const timeout = 30000; // milliseconds
```

## 🎓 Learn More

- Full README: See `README.md`
- Backend API: See `BACKEND_API.md`
- Frontend Code: Check `script.js` for implementation details

---

**Happy Disaster Reporting! 🚨**
