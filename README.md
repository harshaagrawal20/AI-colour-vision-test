# AI-colour-vision-test

🎨 **AI-Powered D-15 Color Vision Testing System**

A clinical-grade web application that detects color vision deficiencies (color blindness) through an interactive D-15 color arrangement test using AI and color science.

## Flowchart
```mermaid
flowchart TD

%% ===================== FRONTEND =====================
subgraph FRONTEND["🌐 Frontend (HTML, CSS, JS)"]
A1[User uploads image via interface] --> A2[Preview & send image to backend via Fetch API]
A2 --> A3[Display loading spinner and progress status]
A3 --> A4[Receive processed test (15 color pads + reference color)]
A4 --> A5[Render D-15 test grid for user interaction]
A5 --> A6[User arranges colors and submits order]
end

%% ===================== BACKEND =====================
subgraph BACKEND["⚙️ Backend (FastAPI + Python)"]
B1[/Receive uploaded image/]
B1 --> B2[Preprocess image (resize, normalize)]
B2 --> B3[Extract dominant colors using K-Means clustering (15 clusters)]
B3 --> B4[Convert RGB → LAB → Hue values using colorsys]
B4 --> B5[Select reference pad color (usually darkest or brown tone)]
B5 --> B6[Sort colors by hue to generate D-15 color sequence]
B6 --> B7[Return JSON response → frontend with test data]
B7 --> B8[/Receive user arrangement/]
B8 --> B9[Compare user order with true hue order]
B9 --> B10[Compute angular difference per color]
B10 --> B11[Generate confusion lines and total error score]
B11 --> B12[Send metrics to GeminiAnalyzer class]
end

%% ===================== GEMINI AI =====================
subgraph GEMINI["🤖 Gemini AI Analyzer"]
G1[Receive test metrics (errors, sequence deviation, hue difference)]
G1 --> G2[Generate interpretation using Gemini 2.0 Flash model]
G2 --> G3[Identify potential color vision deficiency type:
- Normal
- Protanomaly
- Deuteranomaly
- Tritanomaly]
G3 --> G4[Return descriptive analysis + suggestions to backend]
end

%% ===================== DATABASE / STORAGE =====================
subgraph DB["💾 Data & Storage (optional)"]
D1[User session info]
D2[Test results and timestamps]
D3[Color palette history]
end

%% ===================== FLOW CONNECTIONS =====================
A2 -->|POST /upload| B1
B7 -->|Response: test colors| A4
A6 -->|POST /submit-response| B8
B12 -->|Send test metrics| G1
G4 -->|AI interpretation| B12
B12 -->|Final JSON report| A4
B12 --> DB
```

## Features

- 🔍 **Detects 5 Types of Color Vision**:
  - Normal Vision
  - Protanopia (Red-cone loss) - Red-Green Blindness
  - Deuteranopia (Green-cone loss) - Red-Green Blindness
  - Tritanopia (Blue-cone loss) - Blue-Yellow Blindness
  - Monochromacy (Complete color blindness)

- 📊 **Severity Classification**: Mild / Moderate / Severe / Complete
- 🧬 **Color Science**: Uses CIE LAB color space for accurate analysis
- 🤖 **AI Classification**: Random Forest ML model
- 💡 **Medical Recommendations**: Detailed guidance for each deficiency type

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/harshaagrawal20/AI-colour-vision-test.git
cd AI-colour-vision-test
```

2. Create virtual environment and install dependencies:
```bash
cd color-vision-ai
python -m venv venv_cv
venv_cv\Scripts\activate  # Windows
# source venv_cv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### Running the Application

1. Start the backend server:
```bash
cd backend
python main.py
```
Backend runs on: http://localhost:8000

2. Start the frontend server (in a new terminal):
```bash
cd frontend
python -m http.server 8080
```
Frontend runs on: http://localhost:8080

3. Open your browser and go to: **http://localhost:8080**

## How to Use

1. **Upload an Image**: Click to upload any natural image (JPEG, PNG, WebP)
2. **Arrange Colors**: Drag and drop 15 color patches horizontally in order
3. **Submit**: Click "Submit Response" to get your results
4. **View Results**: See your color vision classification, severity, accuracy score, and recommendations

## System Architecture

### Backend (FastAPI)
- **Color Extraction**: K-Means clustering in LAB color space
- **D-15 Test Generation**: 15 colors with shade variants
- **Error Analysis**: LAB distance metrics
- **AI Classification**: Random Forest classifier

### Frontend (HTML5 + JavaScript)
- **Interactive UI**: Drag-and-drop color arrangement
- **Responsive Design**: Works on desktop and mobile
- **Real-time Feedback**: Instant results display

## Color Science

The system uses:
- **CIE LAB color space** for perceptually uniform color analysis
- **L-channel**: Luminance (brightness)
- **a-channel**: Red-Green axis
- **b-channel**: Yellow-Blue axis

Error patterns in specific channels indicate different types of color blindness.

## Medical Disclaimer

⚠️ **Important**: This tool is a **screening aid**, NOT a diagnostic tool.
- Always confirm results with a professional eye care specialist
- For rare deficiencies (Tritanopia/Monochromacy): Seek immediate medical evaluation
- Results depend on monitor calibration and lighting conditions

## Documentation

- 📄 [Complete User Guide](D15_COLOR_VISION_TEST_GUIDE.md)
- 📄 [Implementation Details](IMPLEMENTATION_UPDATE.md)

## Technologies Used

- **Backend**: FastAPI, Python, OpenCV, NumPy, scikit-learn
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Color Science**: CIE LAB color space
- **Machine Learning**: Random Forest Classifier

## License

Research & Educational Use

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions, please open an issue on GitHub.

---

**Version**: 1.0.0  
**Last Updated**: November 4, 2025


