"""
FastAPI backend for AI-powered color vision testing system with Gemini AI.
"""
import os
import json
from io import BytesIO
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import uvicorn

from color_extractor import ColorExtractor
from test_generator import TestGenerator
from error_analyzer import ErrorAnalyzer
from gemini_analyzer import GeminiAnalyzer

app = FastAPI(
    title="Color Vision AI Testing System",
    description="AI-powered color vision testing from custom images",
    version="0.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
color_extractor = ColorExtractor(n_colors=15)
test_generator = TestGenerator(test_type="patch_ordering")
error_analyzer = ErrorAnalyzer()

# Initialize Gemini analyzer (will fail gracefully if no API key)
try:
    gemini_analyzer = GeminiAnalyzer()
    print("✓ Gemini AI analyzer initialized successfully")
except Exception as e:
    gemini_analyzer = None
    print(f"⚠ Gemini AI not available: {e}")

# Storage (in-memory; use DB for production)
test_sessions = {}


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Color Vision AI Testing System",
        "gemini_enabled": gemini_analyzer is not None,
        "endpoints": {
            "POST /upload-image": "Upload image and get generated test",
            "POST /submit-response": "Submit user response and get AI analysis",
            "GET /session/{session_id}": "Get session details",
        }
    }


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image and extract dominant colors to generate a test.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        
    Returns:
        JSON with test_id, test_spec, and reference colors
    """
    try:
        # Validate file
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not file.filename:
            raise HTTPException(status_code=400, detail="File has no name")
        
        # Read and validate image
        contents = await file.read()
        
        if not contents:
            raise HTTPException(status_code=400, detail="File is empty")
        
        try:
            # Use cv2 to decode image from bytes
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image format")
            # Convert BGR to RGB
            image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
        if image_array.size == 0:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Extract dominant colors in LAB
        dominant_colors_lab, labels, inertia = color_extractor.extract_dominant_colors(
            image_array, convert_to_lab=True, use_d15_shading=True
        )
        
        # Generate test
        test_spec = test_generator.generate_test(dominant_colors_lab)
        
        # Store session
        session_id = f"session_{len(test_sessions)}"
        test_sessions[session_id] = {
            "test_spec": test_spec,
            "reference_order": test_spec["reference_order"],
            "dominant_colors_lab": dominant_colors_lab.tolist(),
            "user_response": None,
            "classification": None,
        }
        
        return JSONResponse({
            "session_id": session_id,
            "test_spec": test_spec,
            "message": "Test generated successfully",
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/submit-response")
async def submit_response(
    session_id: str,
    user_order: str,
):
    """
    Submit user's color ordering response and get AI analysis from Gemini.
    
    Args:
        session_id: Session ID from /upload-image
        user_order: JSON string of color indices list
        
    Returns:
        JSON with Gemini AI analysis in structured format
    """
    try:
        if session_id not in test_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = test_sessions[session_id]
        
        # Parse user_order from JSON string
        try:
            user_order = json.loads(user_order)
        except (json.JSONDecodeError, TypeError):
            raise HTTPException(
                status_code=400,
                detail="user_order must be a valid JSON array"
            )
        
        # Verify user_order
        expected_length = session["test_spec"]["n_colors"]
        if len(user_order) != expected_length:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected_length} colors, got {len(user_order)}"
            )
        
        # Get data
        reference_order = np.array(session["reference_order"])
        user_order_array = np.array(user_order)
        dominant_colors_lab = np.array(session["dominant_colors_lab"])
        
        # Compute basic error metrics (for fallback)
        error_metrics = error_analyzer.compute_error_metrics(
            reference_order, user_order_array, dominant_colors_lab
        )
        
        # Classify deficiency (for fallback)
        classification = error_analyzer.classify_deficiency(error_metrics)
        accuracy_score = error_analyzer.compute_color_accuracy_score(error_metrics)
        
        # Get Gemini AI Analysis with structured prompt
        gemini_analysis = None
        if gemini_analyzer:
            try:
                print(f"🤖 Requesting Gemini AI analysis for session {session_id}...")
                
                # Create a structured prompt for minimal output
                structured_prompt = f"""Analyze this color vision test result and provide a BRIEF, STRUCTURED response.

Reference Order: {reference_order.tolist()}
User Order: {user_order_array.tolist()}
Colors (LAB): {dominant_colors_lab.tolist()}
Initial Classification: {classification}
Accuracy Score: {accuracy_score:.1f}%

Provide your analysis in this EXACT format with SHORT answers:

RESULT: [Normal/Protanopia/Deuteranopia/Tritanopia/etc.]

CONFIDENCE: [High/Medium/Low]

KEY FINDINGS:
- [One line finding 1]
- [One line finding 2]
- [One line finding 3]

AFFECTED AREAS:
- [Color range 1]
- [Color range 2]

RECOMMENDATION: [One sentence recommendation]

Keep each point to ONE line maximum. Be concise and direct."""

                # Call Gemini with structured prompt
                gemini_analysis = gemini_analyzer.analyze_with_custom_prompt(
                    structured_prompt
                )
                
                # Parse the structured response
                parsed_analysis = parse_gemini_response(gemini_analysis.get('ai_analysis', ''))
                gemini_analysis['parsed'] = parsed_analysis
                
                print(f"✓ Gemini analysis received and parsed")
            except Exception as e:
                print(f"⚠ Gemini analysis failed: {e}")
                gemini_analysis = {
                    "success": False,
                    "error": str(e),
                    "parsed": create_fallback_analysis(classification, accuracy_score)
                }
        else:
            gemini_analysis = {
                "success": False,
                "error": "Gemini API key not configured",
                "parsed": create_fallback_analysis(classification, accuracy_score)
            }
        
        # Store response
        session["user_response"] = user_order
        session["classification"] = classification
        session["accuracy_score"] = accuracy_score
        session["gemini_analysis"] = gemini_analysis
        
        # Return structured response
        return JSONResponse({
            "classification": classification,
            "accuracy_score": accuracy_score,
            "gemini_analysis": gemini_analysis,
        })
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in submit_response: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def parse_gemini_response(text):
    """Parse Gemini's structured response into JSON."""
    lines = text.strip().split('\n')
    parsed = {
        'result': 'Unknown',
        'confidence': 'Low',
        'key_findings': [],
        'affected_areas': [],
        'recommendation': ''
    }
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('RESULT:'):
            parsed['result'] = line.replace('RESULT:', '').strip()
        elif line.startswith('CONFIDENCE:'):
            parsed['confidence'] = line.replace('CONFIDENCE:', '').strip()
        elif line.startswith('KEY FINDINGS:'):
            current_section = 'findings'
        elif line.startswith('AFFECTED AREAS:'):
            current_section = 'areas'
        elif line.startswith('RECOMMENDATION:'):
            parsed['recommendation'] = line.replace('RECOMMENDATION:', '').strip()
            current_section = None
        elif line.startswith('-') or line.startswith('•'):
            item = line.lstrip('-•').strip()
            if current_section == 'findings':
                parsed['key_findings'].append(item)
            elif current_section == 'areas':
                parsed['affected_areas'].append(item)
    
    return parsed


def create_fallback_analysis(classification, accuracy_score):
    """Create fallback analysis when Gemini is unavailable."""
    return {
        'result': classification,
        'confidence': 'Medium' if accuracy_score < 70 else 'High',
        'key_findings': [
            f'Color arrangement accuracy: {accuracy_score:.1f}%',
            f'Classification based on error patterns: {classification}',
            'Gemini AI analysis unavailable - using algorithmic classification'
        ],
        'affected_areas': [
            'Unable to determine specific color confusion areas',
            'Consult eye care professional for detailed assessment'
        ],
        'recommendation': 'Please configure Gemini API key for detailed AI analysis, or consult an eye care professional.'
    }


@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Get session details."""
    if session_id not in test_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = test_sessions[session_id]
    return JSONResponse({
        "session_id": session_id,
        "has_response": session["user_response"] is not None,
        "classification": session.get("classification"),
        "accuracy_score": session.get("accuracy_score"),
        "has_gemini_analysis": "gemini_analysis" in session,
    })


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎨 Color Vision AI Testing System")
    print("="*60)
    print(f"Gemini AI: {'✓ Enabled' if gemini_analyzer else '⚠ Disabled (no API key)'}")
    print("="*60 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)