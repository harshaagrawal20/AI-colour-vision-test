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
        JSON with session_id and extracted color test specification
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
            # Decode image from bytes
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image format")
            image_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        if image_array.size == 0:
            raise HTTPException(status_code=400, detail="Invalid image")

        # Extract dominant colors (LAB or RGB depending on extractor)
        dominant_colors_lab, labels, inertia = color_extractor.extract_dominant_colors(
            image_array, convert_to_lab=True, use_d15_shading=True
        )

        # Generate test specification (without reference pad)
        test_spec = test_generator.generate_test(dominant_colors_lab)

        # Remove reference pad related data if present
        test_spec.pop("reference_order", None)
        test_spec.pop("reference_colors", None)
        test_spec.pop("reference_pad", None)

        # Store session
        session_id = f"session_{len(test_sessions)}"
        test_sessions[session_id] = {
            "test_spec": test_spec,
            "dominant_colors_lab": dominant_colors_lab.tolist(),
            "user_response": None,
            "classification": None,
        }

        return JSONResponse({
            "session_id": session_id,
            "test_spec": test_spec,
            "message": "Test generated successfully (no reference pad used)",
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
    Now uses VIBGYOR sequence as the reference instead of fixed reference pad.
    """
    try:
        if session_id not in test_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = test_sessions[session_id]
        
        # Parse user_order from JSON string
        try:
            user_order = json.loads(user_order)
        except (json.JSONDecodeError, TypeError):
            raise HTTPException(status_code=400, detail="user_order must be a valid JSON array")
        
        expected_length = session["test_spec"]["n_colors"]
        if len(user_order) != expected_length:
            raise HTTPException(status_code=400, detail=f"Expected {expected_length} colors, got {len(user_order)}")
        
        # Convert to NumPy arrays
        user_order_array = np.array(user_order)
        dominant_colors_lab = np.array(session["dominant_colors_lab"])

        # 🟢 Create VIBGYOR reference sequence (sorted by hue or index order)
        # For simplicity, assume color index 0 → Violet, last → Red (or vice versa)
        reference_order = np.arange(len(user_order))  # can later replace with hue sorting logic

        # Compute basic error metrics
        error_metrics = error_analyzer.compute_error_metrics(reference_order, user_order_array, dominant_colors_lab)
        classification = error_analyzer.classify_deficiency(error_metrics)
        accuracy_score = error_analyzer.compute_color_accuracy_score(error_metrics)
        report = error_analyzer.generate_report(classification, accuracy_score)

        # 🧠 Gemini AI Analysis using VIBGYOR logic
        gemini_analysis = None
        if gemini_analyzer:
            try:
                print(f"🤖 Requesting Gemini AI analysis for session {session_id}...")

                gemini_analysis = gemini_analyzer.analyze_color_arrangement(
                    reference_order=reference_order.tolist(),
                    user_order=user_order_array.tolist(),
                    reference_colors_lab=dominant_colors_lab.tolist(),
                    classification=classification,
                    accuracy_score=accuracy_score
                )

                print(f"✓ Gemini analysis received: {len(gemini_analysis.get('ai_analysis', ''))} characters")

            except Exception as e:
                print(f"⚠ Gemini analysis failed: {e}")
                gemini_analysis = {
                    "ai_analysis": f"Gemini AI analysis unavailable: {str(e)}",
                    "success": False,
                    "error": str(e)
                }
        else:
            gemini_analysis = {
                "ai_analysis": "Gemini AI is not configured. Please set GEMINI_API_KEY environment variable.",
                "success": False,
                "error": "Gemini API key not configured"
            }
        
        # Store results
        session["user_response"] = user_order
        session["classification"] = classification
        session["accuracy_score"] = accuracy_score
        session["report"] = report
        session["gemini_analysis"] = gemini_analysis
        
        return JSONResponse({
            "classification": classification,
            "accuracy_score": accuracy_score,
            "report": report,
            "gemini_analysis": gemini_analysis,
        })
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in submit_response: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
        "report": session.get("report"),
        "has_gemini_analysis": "gemini_analysis" in session,
    })


@app.post("/generate-distractor-test")
async def generate_distractor_test(session_id: str):
    """Generate a harder test variant with distractors."""
    try:
        if session_id not in test_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = test_sessions[session_id]
        dominant_colors_lab = np.array(session["dominant_colors_lab"])
        
        distractor_spec = test_generator.generate_distractor_test(dominant_colors_lab)
        session["distractor_test"] = distractor_spec
        
        return JSONResponse(distractor_spec)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-luminance-test")
async def generate_luminance_test(session_id: str):
    """Generate a luminance-equalized test variant."""
    try:
        if session_id not in test_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = test_sessions[session_id]
        dominant_colors_lab = np.array(session["dominant_colors_lab"])
        
        luminance_spec = test_generator.generate_luminance_test(dominant_colors_lab)
        session["luminance_test"] = luminance_spec
        
        return JSONResponse(luminance_spec)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎨 Color Vision AI Testing System")
    print("="*60)
    print(f"Gemini AI: {'✓ Enabled' if gemini_analyzer else '⚠ Disabled (no API key)'}")
    print("="*60 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)             