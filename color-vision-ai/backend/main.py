"""
FastAPI backend for AI-powered color vision testing system.
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
color_extractor = ColorExtractor(n_colors=15)  # Use 15 colors for D-15 test
test_generator = TestGenerator(test_type="patch_ordering")
error_analyzer = ErrorAnalyzer()

# Storage (in-memory; use DB for production)
test_sessions = {}


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Color Vision AI Testing System",
        "endpoints": {
            "POST /upload-image": "Upload image and get generated test",
            "POST /submit-response": "Submit user response and get classification",
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
        
        # Extract dominant colors in LAB (with D-15 shading for better accuracy)
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
    user_order: str,  # Accept as JSON string
):
    """
    Submit user's color ordering response and get classification.
    
    Args:
        session_id: Session ID from /upload-image
        user_order: JSON string of color indices list
        
    Returns:
        JSON with classification, accuracy score, and recommendations
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
        
        # Compute error metrics
        reference_order = np.array(session["reference_order"])
        user_order_array = np.array(user_order)
        dominant_colors_lab = np.array(session["dominant_colors_lab"])
        
        error_metrics = error_analyzer.compute_error_metrics(
            reference_order, user_order_array, dominant_colors_lab
        )
        
        # Classify deficiency
        classification = error_analyzer.classify_deficiency(error_metrics)
        accuracy_score = error_analyzer.compute_color_accuracy_score(error_metrics)
        report = error_analyzer.generate_report(classification, accuracy_score)
        
        # Store response
        session["user_response"] = user_order
        session["classification"] = classification
        session["accuracy_score"] = accuracy_score
        session["report"] = report
        
        return JSONResponse({
            "classification": classification,
            "accuracy_score": accuracy_score,
            "report": report,
        })
    
    except HTTPException:
        raise
    except Exception as e:
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
    })


@app.post("/generate-distractor-test")
async def generate_distractor_test(session_id: str):
    """
    Generate a harder test variant with distractors.
    
    Args:
        session_id: Session ID
        
    Returns:
        JSON with distractor test spec
    """
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
    """
    Generate a luminance-equalized test variant.
    
    Args:
        session_id: Session ID
        
    Returns:
        JSON with luminance test spec
    """
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
