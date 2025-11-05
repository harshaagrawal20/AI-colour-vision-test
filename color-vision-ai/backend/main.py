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
from dotenv import load_dotenv

from color_extractor import ColorExtractor
from test_generator import TestGenerator
from error_analyzer import ErrorAnalyzer
from gemini_analyzer import GeminiAnalyzer
import traceback

# Load environment variables
load_dotenv()

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
color_extractor = ColorExtractor(n_colors=15)  # Use 15 colors for D-15 test (OpenCV + K-Means, NO AI)
test_generator = TestGenerator(test_type="patch_ordering")  # Test generation (NO AI)
error_analyzer = ErrorAnalyzer()  # ML classification (NO Gemini AI)

# Initialize Gemini AI analyzer (ONLY used AFTER user submits their arrangement)
# NOT used for color extraction or test generation
try:
    gemini_analyzer = GeminiAnalyzer()
    print("✓ Gemini AI analyzer initialized (ONLY for post-submission analysis)")
except Exception as e:
    gemini_analyzer = None
    print(f"⚠ Gemini AI not available: {str(e)}")

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
    print(f"\n=== UPLOAD IMAGE REQUEST ===")
    try:
        # Validate file
        if not file:
            print("ERROR: No file provided")
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not file.filename:
            print("ERROR: File has no name")
            raise HTTPException(status_code=400, detail="File has no name")
        
        print(f"File received: {file.filename}, Content-Type: {file.content_type}")
        
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
        
        print(f"Image decoded successfully: {image_array.shape}")
        
        # Resize large images for faster processing (max 800x800)
        h, w = image_array.shape[:2]
        max_size = 800
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image_array = cv2.resize(image_array, (new_w, new_h))
            print(f"Image resized to: {image_array.shape}")
        
        # Extract dominant colors in LAB
        print("Extracting dominant colors using K-Means...")
        dominant_colors_lab, labels, inertia = color_extractor.extract_dominant_colors(
            image_array, convert_to_lab=True, use_d15_shading=True
        )
        print(f"Extracted {len(dominant_colors_lab)} colors in proper hue sequence")

        # --- Select reference color based on median hue ---
        print("Selecting reference color based on hue angle...")

        a_vals = dominant_colors_lab[:, 1]
        b_vals = dominant_colors_lab[:, 2]
        hues = np.degrees(np.arctan2(b_vals, a_vals))
        hues[hues < 0] += 360  # convert to 0–360 range

        # Sort colors by hue and select median hue
        sorted_indices = np.argsort(hues)
        median_idx = int(sorted_indices[len(sorted_indices) // 2])
        reference_order = np.arange(len(dominant_colors_lab))

        # ✅ FIX: Explicitly use scalar comparison, avoid ambiguous truth
        if len(reference_order) > 0 and int(median_idx) != 0:
            temp = reference_order[0]
            reference_order[0] = reference_order[median_idx]
            reference_order[median_idx] = temp

        print(f"Reference color index chosen: {median_idx} (Hue={hues[median_idx]:.2f})")

        # --- Convert reference LAB to RGB ---
        reference_color_lab = np.expand_dims(np.expand_dims(dominant_colors_lab[median_idx], axis=0), axis=0)
        reference_color_rgb = cv2.cvtColor(reference_color_lab.astype(np.float32), cv2.COLOR_Lab2RGB)
        reference_pad_color_rgb = np.clip(reference_color_rgb[0, 0] * 255, 0, 255).astype(np.uint8).tolist()

        # --- Generate test with updated reference ---
        print("Generating test spec...")
        test_spec = test_generator.generate_test(dominant_colors_lab, reference_order)
        test_spec["reference_pad_color"] = reference_pad_color_rgb

        # Also add an explicit reference_color entry so the frontend doesn't have
        # to rely on array ordering (and to make debugging easier).
        try:
            if "patch_configs" in test_spec and len(test_spec["patch_configs"]) > 0:
                test_spec["reference_color"] = test_spec["patch_configs"][0]
        except Exception:
            traceback.print_exc()

        print(f"Test spec generated with {test_spec['n_colors']} colors, ref color: {reference_pad_color_rgb}")

                
        # Store session
        session_id = f"session_{len(test_sessions)}"
        test_sessions[session_id] = {
            "test_spec": test_spec,
            "reference_order": test_spec["reference_order"],
            "dominant_colors_lab": dominant_colors_lab.tolist(),
            "user_response": None,
            "classification": None,
        }
        
        print(f"Session created: {session_id}")
        print("=== UPLOAD SUCCESS ===\n")
        
        return JSONResponse({
            "session_id": session_id,
            "test_spec": test_spec,
            "message": "Test generated successfully",
        })
    
    except Exception as e:
        # Print full traceback to server logs to diagnose ambiguous-truth errors
        print("UPLOAD ERROR:", repr(e))
        traceback.print_exc()
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
        
        # Get data
        reference_order = np.array(session["reference_order"])
        user_order_array = np.array(user_order)
        dominant_colors_lab = np.array(session["dominant_colors_lab"])
        
        print(f"\n=== SUBMITTING TO GEMINI AI ===")
        print(f"Reference order: {reference_order.tolist()}")
        print(f"User order: {user_order}")
        
        # Send EVERYTHING to Gemini AI for complete analysis
        gemini_analysis = None
        if not gemini_analyzer:
            raise HTTPException(
                status_code=503,
                detail="Gemini AI is not configured. Please set GEMINI_API_KEY in .env file."
            )
        
        try:
            print("Sending to Gemini AI...")
            gemini_analysis = gemini_analyzer.analyze_color_arrangement(
                reference_order=reference_order.tolist(),
                user_order=user_order,
                reference_colors_lab=dominant_colors_lab.tolist(),
                classification=None,  # No ML classification, pure AI
                accuracy_score=None   # Let AI calculate everything
            )
            print("Gemini AI analysis received!")
            print(f"Success: {gemini_analysis.get('success')}")
        except Exception as e:
            print(f"Gemini AI analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Gemini AI analysis failed: {str(e)}"
            )
        
        # Store response
        session["user_response"] = user_order
        session["gemini_analysis"] = gemini_analysis
        
        print("=== GEMINI AI ANALYSIS COMPLETE ===\n")
        
        return JSONResponse({
            "gemini_analysis": gemini_analysis,
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
    import asyncio
    from hypercorn.config import Config
    from hypercorn.asyncio import serve
    
    print("\n" + "="*60)
    print("🚀 Starting Color Vision AI Backend Server")
    print("="*60)
    print(f"📡 Server URL: http://localhost:8000")
    print(f"📡 API Docs: http://localhost:8000/docs")
    print(f"🐍 Python 3.12.7 Compatible (Hypercorn)")
    print("="*60 + "\n")
    
    config = Config()
    config.bind = ["0.0.0.0:8000"]
    config.use_reloader = True
    
    asyncio.run(serve(app, config))