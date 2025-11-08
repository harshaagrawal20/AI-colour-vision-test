"""
FastAPI backend for AI-powered color vision testing system with Gemini AI.
"""
import os
import json
from io import BytesIO
import traceback
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import uvicorn
import colorsys
from typing import List

from color_extractor import ColorExtractor
from test_generator import TestGenerator
from error_analyzer import ErrorAnalyzer
from gemini_analyzer import GeminiAnalyzer


app = FastAPI(
    title="Color Vision AI Testing System",
    description="AI-powered color vision testing from custom images",
    version="0.1.0",
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
        },
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

        # Choose a reference index (perceptual midpoint) and generate test spec
        try:
            # Convert LAB colors to RGB using our existing ColorExtractor method
            rgb_colors = color_extractor.lab_to_rgb(dominant_colors_lab)
            
            # Convert RGB to HSV to get hue angles
            # Reshape for OpenCV color conversion (expects image-like input)
            rgb_reshaped = rgb_colors.reshape(1, -1, 3)
            hsv_colors = cv2.cvtColor(rgb_reshaped, cv2.COLOR_RGB2HSV)
            hue_angles = hsv_colors[0, :, 0]  # Hue is in range [0, 180] for OpenCV
            
            # Convert to HSL for better hue handling
            def get_hue_angle(rgb):
                """Get hue angle [0-360] from RGB color using colorsys."""
                r, g, b = rgb / 255.0
                h, _, _ = colorsys.rgb_to_hsv(r, g, b)
                return h * 360  # Convert [0-1] to degrees [0-360]

            # Calculate hue angles using colorsys (more reliable than OpenCV for this)
            hue_angles = np.array([get_hue_angle(rgb) for rgb in rgb_colors])
            
            # Choose color with lowest hue angle as reference (closest to red at 0°)
            ref_index = int(np.argmin(hue_angles))

            # Reorder colors to put reference first (TestGenerator expects reference at index 0)
            colors_reordered = np.vstack([
                dominant_colors_lab[ref_index:ref_index+1],  # Reference color first
                dominant_colors_lab[:ref_index],             # Colors before reference
                dominant_colors_lab[ref_index+1:]            # Colors after reference
            ])

            # Generate test (reference is now at index 0)
            test_spec = test_generator.generate_test(colors_reordered)
            
            # Ensure n_colors matches the number of returned colors
            test_spec["n_colors"] = int(len(dominant_colors_lab))

            # Build an explicit reference_color object for frontend using the first color (now our chosen reference)
            try:
                ref_rgb = test_spec["reference_pad_rgb"]
                test_spec["reference_color"] = {
                    "color_index": int(ref_index),  # Original index in dominant_colors_lab
                    "lab": test_spec["reference_pad_lab"],
                    "rgb": [int(x) for x in ref_rgb],
                }
            except Exception:
                # Non-fatal: fall back to first patch if mapping fails
                traceback.print_exc()
                if "patch_configs" in test_spec and len(test_spec["patch_configs"]) > 0:
                    test_spec["reference_color"] = test_spec["patch_configs"][0]

            # Also provide a simple RGB triple for quick frontend use
            try:
                reference_pad_color_rgb = test_spec["reference_color"]["rgb"]
                test_spec["reference_pad_color"] = reference_pad_color_rgb
            except Exception:
                test_spec["reference_pad_color"] = None

            print(
                f"Test spec generated: n_colors={test_spec['n_colors']}, patches={len(test_spec.get('patch_configs', []))}, ref_index={ref_index}"
            )
        except Exception as e:
            print("Error generating test spec:", e)
            traceback.print_exc()
            raise

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
            "message": "Test generated successfully (with reference pad)",
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel


class SubmitRequest(BaseModel):
    session_id: str
    user_order: List[int]


@app.post("/submit-response")
async def submit_response(data: SubmitRequest):
    """
    Submit user's color ordering response and get AI analysis from Gemini.

    Expects `user_order` to be the full ordering including the reference at index 0.
    """
    try:
        session_id = data.session_id
        user_order = data.user_order

        if session_id not in test_sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session = test_sessions[session_id]
        test_spec = session["test_spec"]

        # user_order is expected to be the full ordering including reference at index 0
        if not isinstance(user_order, list):
            raise HTTPException(status_code=400, detail="user_order must be a JSON array of indices")

        # Convert to NumPy arrays
        user_order_array = np.array(user_order)
        dominant_colors_lab = np.array(session["dominant_colors_lab"])

        expected_len = int(test_spec.get("n_colors", len(dominant_colors_lab))) + 1
        if len(user_order) != expected_len:
            raise HTTPException(
                status_code=400,
                detail=f"Expected full order length {expected_len} (including reference), got {len(user_order)}",
            )

        # full_user_order is the order submitted (includes reference at index 0)
        full_user_order = user_order_array.tolist()

        # Use reference_order from test_spec if available (generator provides correct arrangement), else default
        full_reference_order = (
            test_spec.get("reference_order")
            if test_spec.get("reference_order") is not None
            else list(range(len(dominant_colors_lab)))
        )

        # Compute basic error metrics
        error_metrics = error_analyzer.compute_error_metrics(
            full_reference_order, full_user_order, dominant_colors_lab
        )
        classification = error_analyzer.classify_deficiency(error_metrics)
        accuracy_score = error_analyzer.compute_color_accuracy_score(error_metrics)
        report = error_analyzer.generate_report(classification, accuracy_score)

        # 🧠 Gemini AI Analysis using VIBGYOR logic
        gemini_analysis = None
        if gemini_analyzer:
            try:
                print(f"🤖 Requesting Gemini AI analysis for session {session_id}...")

                gemini_analysis = gemini_analyzer.analyze_color_arrangement(
                    reference_order=full_reference_order,
                    user_order=full_user_order,
                    reference_colors_lab=dominant_colors_lab.tolist(),
                    classification=classification,
                    accuracy_score=accuracy_score,
                )

                print(f"✓ Gemini analysis received: {len(gemini_analysis.get('ai_analysis', ''))} characters")

            except Exception as e:
                print(f"⚠ Gemini analysis failed: {e}")
                traceback.print_exc()
                gemini_analysis = {
                    "ai_analysis": f"Gemini AI analysis unavailable: {str(e)}",
                    "success": False,
                    "error": str(e),
                }
        else:
            gemini_analysis = {
                "ai_analysis": "Gemini AI is not configured. Please set GEMINI_API_KEY environment variable.",
                "success": False,
                "error": "Gemini API key not configured",
            }

        # Store results
        session["user_response"] = full_user_order
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
        traceback.print_exc()
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎨 Color Vision AI Testing System")
    print("=" * 60)
    print(f"Gemini AI: {'✓ Enabled' if gemini_analyzer else '⚠ Disabled (no API key)'}")
    print("=" * 60 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)