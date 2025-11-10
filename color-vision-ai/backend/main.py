"""
FastAPI backend for AI-powered color vision testing system with Gemini AI.
"""
import os
import json
from io import BytesIO
import traceback
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import cv2
import uvicorn
import colorsys
from typing import List
import base64

from color_extractor import ColorExtractor
from test_generator import TestGenerator
from error_analyzer import ErrorAnalyzer
from gemini_analyzer import GeminiAnalyzer
from d15_graph_generator import D15GraphGenerator


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
graph_generator = D15GraphGenerator()  # D-15 graph generator


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

            # ✅ REFERENCE COLOR KO DUPLICATE KARO
            # Array mein reference color ko include karo (total 15 colors with reference duplicate)
            # Index 0: Reference color (duplicate for pad)
            # Index 1-15: All 15 original colors (including reference again at its original position)
            colors_with_reference_duplicate = np.vstack([
                dominant_colors_lab[ref_index:ref_index+1],  # Reference color (duplicate for pad)
                dominant_colors_lab                          # All 15 colors (including reference)
            ])

            # Generate test (now has 16 total: 1 reference duplicate + 15 original colors)
            test_spec = test_generator.generate_test(colors_with_reference_duplicate)
            
            # 🔧 REBUILD patch_configs in PREDICTABLE ORDER!
            # patch_configs[i] should contain the RGB for color index i
            rgb_colors_with_ref = color_extractor.lab_to_rgb(colors_with_reference_duplicate)
            test_spec["patch_configs"] = [
                {
                    "color_index": i,
                    "lab": colors_with_reference_duplicate[i].tolist(),
                    "rgb": rgb_colors_with_ref[i].tolist(),
                    "luminance": float(colors_with_reference_duplicate[i, 0]),
                }
                for i in range(len(colors_with_reference_duplicate))
            ]
            
            # 🎲 CREATE A SHUFFLED DISPLAY ORDER for the available colors
            # This determines the order colors appear in the "Available Colors" section
            # We shuffle indices 1-15 (skipping 0 which is the reference)
            available_color_indices = list(range(1, len(colors_with_reference_duplicate)))
            np.random.shuffle(available_color_indices)
            test_spec["display_order"] = available_color_indices  # Shuffled order for display
            
            print(f"  Display order (shuffled): {available_color_indices[:5]}... (showing first 5)")
            
            # n_colors = 15 (original colors for arrangement)
            test_spec["n_colors"] = int(len(dominant_colors_lab))
            test_spec["reference_index"] = 0  # Reference is at index 0
            test_spec["total_colors"] = int(len(colors_with_reference_duplicate))  # 16 total

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
        
        # Get the full colors array (including reference duplicate at index 0)
        # This should match what was used to generate the test
        ref_index = test_spec.get("reference_color", {}).get("color_index", 0)
        colors_with_reference = np.vstack([
            dominant_colors_lab[ref_index:ref_index+1],  # Reference at index 0
            dominant_colors_lab                          # All 15 colors at indices 1-15
        ])

        expected_len = int(test_spec.get("n_colors", len(dominant_colors_lab))) + 1
        if len(user_order) != expected_len:
            raise HTTPException(
                status_code=400,
                detail=f"Expected full order length {expected_len} (including reference), got {len(user_order)}",
            )

        # full_user_order is the order submitted (includes reference at index 0)
        full_user_order = user_order_array.tolist()

        # Use reference_order from test_spec if available, else create default with reference at index 0
        if test_spec.get("reference_order") is not None:
            full_reference_order = test_spec.get("reference_order")
        else:
            # Create reference order: [0, 1, 2, 3, ..., 15] (16 items total)
            full_reference_order = list(range(len(colors_with_reference)))

        # Compute basic error metrics using the full 16-color array
        error_metrics = error_analyzer.compute_error_metrics(
            full_reference_order, full_user_order, colors_with_reference
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
                    reference_colors_lab=colors_with_reference.tolist(),  # Use full 16-color array
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

        # 📊 Generate D-15 Graph
        d15_graph_base64 = None
        try:
            # Build array of ALL colors from patch_configs (including reference)
            # patch_configs[i] contains the RGB for color index i
            all_patch_colors = []
            for config in test_spec.get("patch_configs", []):
                all_patch_colors.append(config["rgb"])
            
            # Get reference color (index 0)
            reference_color_rgb = all_patch_colors[0] if len(all_patch_colors) > 0 else [128, 0, 0]
            
            # user_order = [0, idx1, idx2, ..., idx15]
            # where idx1 is the COLOR INDEX placed at position 1, idx2 at position 2, etc.
            # We need to skip the reference (index 0) and pass the color indices
            user_arrangement = user_order[1:]  # [idx1, idx2, ..., idx15]
            
            print(f"\n📊 D-15 Graph Generation Debug:")
            print(f"  Total colors in patch_configs: {len(all_patch_colors)}")
            print(f"  Full user_order: {user_order}")
            print(f"  User arrangement (positions 1-15): {user_arrangement}")
            print(f"  Example: Position 1 has color index {user_arrangement[0]}")
            print(f"  Example: Position 1 color RGB: {all_patch_colors[user_arrangement[0]] if user_arrangement[0] < len(all_patch_colors) else 'OUT OF RANGE'}")
            
            # Generate graph
            # user_arrangement contains color indices that map to all_patch_colors
            graph_bytes = graph_generator.generate_graph(
                user_order=user_arrangement,  # Which color INDEX at each position
                all_colors_rgb=all_patch_colors,  # ALL colors (including reference at index 0)
                reference_color_rgb=reference_color_rgb,
                show_confusion_lines=True,
                title="Your D-15 Test Result"
            )
            
            # Convert to base64 for JSON response
            d15_graph_base64 = base64.b64encode(graph_bytes).decode('utf-8')
            print(f"✓ D-15 graph generated successfully")
            
        except Exception as e:
            print(f"⚠ D-15 graph generation failed: {e}")
            traceback.print_exc()

        # 🎨 Generate Hue Spectrum Error Visualization
        hue_spectrum_base64 = None
        try:
            # Same setup as D-15 graph
            all_patch_colors = []
            for config in test_spec.get("patch_configs", []):
                all_patch_colors.append(config["rgb"])
            
            user_arrangement = user_order[1:]  # Skip reference
            
            print(f"\n🎨 Hue Spectrum Generation Debug:")
            print(f"  Generating hue error visualization for {len(user_arrangement)} colors")
            
            # Generate hue spectrum visualization
            hue_spectrum_bytes = graph_generator.generate_hue_error_visualization(
                user_order=user_arrangement,  # Which color INDEX at each position
                all_colors_rgb=all_patch_colors,  # ALL colors (including reference at index 0)
                title="Your Color Arrangement Errors"
            )
            
            # Convert to base64 for JSON response
            hue_spectrum_base64 = base64.b64encode(hue_spectrum_bytes).decode('utf-8')
            print(f"✓ Hue spectrum generated successfully")
            
        except Exception as e:
            print(f"⚠ Hue spectrum generation failed: {e}")
            traceback.print_exc()

        # Store results
        session["user_response"] = full_user_order
        session["classification"] = classification
        session["accuracy_score"] = accuracy_score
        session["report"] = report
        session["gemini_analysis"] = gemini_analysis
        session["d15_graph"] = d15_graph_base64
        session["hue_spectrum"] = hue_spectrum_base64

        return JSONResponse({
            "classification": classification,
            "accuracy_score": accuracy_score,
            "report": report,
            "gemini_analysis": gemini_analysis,
            "d15_graph": d15_graph_base64,  # Base64 encoded PNG
            "hue_spectrum": hue_spectrum_base64,  # Base64 encoded PNG
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)  # Disabled reload for stability