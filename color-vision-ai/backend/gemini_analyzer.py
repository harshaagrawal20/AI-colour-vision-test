from dotenv import load_dotenv
import os

# Load .env file at startup
import google.generativeai as genai
from typing import Dict, List, Any
import math



class GeminiAnalyzer:
    def __init__(self, api_key: str = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass it to constructor.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    # ---------------------------------------------------------
    # 🔹 MAIN FUNCTION
    # ---------------------------------------------------------
    def analyze_color_arrangement(
        self,
        reference_order: List[int],
        user_order: List[int],
        reference_colors_lab: List[List[float]],
        classification: Dict[str, Any],
        accuracy_score: float
    ) -> Dict[str, Any]:
        try:
            # 1️⃣ Compute vibgyor order analysis before building prompt
            vibgyor_result = self._calculate_vibgyor_alignment(reference_colors_lab, user_order)
            
            # 2️⃣ Build prompt (now includes vibgyor_result)
            prompt = self._build_analysis_prompt(
                reference_order,
                user_order,
                reference_colors_lab,
                classification,
                accuracy_score,
                vibgyor_result
            )
            
            # 3️⃣ Generate Gemini analysis
            response = self.model.generate_content(prompt)
            analysis_text = response.text
            
            return {
                "ai_analysis": analysis_text,
                "vibgyor_check": vibgyor_result,
                "generated_by": "Gemini AI 2.0",
                "model": "gemini-2.0-flash",
                "success": True
            }

        except Exception as e:
            return {
                "ai_analysis": f"Unable to generate AI analysis: {str(e)}",
                "generated_by": "Error",
                "model": "gemini-2.0-flash",
                "success": False,
                "error": str(e)
            }

    # ---------------------------------------------------------
    # 🔹 VIBGYOR CHECK FUNCTION
    # ---------------------------------------------------------
    def _calculate_vibgyor_alignment(self, reference_colors_lab, user_order):
        """
        Calculate whether the user's order roughly matches the VIBGYOR hue sequence.
        """
        def lab_to_hue(lab):
            L, a, b = lab
            hue = math.degrees(math.atan2(b, a))
            if hue < 0:
                hue += 360
            return hue

        # Convert colors to hue values
        hues = [lab_to_hue(reference_colors_lab[i]) for i in user_order]
        
        # Check monotonic increase in hue values (approximation)
        increasing = all(hues[i] <= hues[i+1] + 10 for i in range(len(hues)-1))  # tolerance of ±10 degrees

        # Simple interpretation
        if increasing:
            return {
                "status": "Aligned",
                "message": "The user's color arrangement approximately follows the VIBGYOR order (Red → Violet).",
                "hues": [round(h, 1) for h in hues]
            }
        else:
            return {
                "status": "Not Aligned",
                "message": "The user's arrangement does not follow the expected VIBGYOR hue progression.",
                "hues": [round(h, 1) for h in hues]
            }

    # ---------------------------------------------------------
    # 🔹 BUILD PROMPT WITH VIBGYOR SECTION
    # ---------------------------------------------------------
    def _build_analysis_prompt(
        self,
        reference_order: List[int],
        user_order: List[int],
        reference_colors_lab: List[List[float]],
        classification: Dict[str, Any],
        accuracy_score: float,
        vibgyor_result: Dict[str, Any]
    ) -> str:
        colors_info = []
        for idx, color_lab in enumerate(reference_colors_lab):
            colors_info.append(f"Color {idx}: LAB({color_lab[0]:.1f}, {color_lab[1]:.1f}, {color_lab[2]:.1f})")

        # Calculate arrangement errors
        errors = []
        correct_count = 0
        for i, (ref, user) in enumerate(zip(reference_order, user_order)):
            if ref != user:
                errors.append(f"Position {i+1}: Expected color {ref}, got color {user}")
            else:
                correct_count += 1

        calculated_accuracy = (correct_count / len(reference_order)) * 100

        # ✅ NEW SECTION: VIBGYOR order info
        vibgyor_section = f"""
**VIBGYOR Order Analysis**
- Status: {vibgyor_result['status']}
- Details: {vibgyor_result['message']}
- Computed Hue Angles: {vibgyor_result['hues']}
"""

        prompt = f"""You are an expert ophthalmologist specializing in color vision analysis.
Analyze this Farnsworth D-15 test result and determine if the user arranged colors in proper perceptual order (VIBGYOR).

**Test Summary**
- Total Colors: {len(reference_order)}
- Correct Placements: {correct_count} / {len(reference_order)}
- Accuracy: {calculated_accuracy:.1f}%
- Classification: {classification}
- Accuracy Score from ML: {accuracy_score:.1f}%

**Color Data (LAB):**
{chr(10).join(colors_info)}

**Reference Order:** {reference_order}
**User Order:** {user_order}

{vibgyor_section}

**Errors:** {chr(10).join(errors) if errors else "No errors detected."}

Now, provide a full diagnostic analysis as before — include color deficiency diagnosis, accuracy interpretation, and practical recommendations.
If the VIBGYOR order is not followed, explain what color transitions are confused.
"""
        return prompt
