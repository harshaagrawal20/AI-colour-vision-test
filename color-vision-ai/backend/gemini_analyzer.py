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
            # Compute vibgyor order analysis before building prompt
            vibgyor_result = self._calculate_hue_alignment(reference_colors_lab, user_order)
            
            # Build prompt (now includes vibgyor_result)
            prompt = self._build_analysis_prompt(
                reference_order,
                user_order,
                reference_colors_lab,
                classification,
                accuracy_score,
                vibgyor_result
            )
            
            # Generate Gemini analysis
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
    # VIBGYOR CHECK FUNCTION
    # ---------------------------------------------------------
    def _calculate_hue_alignment(self, reference_colors_lab, user_order):
        """
        Compare user's color sequence to the reference pad hues.
        Uses hue angle similarity (in LAB -> hue conversion) instead of fixed VIBGYOR order.
        """

        def lab_to_hue(lab):
            L, a, b = lab
            hue = math.degrees(math.atan2(b, a))
            if hue < 0:
                hue += 360
            return hue

        # Compute hue angles for both reference pad and user-selected order
        ref_hues = [lab_to_hue(c) for c in reference_colors_lab]
        user_hues = [ref_hues[i] for i in user_order]  # reorder based on user’s arrangement

        # Calculate average absolute hue difference from the perfect reference order
        total_diff = 0
        for i in range(len(ref_hues)):
            diff = abs(user_hues[i] - ref_hues[i])
            diff = min(diff, 360 - diff)  # handle hue wrap-around (e.g., 350° vs 10°)
            total_diff += diff
        avg_diff = total_diff / len(ref_hues)

        # Interpretation
        if avg_diff < 15:
            status = "Excellent alignment"
            message = "User closely followed the reference color sequence."
        elif avg_diff < 35:
            status = "Moderate alignment"
            message = "User’s order roughly follows the reference hues, with minor deviations."
        else:
            status = "Poor alignment"
            message = "User’s color arrangement significantly deviates from the reference hues."

        return {
            "status": status,
            "message": message,
            "average_hue_difference": round(avg_diff, 2),
            "reference_hues": [round(h, 1) for h in ref_hues],
            "user_hues": [round(h, 1) for h in user_hues],
        }


    # ---------------------------------------------------------
    # BUILD PROMPT WITH VIBGYOR SECTION
    # ---------------------------------------------------------
    def _build_analysis_prompt(
        self,
        reference_order: List[int],
        user_order: List[int],
        reference_colors_lab: List[List[float]],
        classification: Dict[str, Any],
        accuracy_score: float,
        hue_alignment: Dict[str, Any]
    ) -> str:
        # Calculate correct placements
        correct_count = sum(1 for i, (ref, usr) in enumerate(zip(reference_order, user_order)) if ref == usr)
        calculated_accuracy = (correct_count / len(reference_order)) * 100 if len(reference_order) > 0 else 0
        
        # Build color information
        colors_info = []
        for i, lab in enumerate(reference_colors_lab):
            colors_info.append(f"Color {i}: L={lab[0]:.1f}, a={lab[1]:.1f}, b={lab[2]:.1f}")
        
        # Build error list
        errors = []
        for i, (ref, usr) in enumerate(zip(reference_order, user_order)):
            if ref != usr:
                errors.append(f"Position {i}: Expected color {ref}, got color {usr}")
        
        hue_section = f"""
**Hue Angle Alignment Analysis**
- Status: {hue_alignment['status']}
- Details: {hue_alignment['message']}
- Average Hue Difference: {hue_alignment['average_hue_difference']}°
- Reference Hues: {hue_alignment['reference_hues']}
- User Hues: {hue_alignment['user_hues']}
"""
        
        prompt = f"""You are an expert ophthalmologist specializing in color vision analysis.
Analyze this Farnsworth D-15 test result and evaluate hue-based alignment with the reference pad.

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

{hue_section}

**Errors:** {chr(10).join(errors) if errors else "No errors detected."}

Please provide your analysis in the following format:

## Executive Summary

Provide a brief 1-2 sentence summary for each of the following 8 key points:

1. **Diagnosis:** [Brief diagnosis]
2. **Clinical Assessment:** [Key findings]
3. **Accuracy Analysis:** [Performance summary]
4. **Recommendations:** [Top recommendation]
5. **Medical Guidance:** [When to see a doctor]
6. **Career Guidance:** [Career considerations]
7. **Coping Strategies:** [Key tip]
8. **Technical Explanation:** [Main color confusion pattern]

---

## Detailed Analysis

### 1. Diagnosis
Provide a clear and comprehensive diagnosis based on the test results. Explain what type of color vision deficiency (if any) is indicated.

### 2. Clinical Assessment
Detailed analysis of the color arrangement pattern. Explain which colors were confused, what patterns emerged, and what this indicates clinically.

### 3. Accuracy Analysis
Interpret the accuracy score in detail. Explain what the {calculated_accuracy:.1f}% accuracy means, how it compares to normal vision, and what factors influenced this score.

### 4. Recommendations
Provide practical, actionable recommendations for the user. Include testing recommendations, lifestyle adjustments, and next steps.

### 5. Medical Guidance
Explain when professional medical consultation is needed, what to expect during a professional eye exam, and what treatments or interventions might be available.

### 6. Career Guidance
Discuss career implications and considerations. Include professions that may be affected, alternative career paths, and workplace accommodations.

### 7. Coping Strategies
Provide detailed daily life tips, assistive technologies, and practical tools for managing color vision challenges.

### 8. Technical Explanation
Provide a scientific explanation of the color confusion patterns observed. Explain the biology of color vision, which cone cells are affected, and why specific color confusions occur.

Keep the language clear, professional, and compassionate. Use markdown formatting for better readability.
"""
        return prompt

