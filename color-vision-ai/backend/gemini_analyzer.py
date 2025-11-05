"""
Gemini AI analyzer for color vision test results.
Uses Google's Gemini AI to provide detailed analysis and recommendations.
"""
import os
import google.generativeai as genai
from typing import Dict, List, Any
import json


class GeminiAnalyzer:
    def __init__(self, api_key: str = None):
        """
        Initialize Gemini analyzer.
        
        Args:
            api_key: Gemini API key (if not provided, reads from environment)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass it to constructor.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Use Gemini 2.0 Flash - Latest fast model with great performance
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
    def analyze_color_arrangement(
        self,
        reference_order: List[int],
        user_order: List[int],
        reference_colors_lab: List[List[float]],
        classification: Dict[str, Any],
        accuracy_score: float
    ) -> Dict[str, Any]:
        """
        Analyze user's color arrangement using Gemini AI.
        
        Args:
            reference_order: Correct color order (indices)
            user_order: User's ordered indices
            reference_colors_lab: Original LAB colors
            classification: ML classification results
            accuracy_score: Color accuracy score
            
        Returns:
            Dictionary with AI analysis and recommendations
        """
        try:
            # Prepare prompt for Gemini
            prompt = self._build_analysis_prompt(
                reference_order,
                user_order,
                reference_colors_lab,
                classification,
                accuracy_score
            )
            
            # Generate analysis using Gemini
            response = self.model.generate_content(prompt)
            
            # Parse response
            analysis_text = response.text
            
            # Structure the response
            result = {
                "ai_analysis": analysis_text,
                "generated_by": "Gemini AI 2.0",
                "model": "gemini-2.0-flash",
                "success": True
            }
            
            return result
            
        except Exception as e:
            return {
                "ai_analysis": f"Unable to generate AI analysis: {str(e)}",
                "generated_by": "Error",
                "model": "gemini-2.0-flash",
                "success": False,
                "error": str(e)
            }
    
    def _build_analysis_prompt(
        self,
        reference_order: List[int],
        user_order: List[int],
        reference_colors_lab: List[List[float]],
        classification: Dict[str, Any],
        accuracy_score: float
    ) -> str:
        """
        Build comprehensive prompt for Gemini AI analysis.
        
        Args:
            reference_order: Correct color order
            user_order: User's ordered indices
            reference_colors_lab: LAB color values
            classification: Classification results (can be None for pure AI analysis)
            accuracy_score: Accuracy score (can be None, AI will calculate)
            
        Returns:
            Formatted prompt string
        """
        # Convert LAB colors to readable format
        colors_info = []
        for idx, color_lab in enumerate(reference_colors_lab):
            colors_info.append(f"Color {idx}: LAB({color_lab[0]:.1f}, {color_lab[1]:.1f}, {color_lab[2]:.1f})")
        
        # Calculate error positions and accuracy
        errors = []
        correct_count = 0
        for i, (ref, user) in enumerate(zip(reference_order, user_order)):
            if ref != user:
                errors.append(f"Position {i+1}: Expected color {ref}, got color {user}")
            else:
                correct_count += 1
        
        calculated_accuracy = (correct_count / len(reference_order)) * 100
        
        prompt = f"""You are an expert ophthalmologist specializing in color vision deficiencies. Analyze this D-15 color vision test result and provide a COMPLETE, COMPREHENSIVE analysis.

**Test Information:**
- Test Type: Farnsworth D-15 (Dichotomous Test for Color Blindness)
- Number of Colors: {len(reference_order)}
- Correct Placements: {correct_count} out of {len(reference_order)}
- Accuracy: {calculated_accuracy:.1f}%

**Color Data (LAB Color Space):**
{chr(10).join(colors_info)}

**Reference Order (Correct Hue Progression):**
{reference_order}

**User's Arrangement:**
{user_order}

**Arrangement Errors ({len(errors)} total):**
{chr(10).join(errors) if errors else 'No errors - PERFECT arrangement! This indicates normal color vision.'}

**YOUR COMPLETE ANALYSIS MUST INCLUDE:**

## 1. Color Vision Diagnosis
- What type of color vision deficiency (if any): Normal, Protanopia, Deuteranopia, Tritanopia, or Monochromacy?
- Severity level: None, Mild, Moderate, Severe, or Complete
- Confidence in your diagnosis
- Explain the specific error pattern

## 2. Clinical Assessment
- Analyze the arrangement pattern in detail
- Identify confusion axes (red-green, blue-yellow, or none)
- Which wavelengths/hues are confused?
- What does this reveal about cone photoreceptor function (L, M, S cones)?

## 3. Accuracy Score Analysis
- Interpret the {calculated_accuracy:.1f}% accuracy score
- Is this within normal range or indicates deficiency?
- Compare to typical scores for different conditions

## 4. Personalized Recommendations
- **Daily Life:** Specific tips for work, driving, clothing selection
- **Technology:** Recommend specific apps (Color Blind Pal, CVSimulator, Chromatic Vision Simulator)
- **Workplace:** How to inform employers and request accommodations
- **Safety:** Situations requiring extra caution

## 5. Medical Guidance
- Is this likely congenital (genetic) or acquired?
- Should they see an ophthalmologist? When?
- Red flags that require immediate medical attention
- Recommended follow-up tests

## 6. Career & Education Guidance
- Careers to consider or avoid
- Educational accommodations available
- Fields where color vision is critical

## 7. Positive Insights & Coping Strategies
- What they CAN do well
- Success stories of people with similar vision
- Practical coping mechanisms
- Mental health and acceptance

## 8. Technical Explanation
- Explain how the D-15 test works
- Why their specific errors indicate their diagnosis
- Color science behind their vision

**IMPORTANT:** 
- Provide DETAILED, SPECIFIC analysis - not generic advice
- Use warm, empathetic, encouraging tone
- Give ACTIONABLE recommendations
- Be medically accurate but easy to understand
- Format with clear headers and bullet points"""
        
        return prompt
    
    def get_quick_insights(
        self,
        classification: Dict[str, Any],
        accuracy_score: float
    ) -> str:
        """
        Get quick AI insights without detailed analysis.
        
        Args:
            classification: Classification results
            accuracy_score: Accuracy score
            
        Returns:
            Quick insight string
        """
        try:
            prompt = f"""As a color vision expert, provide a brief 2-3 sentence insight about this result:
- Deficiency: {classification.get('deficiency_type', 'Unknown')}
- Severity: {classification.get('severity', 'Unknown')}
- Accuracy: {accuracy_score:.1f}%

Keep it concise, empathetic, and actionable."""
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Quick insight unavailable: {str(e)}"
    
    def explain_deficiency_type(self, deficiency_type: str) -> str:
        """
        Get detailed explanation of a specific deficiency type.
        
        Args:
            deficiency_type: Type of deficiency
            
        Returns:
            Detailed explanation
        """
        try:
            prompt = f"""As a color vision expert, explain {deficiency_type} in detail:
1. What causes it?
2. What colors are affected?
3. How common is it?
4. How does it impact daily life?
5. Are there treatments or assistive tools?

Provide a clear, educational explanation suitable for patients."""
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Explanation unavailable: {str(e)}"
