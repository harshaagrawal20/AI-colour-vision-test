"""
Error analysis and deficiency classifier module.
Detects color vision deficiencies (Protan, Deutan, Tritan) from user responses.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json


class ErrorAnalyzer:
    def __init__(self):
        """Initialize error analyzer and classifier."""
        self.scaler = StandardScaler()
        self.clf = None
        self._init_classifier()

    def _init_classifier(self):
        """
        Initialize a simple rule-based + ML classifier.
        For now, we use a pre-configured RandomForest based on known patterns.
        """
        # Dummy training: In production, train on real data
        self.clf = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Synthetic training data: error patterns and labels
        # Features: [protanopia, deuteranopia, tritanopia, monochromacy, normal]
        # Labels: 0=Normal, 1=Protanopia, 2=Deuteranopia, 3=Tritanopia, 4=Monochromacy
        X_train = np.array([
            [0.1, 0.1, 0.1, 0.05, 0.9],    # Normal
            [0.1, 0.1, 0.1, 0.08, 0.85],   # Normal
            [0.8, 0.3, 0.2, 0.2, 0.1],     # Protanopia
            [0.3, 0.8, 0.2, 0.2, 0.1],     # Deuteranopia
            [0.2, 0.2, 0.9, 0.1, 0.05],    # Tritanopia
            [0.4, 0.4, 0.4, 0.9, 0.05],    # Monochromacy
        ])
        y_train = np.array([0, 0, 1, 2, 3, 4])
        
        self.clf.fit(X_train, y_train)
        self.scaler.fit(X_train)

    def compute_error_metrics(self, reference_order, user_order, reference_colors_lab):
        """
        Compute error metrics from user's ordering vs reference.
        
        Args:
            reference_order: Correct color order (indices)
            user_order: User's ordered indices
            reference_colors_lab: Original LAB colors
            
        Returns:
            metrics: Dictionary with position errors, distance errors, etc.
        """
        metrics = {}
        
        # Position error (how far each color ended up from correct position)
        position_errors = np.abs(np.argsort(user_order) - np.argsort(reference_order))
        metrics["position_errors"] = position_errors.tolist()
        metrics["mean_position_error"] = float(np.mean(position_errors))
        metrics["max_position_error"] = int(np.max(position_errors))
        
        # Color distance errors in LAB space
        user_colors_sorted = reference_colors_lab[user_order]
        ref_colors_sorted = reference_colors_lab[reference_order]
        
        # Compute ΔE (Euclidean in LAB)
        color_diffs = np.sqrt(np.sum((user_colors_sorted - ref_colors_sorted) ** 2, axis=1))
        metrics["color_diffs_lab"] = color_diffs.tolist()
        metrics["mean_color_diff"] = float(np.mean(color_diffs))
        
        # Analyze which channels have largest errors
        l_errors = np.abs(user_colors_sorted[:, 0] - ref_colors_sorted[:, 0])
        a_errors = np.abs(user_colors_sorted[:, 1] - ref_colors_sorted[:, 1])
        b_errors = np.abs(user_colors_sorted[:, 2] - ref_colors_sorted[:, 2])
        
        metrics["l_channel_error"] = float(np.mean(l_errors))  # Luminance
        metrics["a_channel_error"] = float(np.mean(a_errors))  # Red-Green axis
        metrics["b_channel_error"] = float(np.mean(b_errors))  # Yellow-Blue axis
        
        return metrics

    def compute_deficiency_scores(self, error_metrics):
        """
        Map error patterns to deficiency probabilities using color science.
        
        Protanopia: Loss of L-cones (Red) → Reds appear dark, red-green confusion
        Deuteranopia: Loss of M-cones (Green) → Greens appear dark, red-green confusion  
        Tritanopia: Loss of S-cones (Blue) → Blues/yellows appear wrong
        Monochromacy: Loss of all color cones → Complete grayscale vision
        
        Args:
            error_metrics: Output from compute_error_metrics
            
        Returns:
            scores: Dictionary with deficiency type scores
        """
        a_error = error_metrics["a_channel_error"]  # Red-Green axis
        b_error = error_metrics["b_channel_error"]  # Yellow-Blue axis
        l_error = error_metrics["l_channel_error"]  # Luminance
        position_error = error_metrics["mean_position_error"]
        
        scores = {}
        
        # Normalize errors for consistent scoring
        total_color_error = a_error + b_error + 1e-6
        
        # Red-Green axis dominance ratio
        rg_dominance = a_error / total_color_error
        
        # Yellow-Blue axis dominance ratio
        yb_dominance = b_error / total_color_error
        
        # Luminance-to-color error ratio (high = monochromacy tendency)
        luminance_dominance = l_error / total_color_error
        
        # ==== PROTANOPIA (Red cone loss) ====
        # Characteristics: Red-Green errors prominent, but with specific pattern
        # Reds confused with yellows/greens, high luminance errors on reds
        protan_score = (rg_dominance * 0.7) + (1 - yb_dominance) * 0.3
        scores["protanopia"] = float(np.clip(protan_score, 0, 1))
        
        # ==== DEUTERANOPIA (Green cone loss) ====
        # Characteristics: Red-Green errors prominent, similar to protan but inverted
        # Greens confused with reds, red-green confusion pattern
        deutan_score = (rg_dominance * 0.7) + (luminance_dominance * 0.2) + 0.1
        scores["deuteranopia"] = float(np.clip(deutan_score, 0, 1))
        
        # ==== TRITANOPIA (Blue cone loss) ====
        # Characteristics: Yellow-Blue axis errors dominant
        # Blues/yellows show most confusion, a-channel errors minimal
        tritan_score = (yb_dominance * 0.8) + (1 - rg_dominance) * 0.2
        scores["tritanopia"] = float(np.clip(tritan_score, 0, 1))
        
        # ==== MONOCHROMACY (Complete color blindness) ====
        # Characteristics: High luminance errors, very low distinguishable color vision
        # All channels show similar errors, arrangement mostly based on brightness
        monochromacy_score = (luminance_dominance * 0.6) + (1 - (rg_dominance * yb_dominance)) * 0.4
        scores["monochromacy"] = float(np.clip(monochromacy_score, 0, 1))
        
        # ==== NORMAL (No deficiency) ====
        # Characteristics: Low errors overall, balanced axis errors
        total_error_norm = (a_error + b_error + l_error) / 100
        normal_score = 1 - min(total_error_norm, 1.0)
        scores["normal"] = float(np.clip(normal_score, 0, 1))
        
        return scores

    def classify_deficiency(self, error_metrics):
        """
        Classify the color vision deficiency based on error patterns.
        
        Args:
            error_metrics: Output from compute_error_metrics
            
        Returns:
            classification: Dictionary with predicted class and confidence
        """
        deficiency_scores = self.compute_deficiency_scores(error_metrics)
        
        # Prepare features for ML classifier
        features = np.array([
            deficiency_scores["protanopia"],
            deficiency_scores["deuteranopia"],
            deficiency_scores["tritanopia"],
            deficiency_scores["monochromacy"],
            deficiency_scores["normal"],
        ]).reshape(1, -1)
        
        features_scaled = self.scaler.transform(features)
        prediction = self.clf.predict(features_scaled)[0]
        probabilities = self.clf.predict_proba(features_scaled)[0]
        
        class_names = ["Normal", "Protanopia", "Deuteranopia", "Tritanopia", "Monochromacy"]
        
        classification = {
            "predicted_class": class_names[prediction],
            "class_probabilities": {name: float(prob) for name, prob in zip(class_names, probabilities)},
            "confidence": float(np.max(probabilities)),
            "deficiency_scores": deficiency_scores,
            "error_metrics": error_metrics,
        }
        
        return classification

    def compute_color_accuracy_score(self, error_metrics):
        """
        Compute an overall color accuracy score (0-100%).
        
        Args:
            error_metrics: Output from compute_error_metrics
            
        Returns:
            accuracy_score: Score between 0 and 100
        """
        # Normalize various errors and compute overall accuracy
        position_error_norm = error_metrics["mean_position_error"] / 10  # Max ~1.0
        color_error_norm = error_metrics["mean_color_diff"] / 100  # Max ~1.0
        
        # Accuracy is inverse of total normalized error
        total_error = (position_error_norm + color_error_norm) / 2
        accuracy = max(0, 100 * (1 - total_error))
        
        return float(accuracy)

    def generate_report(self, classification, accuracy_score):
        """
        Generate a human-readable report with detailed recommendations.
        
        Args:
            classification: Output from classify_deficiency
            accuracy_score: Output from compute_color_accuracy_score
            
        Returns:
            report: Dictionary with recommendations
        """
        predicted_class = classification["predicted_class"]
        confidence = classification["confidence"]
        severity = self._confidence_to_severity(confidence)
        
        report = {
            "color_vision_type": predicted_class,
            "severity": severity,
            "accuracy_score": accuracy_score,
            "confidence": confidence,
        }
        
        # Add detailed recommendations based on deficiency type and severity
        if predicted_class == "Normal":
            report["recommendation"] = (
                "✓ Normal Color Vision\n"
                "Your color vision appears normal. You have typical trichromatic color perception. "
                "No intervention needed."
            )
            report["description"] = "Your eyes have normal L, M, and S cone function."
            
        elif predicted_class == "Protanopia":
            report["recommendation"] = (
                f"🔴 Red-Green Color Blindness (Protanopia) - {severity}\n"
                "You have reduced or absent red cone (L-cone) function.\n"
                "• Reds appear darker or as dark yellow-greens\n"
                "• You may confuse reds with greens, especially in low light\n"
                "• Severity: {}\n"
                "• Recommendations:\n"
                "  - Use color-blind friendly palettes (blue/yellow/gray)\n"
                "  - Enable accessibility tools on your devices\n"
                "  - Consult an eye care professional for confirmation\n"
                "  - Special tinted lenses may help in some cases"
            ).format(severity)
            report["description"] = (
                "Protanopia (Red-cone deficiency) affects ~1% of males. "
                "It's the most severe form of red-green color blindness."
            )
            
        elif predicted_class == "Deuteranopia":
            report["recommendation"] = (
                f"🟢 Red-Green Color Blindness (Deuteranopia) - {severity}\n"
                "You have reduced or absent green cone (M-cone) function.\n"
                "• Greens appear darker or as dark red-browns\n"
                "• You may confuse reds with greens\n"
                "• Severity: {}\n"
                "• Recommendations:\n"
                "  - Use color-blind friendly palettes (blue/yellow/gray)\n"
                "  - Similar accommodations as Protanopia\n"
                "  - Enable color-blind mode on devices\n"
                "  - Consider consultation with a color vision specialist"
            ).format(severity)
            report["description"] = (
                "Deuteranopia (Green-cone deficiency) affects ~1% of males. "
                "Similar in appearance to Protanopia but with different cone affected."
            )
            
        elif predicted_class == "Tritanopia":
            report["recommendation"] = (
                f"🔵 Blue-Yellow Color Blindness (Tritanopia) - {severity}\n"
                "You have reduced or absent blue cone (S-cone) function.\n"
                "• Blues appear as pinks or light purples\n"
                "• Yellows appear as pinks or pale reds\n"
                "• Severity: {}\n"
                "• Recommendations:\n"
                "  - This is very rare (affects <0.01% of people)\n"
                "  - Consult an eye care specialist immediately\n"
                "  - Use red/green based color schemes\n"
                "  - Special color correction glasses may help\n"
                "  - Professional diagnosis strongly recommended"
            ).format(severity)
            report["description"] = (
                "Tritanopia (Blue-cone deficiency) is extremely rare. "
                "It's often acquired rather than congenital."
            )
            
        elif predicted_class == "Monochromacy":
            report["recommendation"] = (
                f"⚫ Complete Color Blindness (Achromatopsia) - {severity}\n"
                "You have very limited or no color vision ability.\n"
                "• You perceive the world in shades of gray and brightness\n"
                "• All colors appear as similar tones\n"
                "• Severity: Complete\n"
                "• Recommendations:\n"
                "  - Seek immediate consultation with an ophthalmologist\n"
                "  - This may indicate a serious eye condition\n"
                "  - Explore specialized accessibility tools\n"
                "  - Consider genetic counseling\n"
                "  - Professional medical evaluation is essential"
            )
            report["description"] = (
                "Complete color blindness (Monochromacy/Achromatopsia) is rare. "
                "It requires professional medical attention."
            )
        
        return report

    def _confidence_to_severity(self, confidence):
        """
        Map confidence to severity level for color vision deficiency.
        
        Confidence = probability that the detected deficiency is present
        - Low confidence (<0.5): Mild or no deficiency
        - Medium confidence (0.5-0.75): Moderate deficiency
        - High confidence (0.75-0.9): Severe deficiency
        - Very high confidence (>0.9): Complete/Total deficiency
        """
        if confidence < 0.5:
            return "Mild"
        elif confidence < 0.7:
            return "Moderate"
        elif confidence < 0.85:
            return "Severe"
        else:
            return "Complete"
