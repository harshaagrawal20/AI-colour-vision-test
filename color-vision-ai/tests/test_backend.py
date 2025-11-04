"""
Unit tests for color extraction, test generation, and classification.
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../backend'))

import numpy as np
import pytest
from color_extractor import ColorExtractor
from test_generator import TestGenerator
from error_analyzer import ErrorAnalyzer


class TestColorExtractor:
    def setup_method(self):
        self.extractor = ColorExtractor(n_colors=5)

    def test_load_image_from_array(self):
        """Test loading image from array."""
        # Create synthetic RGB image
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        loaded = self.extractor.load_image_from_array(img)
        assert loaded.shape == (100, 100, 3)
        assert np.array_equal(loaded, img)

    def test_rgb_to_lab_conversion(self):
        """Test RGB to LAB conversion."""
        # Test pure colors
        red = np.array([255, 0, 0], dtype=np.uint8)
        green = np.array([0, 255, 0], dtype=np.uint8)
        blue = np.array([0, 0, 255], dtype=np.uint8)
        white = np.array([255, 255, 255], dtype=np.uint8)
        
        lab_red = self.extractor.rgb_to_lab(red)
        lab_green = self.extractor.rgb_to_lab(green)
        lab_blue = self.extractor.rgb_to_lab(blue)
        lab_white = self.extractor.rgb_to_lab(white)
        
        # Check dimensions
        assert lab_red.shape == (1, 3)
        assert lab_green.shape == (1, 3)
        
        # White should have high L, low a/b
        assert lab_white[0, 0] > 95  # L > 95
        assert np.abs(lab_white[0, 1]) < 5  # a ~= 0
        assert np.abs(lab_white[0, 2]) < 5  # b ~= 0

    def test_lab_to_rgb_conversion(self):
        """Test LAB to RGB conversion produces valid output."""
        # Create realistic LAB colors
        lab_colors = np.array([
            [50.0, 20.0, 30.0],
            [75.0, -30.0, 40.0],
            [25.0, 10.0, -20.0],
        ], dtype=np.float32)
        
        # Convert to RGB
        rgb = self.extractor.lab_to_rgb(lab_colors)
        
        # Check shape and value ranges
        assert rgb.shape == (3, 3)
        assert np.all(rgb >= 0) and np.all(rgb <= 255)
        assert rgb.dtype == np.uint8
        
        # Verify it's not all zeros or one color
        assert np.std(rgb) > 0  # Has variation

    def test_extract_dominant_colors(self):
        """Test dominant color extraction."""
        # Create a simple image with 3 distinct colors
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[0:33, :] = [255, 0, 0]      # Red
        img[33:66, :] = [0, 255, 0]     # Green
        img[66:100, :] = [0, 0, 255]    # Blue
        
        extractor = ColorExtractor(n_colors=3)
        lab_colors, labels, inertia = extractor.extract_dominant_colors(img, convert_to_lab=True)
        
        assert lab_colors.shape == (3, 3)
        assert inertia >= 0

    def test_color_distance_lab(self):
        """Test color distance computation in LAB."""
        lab1 = np.array([[50, 0, 0]])
        lab2 = np.array([[50, 3, 4]])  # Distance should be 5 (3-4-5 triangle)
        
        dist = self.extractor.compute_color_distance_lab(lab1, lab2)
        assert np.isclose(dist[0], 5.0, atol=0.1)


class TestTestGenerator:
    def setup_method(self):
        self.generator = TestGenerator(test_type="patch_ordering")
        self.extractor = ColorExtractor(n_colors=6)
        
        # Create synthetic colors in LAB
        self.test_colors_lab = np.array([
            [50, 20, 30],
            [60, -15, 40],
            [70, 25, -20],
            [45, -30, 10],
            [65, 10, -35],
            [55, 35, 25],
        ])

    def test_generate_test(self):
        """Test basic test generation."""
        test_spec = self.generator.generate_test(self.test_colors_lab)
        
        assert test_spec["test_type"] == "patch_ordering"
        assert test_spec["n_colors"] == 6
        assert len(test_spec["reference_order"]) == 6
        assert len(test_spec["user_test_order"]) == 6
        assert len(test_spec["patch_configs"]) == 6
        assert "reference_colors_rgb" in test_spec

    def test_generate_distractor_test(self):
        """Test distractor test generation."""
        test_spec = self.generator.generate_distractor_test(self.test_colors_lab, n_distractors=2)
        
        assert test_spec["test_type"] == "distractor_matching"
        assert len(test_spec["groups"]) == 6
        
        for group in test_spec["groups"]:
            assert "reference_index" in group
            assert "options" in group
            assert "correct_index" in group
            assert len(group["options"]) == 3  # 1 reference + 2 distractors

    def test_generate_luminance_test(self):
        """Test luminance-equalized test generation."""
        test_spec = self.generator.generate_luminance_test(self.test_colors_lab)
        
        assert test_spec["test_type"] == "luminance_equalized"
        
        # All colors should have same luminance
        colors_lab = np.array(test_spec["reference_colors_lab"])
        luminance = colors_lab[:, 0]
        assert np.allclose(luminance, luminance[0], atol=0.1)


class TestErrorAnalyzer:
    def setup_method(self):
        self.analyzer = ErrorAnalyzer()
        
        # Create synthetic test scenario
        self.reference_colors_lab = np.array([
            [50, 20, 30],
            [60, -15, 40],
            [70, 25, -20],
            [45, -30, 10],
        ])
        self.reference_order = np.array([0, 1, 2, 3])

    def test_compute_error_metrics_perfect(self):
        """Test error metrics with perfect user response."""
        user_order = np.array([0, 1, 2, 3])  # Perfect
        
        metrics = self.analyzer.compute_error_metrics(
            self.reference_order, user_order, self.reference_colors_lab
        )
        
        assert metrics["mean_position_error"] == 0
        assert metrics["mean_color_diff"] == 0

    def test_compute_error_metrics_shuffled(self):
        """Test error metrics with shuffled response."""
        user_order = np.array([3, 2, 1, 0])  # Reversed
        
        metrics = self.analyzer.compute_error_metrics(
            self.reference_order, user_order, self.reference_colors_lab
        )
        
        assert metrics["mean_position_error"] > 0
        assert len(metrics["position_errors"]) == 4
        assert len(metrics["color_diffs_lab"]) == 4

    def test_compute_deficiency_scores(self):
        """Test deficiency score computation."""
        error_metrics = {
            "a_channel_error": 25.0,
            "b_channel_error": 5.0,
            "l_channel_error": 2.0,
            "mean_position_error": 2.0,
        }
        
        scores = self.analyzer.compute_deficiency_scores(error_metrics)
        
        assert "protan" in scores
        assert "deutan" in scores
        assert "tritan" in scores
        assert "normal" in scores
        assert "display_calibration" in scores
        
        # All scores should be between 0 and 1
        for score in scores.values():
            assert 0 <= score <= 1

    def test_classify_deficiency(self):
        """Test deficiency classification."""
        # Simulate a-channel heavy errors (red-green axis)
        error_metrics = {
            "a_channel_error": 30.0,
            "b_channel_error": 5.0,
            "l_channel_error": 2.0,
            "mean_position_error": 2.0,
            "position_errors": [1, 1, 1, 1],
            "color_diffs_lab": [5, 5, 5, 5],
            "mean_color_diff": 5.0,
            "max_position_error": 1,
        }
        
        classification = self.analyzer.classify_deficiency(error_metrics)
        
        assert "predicted_class" in classification
        assert "confidence" in classification
        assert "class_probabilities" in classification
        assert 0 <= classification["confidence"] <= 1

    def test_compute_color_accuracy_score(self):
        """Test color accuracy score computation."""
        error_metrics = {
            "a_channel_error": 10.0,
            "b_channel_error": 5.0,
            "l_channel_error": 2.0,
            "mean_position_error": 1.0,
            "position_errors": [1, 0, 0, 1],
            "color_diffs_lab": [2, 1, 1, 2],
            "mean_color_diff": 1.5,
            "max_position_error": 1,
        }
        
        accuracy = self.analyzer.compute_color_accuracy_score(error_metrics)
        
        assert 0 <= accuracy <= 100
        assert isinstance(accuracy, float)

    def test_generate_report(self):
        """Test report generation."""
        classification = {
            "predicted_class": "Normal",
            "confidence": 0.95,
        }
        accuracy_score = 92.5
        
        report = self.analyzer.generate_report(classification, accuracy_score)
        
        assert report["color_vision_type"] == "Normal"
        assert report["accuracy_score"] == 92.5
        assert "recommendation" in report
        assert "severity" in report


class TestIntegration:
    def test_end_to_end_workflow(self):
        """Test complete workflow from image to classification."""
        # Create synthetic image with dominant colors
        img = np.zeros((150, 150, 3), dtype=np.uint8)
        img[0:50, :] = [255, 0, 0]      # Red
        img[50:100, :] = [0, 255, 0]    # Green
        img[100:150, :] = [0, 0, 255]   # Blue
        
        # Extract colors
        extractor = ColorExtractor(n_colors=3)
        lab_colors, _, _ = extractor.extract_dominant_colors(img, convert_to_lab=True)
        
        # Generate test
        generator = TestGenerator()
        test_spec = generator.generate_test(lab_colors)
        
        assert test_spec["n_colors"] == 3
        
        # Simulate user response with small perturbation
        reference_order = np.array(test_spec["reference_order"])
        user_order = reference_order.copy()
        user_order[0], user_order[1] = user_order[1], user_order[0]  # Swap first two
        
        # Analyze errors
        analyzer = ErrorAnalyzer()
        error_metrics = analyzer.compute_error_metrics(
            reference_order, user_order, lab_colors
        )
        
        classification = analyzer.classify_deficiency(error_metrics)
        accuracy = analyzer.compute_color_accuracy_score(error_metrics)
        
        assert error_metrics["mean_position_error"] > 0
        assert 0 <= accuracy <= 100
        assert 0 <= classification["confidence"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
