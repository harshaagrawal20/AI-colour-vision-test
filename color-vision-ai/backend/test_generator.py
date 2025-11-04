"""
Test generation module: Create dynamic color vision tests from extracted colors.
"""
import numpy as np
from color_extractor import ColorExtractor


class TestGenerator:
    def __init__(self, test_type="patch_ordering"):
        """
        Initialize test generator.
        
        Args:
            test_type: 'patch_ordering' (arrange colors) or 'matching' (identify similar)
        """
        self.test_type = test_type
        self.color_extractor = ColorExtractor()

    def generate_test(self, dominant_colors_lab, reference_order=None):
        """
        Generate a color vision test from dominant colors in LAB space.
        
        Args:
            dominant_colors_lab: (N, 3) array of colors in LAB space
            reference_order: If None, use given order; else shuffle for user test
            
        Returns:
            test_spec: Dictionary with test metadata and visual spec
        """
        n_colors = len(dominant_colors_lab)
        
        if reference_order is None:
            reference_order = np.arange(n_colors)
        
        # Shuffle for user test
        user_order = np.random.permutation(n_colors)
        
        test_spec = {
            "test_type": self.test_type,
            "reference_colors_lab": dominant_colors_lab.tolist(),
            "reference_order": reference_order.tolist(),
            "user_test_order": user_order.tolist(),
            "n_colors": n_colors,
        }
        
        # Add visual properties for frontend
        rgb_colors = self.color_extractor.lab_to_rgb(dominant_colors_lab)
        test_spec["reference_colors_rgb"] = rgb_colors.tolist()
        
        # Add patch sizes and luminance info
        test_spec["patch_configs"] = [
            {
                "color_index": int(idx),
                "lab": dominant_colors_lab[idx].tolist(),
                "rgb": rgb_colors[idx].tolist(),
                "luminance": float(dominant_colors_lab[idx, 0]),  # L value
            }
            for idx in user_order
        ]
        
        return test_spec

    def generate_distractor_test(self, dominant_colors_lab, n_distractors=3):
        """
        Generate a test with distractor colors (simulating harder discrimination).
        
        Args:
            dominant_colors_lab: (N, 3) reference colors in LAB
            n_distractors: Number of distractor colors per reference
            
        Returns:
            test_spec: Dictionary with reference + distractor configs
        """
        n_colors = len(dominant_colors_lab)
        test_spec = {
            "test_type": "distractor_matching",
            "reference_colors_lab": dominant_colors_lab.tolist(),
            "reference_colors_rgb": self.color_extractor.lab_to_rgb(dominant_colors_lab).tolist(),
            "n_colors": n_colors,
            "groups": []
        }
        
        for i, color_lab in enumerate(dominant_colors_lab):
            group = {"reference_index": i, "reference_lab": color_lab.tolist()}
            
            # Create distractors by perturbing in a, b channels
            distractors = []
            for j in range(n_distractors):
                distractor_lab = color_lab.copy()
                # Add random perturbation in color opponent channels
                distractor_lab[1] += np.random.uniform(-20, 20)  # a channel
                distractor_lab[2] += np.random.uniform(-20, 20)  # b channel
                # Clip to valid range
                distractor_lab[1] = np.clip(distractor_lab[1], -127, 127)
                distractor_lab[2] = np.clip(distractor_lab[2], -127, 127)
                distractors.append(distractor_lab.tolist())
            
            # Shuffle reference + distractors
            options = [color_lab.tolist()] + distractors
            option_indices = np.random.permutation(len(options))
            group["options"] = [options[idx] for idx in option_indices]
            group["correct_index"] = int(np.where(option_indices == 0)[0][0])
            
            test_spec["groups"].append(group)
        
        return test_spec

    def generate_luminance_test(self, dominant_colors_lab):
        """
        Generate a luminance-based test (control for brightness differences).
        
        Args:
            dominant_colors_lab: (N, 3) colors in LAB
            
        Returns:
            test_spec: Test with equalized luminance
        """
        normalized_colors = dominant_colors_lab.copy()
        
        # Equalize luminance to median
        median_l = np.median(dominant_colors_lab[:, 0])
        normalized_colors[:, 0] = median_l
        
        test_spec = {
            "test_type": "luminance_equalized",
            "reference_colors_lab": normalized_colors.tolist(),
            "reference_colors_rgb": self.color_extractor.lab_to_rgb(normalized_colors).tolist(),
            "n_colors": len(dominant_colors_lab),
            "luminance_level": float(median_l),
        }
        
        return test_spec
