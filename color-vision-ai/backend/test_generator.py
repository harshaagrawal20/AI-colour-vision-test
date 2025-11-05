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

    def generate_test(self, dominant_colors_lab):
        """
        Generate a simplified color vision test from dominant colors in LAB space.
        (No reference pad or reference color.)
        """
        n_colors = len(dominant_colors_lab)

        # Randomized user order
        user_order = np.random.permutation(n_colors)

        # Convert LAB → RGB
        rgb_colors = self.color_extractor.lab_to_rgb(dominant_colors_lab)

        # Build test specification
        test_spec = {
            "test_type": self.test_type,
            "n_colors": n_colors,
            "colors_lab": dominant_colors_lab.tolist(),
            "colors_rgb": rgb_colors.tolist(),
            "user_test_order": [int(x) for x in user_order],
            "message": "Color vision test generated successfully (no reference pad)",
            "patch_configs": [
                {
                    "color_index": int(idx),
                    "lab": dominant_colors_lab[idx].tolist(),
                    "rgb": rgb_colors[idx].tolist(),
                    "luminance": float(dominant_colors_lab[idx, 0]),
                }
                for idx in user_order
            ],
        }

        return test_spec


    def generate_distractor_test(self, dominant_colors_lab, n_distractors=3):
        """
        Generate a test with distractor colors (simulating harder discrimination).
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
                distractor_lab[1] += np.random.uniform(-20, 20)
                distractor_lab[2] += np.random.uniform(-20, 20)
                distractor_lab[1] = np.clip(distractor_lab[1], -127, 127)
                distractor_lab[2] = np.clip(distractor_lab[2], -127, 127)
                distractors.append(distractor_lab.tolist())
            
            options = [color_lab.tolist()] + distractors
            option_indices = np.random.permutation(len(options))
            group["options"] = [options[idx] for idx in option_indices]
            group["correct_index"] = int(np.where(option_indices == 0)[0][0])
            
            test_spec["groups"].append(group)
        
        return test_spec

    def generate_luminance_test(self, dominant_colors_lab):
        """
        Generate a luminance-based test (control for brightness differences).
        """
        normalized_colors = dominant_colors_lab.copy()
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
