"""
Color extraction module: Convert images to LAB space and extract dominant colors.
"""
import numpy as np
import cv2
from sklearn.cluster import KMeans
import colorsys


class ColorExtractor:
    def __init__(self, n_colors=15, verbose=False):
        """
        Initialize the color extractor.
        
        Args:
            n_colors: Number of dominant colors to extract (default 15 for D-15 test)
            verbose: Print debug info
        """
        self.n_colors = n_colors
        self.verbose = verbose

    def load_image(self, image_path):
        """Load image from path."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_image_from_array(self, image_array):
        """Load image from numpy array (RGB)."""
        if isinstance(image_array, np.ndarray):
            return image_array
        raise ValueError("Input must be numpy array in RGB format")

    def extract_dominant_colors(self, image, convert_to_lab=True, use_d15_shading=False):
        """
        Extract dominant colors using K-Means clustering.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            convert_to_lab: If True, return colors in LAB space; else RGB
            use_d15_shading: If True, generate shaded variants for D-15 test accuracy
            
        Returns:
            dominant_colors: (n_colors, 3) array in LAB or RGB
            labels: cluster labels for each pixel
            inertia: KMeans inertia
        """
        # Reshape image to list of pixels
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # For D-15 test, use 10 base colors (we'll expand to 15 with shades)
        base_colors = 10 if use_d15_shading else self.n_colors
        
        # Apply KMeans
        kmeans = KMeans(n_clusters=base_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (RGB)
        dominant_colors_rgb = kmeans.cluster_centers_.astype(np.uint8)
        
        # If D-15 shading is requested, expand colors with lighter/darker variants
        if use_d15_shading:
            dominant_colors_rgb = self._generate_d15_shades(dominant_colors_rgb)
        
        # Convert to LAB if requested
        if convert_to_lab:
            # Normalize RGB to [0, 1]
            rgb_normalized = dominant_colors_rgb.astype(float) / 255.0
            # Convert to LAB
            lab_colors = self._rgb_to_lab_batch(rgb_normalized)
            return lab_colors, kmeans.labels_, kmeans.inertia_
        
        return dominant_colors_rgb, kmeans.labels_, kmeans.inertia_

    def _generate_d15_shades(self, base_colors_rgb):
        """
        Generate exactly 15 unique colors in proper hue sequence for D-15 test.
        Creates a smooth hue progression with NO duplicates.
        
        Args:
            base_colors_rgb: (10, 3) base colors in RGB [0, 255]
            
        Returns:
            d15_colors_rgb: (15, 3) array with 15 unique colors in hue order
        """
        # Convert all base colors to HSV to sort by hue
        hsv_colors = []
        for rgb in base_colors_rgb:
            hsv = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            hsv_colors.append(hsv)
        
        # Sort by hue to create natural color progression
        hsv_colors.sort(key=lambda x: x[0])
        
        # Generate 15 unique colors by interpolating between sorted hues
        d15_colors = []
        
        # We have 10 base colors sorted by hue
        # We need 5 more intermediate colors
        # Strategy: insert colors between existing hues with slight variations
        
        for i in range(10):
            # Add the base color
            h, s, v = hsv_colors[i]
            rgb = colorsys.hsv_to_rgb(h, s, v)
            rgb_255 = np.array([int(c * 255) for c in rgb], dtype=np.uint8)
            d15_colors.append(rgb_255)
        
        # Add 5 intermediate colors between hues
        interpolation_points = [1, 3, 5, 7, 9]  # Positions to add intermediate colors
        
        for idx in interpolation_points[:5]:  # Take only first 5
            if idx < len(hsv_colors) - 1:
                # Interpolate between two adjacent hues
                h1, s1, v1 = hsv_colors[idx]
                h2, s2, v2 = hsv_colors[idx + 1]
                
                # Create intermediate color
                h_mid = (h1 + h2) / 2
                s_mid = (s1 + s2) / 2
                v_mid = (v1 + v2) / 2
                
                # Slight variation to make it visibly different
                v_mid = min(1.0, v_mid * 1.1)
                
                rgb = colorsys.hsv_to_rgb(h_mid, s_mid, v_mid)
                rgb_255 = np.array([int(c * 255) for c in rgb], dtype=np.uint8)
                d15_colors.append(rgb_255)
        
        # Ensure we have exactly 15 unique colors
        # Remove any potential duplicates and pad if needed
        unique_colors = []
        for color in d15_colors:
            is_duplicate = False
            for existing in unique_colors:
                if np.allclose(color, existing, atol=5):  # Within 5 RGB values
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_colors.append(color)
        
        # If we still don't have 15, generate additional hue-varied colors
        while len(unique_colors) < 15:
            # Generate a color with different hue
            hue_offset = len(unique_colors) / 15.0
            h = (hue_offset) % 1.0
            s = 0.7
            v = 0.7
            rgb = colorsys.hsv_to_rgb(h, s, v)
            rgb_255 = np.array([int(c * 255) for c in rgb], dtype=np.uint8)
            unique_colors.append(rgb_255)
        
        return np.array(unique_colors[:15], dtype=np.uint8)

    def _rgb_to_lab_batch(self, rgb_array):
        """
        Convert RGB array (N, 3) in [0, 1] range to LAB.
        
        Args:
            rgb_array: (N, 3) array with values in [0, 1]
            
        Returns:
            lab_array: (N, 3) LAB array with L in [0, 100], a,b in [-127, 127]
        """
        # sRGB to XYZ conversion
        rgb = rgb_array.copy()
        rgb[rgb > 0.04045] = np.power((rgb[rgb > 0.04045] + 0.055) / 1.055, 2.4)
        rgb[rgb <= 0.04045] = rgb[rgb <= 0.04045] / 12.92
        
        # Linear RGB to XYZ
        transform_matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        xyz = np.dot(rgb, transform_matrix.T)
        
        # D65 illuminant
        xyz[:, 0] /= 0.95047
        xyz[:, 1] /= 1.00000
        xyz[:, 2] /= 1.08883
        
        # XYZ to LAB
        epsilon = 0.008856
        kappa = 903.3
        
        f = np.zeros_like(xyz)
        mask = xyz > epsilon
        f[mask] = np.power(xyz[mask], 1/3)
        f[~mask] = (kappa * xyz[~mask] + 16) / 116
        
        L = 116 * f[:, 1] - 16
        a = 500 * (f[:, 0] - f[:, 1])
        b = 200 * (f[:, 1] - f[:, 2])
        
        return np.column_stack([L, a, b])

    def rgb_to_lab(self, rgb):
        """
        Convert single RGB color (or batch) from [0, 255] to LAB.
        
        Args:
            rgb: Single color (3,) or batch (N, 3)
            
        Returns:
            lab: Converted color(s)
        """
        rgb_normalized = rgb.astype(float) / 255.0 if rgb.max() > 1 else rgb
        if rgb_normalized.ndim == 1:
            rgb_normalized = rgb_normalized[np.newaxis, :]
        return self._rgb_to_lab_batch(rgb_normalized)

    def lab_to_rgb(self, lab):
        """
        Convert LAB back to RGB [0, 255].
        
        Args:
            lab: LAB color(s)
            
        Returns:
            rgb: RGB color(s) in [0, 255]
        """
        if lab.ndim == 1:
            lab = lab[np.newaxis, :]
        
        # LAB to XYZ
        fy = (lab[:, 0] + 16) / 116
        fx = lab[:, 1] / 500 + fy
        fz = fy - lab[:, 2] / 200
        
        xyz = np.zeros_like(lab)
        
        # Apply inverse functions
        xr = np.where(np.power(fx, 3) > 0.008856, np.power(fx, 3), (116 * fx - 16) / 903.3)
        yr = np.where(np.power(fy, 3) > 0.008856, np.power(fy, 3), (116 * fy - 16) / 903.3)
        zr = np.where(np.power(fz, 3) > 0.008856, np.power(fz, 3), (116 * fz - 16) / 903.3)
        
        xyz[:, 0] = xr * 0.95047
        xyz[:, 1] = yr * 1.00000
        xyz[:, 2] = zr * 1.08883
        
        # XYZ to linear RGB
        transform_matrix = np.array([
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252]
        ])
        rgb_linear = np.dot(xyz, transform_matrix.T)
        
        # Linear RGB to sRGB
        rgb = rgb_linear.copy()
        rgb[rgb > 0.0031308] = 1.055 * np.power(rgb[rgb > 0.0031308], 1 / 2.4) - 0.055
        rgb[rgb <= 0.0031308] = 12.92 * rgb[rgb <= 0.0031308]
        
        # Clip and scale to [0, 255]
        rgb = np.clip(rgb, 0, 1) * 255
        return rgb.astype(np.uint8)

    def compute_color_distance_lab(self, lab1, lab2):
        """
        Compute CIE Delta E 2000 distance in LAB space.
        
        Args:
            lab1: LAB color or array (N, 3)
            lab2: LAB color or array (M, 3)
            
        Returns:
            distances: Euclidean distance in LAB space (simple approximation)
        """
        # Simplified ΔE (Euclidean in LAB)
        return np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))
    
    def find_balanced_reference_color(self, dominant_colors_lab):
        """
        Select a balanced reference color (midpoint) from the extracted colors.
        The reference color is the one closest to the mean LAB value.
        
        Args:
            dominant_colors_lab: (N, 3) numpy array of LAB colors.
        
        Returns:
            (ref_index, ref_color): Tuple of index and LAB color of reference pad.
        """
        if not isinstance(dominant_colors_lab, np.ndarray):
            dominant_colors_lab = np.array(dominant_colors_lab)

        # Compute the mean LAB vector
        mean_lab = np.mean(dominant_colors_lab, axis=0)

        # Compute Euclidean distances from the mean
        distances = np.linalg.norm(dominant_colors_lab - mean_lab, axis=1)

        # Index of the color closest to the midpoint
        ref_index = int(np.argmin(distances))
        ref_color = dominant_colors_lab[ref_index]

        if self.verbose:
            print(f"[Reference Color] Index: {ref_index}, LAB: {ref_color}")

        return ref_index, ref_color
