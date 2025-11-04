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
        Generate D-15 test colors by creating lighter and darker shades.
        This creates similar-hue color groups which are challenging for color vision deficiency detection.
        
        Args:
            base_colors_rgb: (10, 3) base colors in RGB [0, 255]
            
        Returns:
            d15_colors_rgb: (15, 3) array with original + shaded variants
        """
        d15_colors = []
        
        for base_color in base_colors_rgb:
            # Add original color
            d15_colors.append(base_color.copy())
            
            # Convert to HSV to manipulate lightness more intuitively
            hsv = colorsys.rgb_to_hsv(base_color[0]/255, base_color[1]/255, base_color[2]/255)
            h, s, v = hsv
            
            # Create a slightly lighter shade (increase V, slightly decrease S)
            lighter_v = min(1.0, v * 1.15)  # 15% lighter
            lighter_s = max(0.3, s * 0.9)   # Slightly less saturated
            lighter_rgb = colorsys.hsv_to_rgb(h, lighter_s, lighter_v)
            lighter_rgb = tuple(int(c * 255) for c in lighter_rgb)
            d15_colors.append(np.array(lighter_rgb, dtype=np.uint8))
            
            # Create a slightly darker shade (decrease V, slightly increase S)
            darker_v = max(0.2, v * 0.85)   # 15% darker
            darker_s = min(1.0, s * 1.1)    # Slightly more saturated
            darker_rgb = colorsys.hsv_to_rgb(h, darker_s, darker_v)
            darker_rgb = tuple(int(c * 255) for c in darker_rgb)
            d15_colors.append(np.array(darker_rgb, dtype=np.uint8))
        
        # Take 15 colors total (skip some to get to 15)
        # Keep: all originals (10) + 5 lighter/darker variants
        d15_colors_result = d15_colors[:10]  # All base colors
        d15_colors_result.extend(d15_colors[12:17])  # Add 5 shade variants
        
        return np.array(d15_colors_result[:15], dtype=np.uint8)

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
