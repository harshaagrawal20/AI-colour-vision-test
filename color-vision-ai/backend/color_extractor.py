"""
Color extraction module: Convert images to LAB space and extract dominant colors.
Optimized for D-15 test with smooth hue progression.
NO BLACK/WHITE/GRAY shades - only colorful vibrant colors.
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

    def extract_colors(self, image):
        """
        Extract 15 COLORFUL dominant colors (NO black/white/gray).
        Returns both LAB and RGB formats.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            colors_lab: Colors in LAB space (15, 3)
            colors_rgb: Colors in RGB space (15, 3) as list
        """
        # Reshape image to list of pixels
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # FILTER OUT BLACK, WHITE, AND GRAY PIXELS
        # Calculate brightness
        brightness = np.mean(pixels, axis=1)
        
        # Calculate saturation (colorfulness)
        rgb_max = np.max(pixels, axis=1)
        rgb_min = np.min(pixels, axis=1)
        saturation = (rgb_max - rgb_min) / (rgb_max + 1e-6)
        
        # Keep only COLORFUL pixels:
        # - Not too dark (brightness > 40)
        # - Not too bright (brightness < 240)
        # - High saturation (> 0.25 = colorful, not gray)
        mask = (brightness > 40) & (brightness < 240) & (saturation > 0.25)
        colorful_pixels = pixels[mask]
        
        if self.verbose:
            print(f"🎨 Filtered {len(colorful_pixels)} colorful pixels from {len(pixels)} total")
        
        # If not enough colorful pixels, lower threshold
        if len(colorful_pixels) < 1000:
            mask = (brightness > 30) & (brightness < 250) & (saturation > 0.15)
            colorful_pixels = pixels[mask]
            if self.verbose:
                print(f"⚠️  Using relaxed filter: {len(colorful_pixels)} pixels")
        
        # Fallback: use all pixels if still not enough
        if len(colorful_pixels) < 100:
            colorful_pixels = pixels
        
        # Extract 5 base hues using K-means
        base_clusters = min(5, len(colorful_pixels) // 100)
        kmeans = KMeans(n_clusters=base_clusters, random_state=42, n_init=10)
        kmeans.fit(colorful_pixels)
        
        # Get base colors (RGB)
        base_colors_rgb = kmeans.cluster_centers_.astype(np.uint8)
        
        # Generate smooth 15-color D-15 progression
        colors_rgb = self._generate_d15_smooth_progression(base_colors_rgb)
        
        # Convert to LAB
        rgb_normalized = colors_rgb.astype(float) / 255.0
        colors_lab = self._rgb_to_lab_batch(rgb_normalized)
        
        return colors_lab, colors_rgb.tolist()

    def extract_dominant_colors(self, image, convert_to_lab=True, use_d15_shading=False):
        """
        Extract dominant colors using K-Means clustering.
        Legacy method for backward compatibility.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            convert_to_lab: If True, return colors in LAB space; else RGB
            use_d15_shading: If True, generate 15 colors optimized for D-15 test
            
        Returns:
            dominant_colors: (n_colors, 3) array in LAB or RGB
            labels: cluster labels for each pixel
            inertia: KMeans inertia
        """
        if use_d15_shading:
            # Use new colorful extraction method
            colors_lab, colors_rgb = self.extract_colors(image)
            if convert_to_lab:
                return colors_lab, None, None
            return np.array(colors_rgb), None, None
        
        # Original method
        pixels = image.reshape(-1, 3).astype(np.float32)
        base_colors = 5 if use_d15_shading else self.n_colors
        
        kmeans = KMeans(n_clusters=base_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        dominant_colors_rgb = kmeans.cluster_centers_.astype(np.uint8)
        
        if use_d15_shading:
            dominant_colors_rgb = self._generate_d15_smooth_progression(dominant_colors_rgb)
        
        if convert_to_lab:
            rgb_normalized = dominant_colors_rgb.astype(float) / 255.0
            lab_colors = self._rgb_to_lab_batch(rgb_normalized)
            return lab_colors, kmeans.labels_, kmeans.inertia_
        
        return dominant_colors_rgb, kmeans.labels_, kmeans.inertia_

    def _generate_d15_smooth_progression(self, base_colors_rgb):
        """
        Generate exactly 15 COLORFUL colors in smooth hue progression for D-15 test.
        NO black, white, or gray shades.
        
        Strategy: Extract base hues, sort by hue angle, create 3 vibrant shades per hue.
        
        Reference Color Selection:
        - Colors sorted by hue angle (0-360° on color wheel)
        - First color (smallest angle, ~0° red region) = FIXED REFERENCE
        - This mimics real D-15 Farnsworth test methodology
        
        Color Wheel Reference:
        - 0°/360° = Red (typically the reference)
        - 30° = Orange
        - 60° = Yellow
        - 120° = Green
        - 180° = Cyan
        - 240° = Blue
        - 300° = Magenta
        
        Args:
            base_colors_rgb: (N, 3) base colors in RGB [0, 255]
            
        Returns:
            d15_colors_rgb: (15, 3) array with 15 vibrant colors in smooth progression
        """
        if self.verbose:
            print(f"🎨 Generating D-15 progression from {len(base_colors_rgb)} base colors")
        
        # Convert base colors to HSV and sort by hue
        hsv_base = []
        for rgb in base_colors_rgb:
            # Ensure RGB values are in proper range
            rgb_norm = np.clip(rgb, 0, 255)
            h, s, v = colorsys.rgb_to_hsv(rgb_norm[0]/255, rgb_norm[1]/255, rgb_norm[2]/255)
            
            # SKIP grayscale colors (low saturation or extreme brightness)
            if s < 0.2 or v < 0.15 or v > 0.95:
                if self.verbose:
                    print(f"⚠️  Skipping grayscale color: RGB{rgb}, S={s:.2f}, V={v:.2f}")
                continue
                
            hsv_base.append((h, s, v, rgb_norm))
        
        # If too few colorful base colors, generate some
        if len(hsv_base) < 3:
            if self.verbose:
                print(f"⚠️  Only {len(hsv_base)} colorful base colors, generating more...")
            hsv_base = self._generate_colorful_base_hues()
        
        # Sort by hue to create natural color wheel progression
        # First hue (smallest angle) will be the REFERENCE color
        hsv_base.sort(key=lambda x: x[0])
        
        if self.verbose:
            hue_degrees = [round(h*360) for h, s, v, _ in hsv_base]
            print(f"📊 Base hues (sorted): {hue_degrees}°")
            print(f"🎯 Reference color will be at ~{hue_degrees[0]}° (first in progression)")
        
        # Generate 15 colors: 3 vibrant shades per hue
        d15_colors = []
        n_base = len(hsv_base)
        shades_per_hue = 3
        
        for i in range(min(5, n_base)):
            h, s, v, original_rgb = hsv_base[i % n_base]
            
            # Create 3 VIBRANT shades (ensure high saturation, no gray)
            # Shade 1: Lighter, vibrant
            h1 = h
            s1 = max(0.5, min(1.0, s * 0.9))  # Keep saturation high
            v1 = min(0.95, v * 1.2)
            
            # Shade 2: Medium, vibrant (original)
            h2 = h
            s2 = max(0.5, min(1.0, s))
            v2 = max(0.3, min(0.9, v))
            
            # Shade 3: Darker, vibrant
            h3 = h
            s3 = max(0.6, min(1.0, s * 1.1))  # Increase saturation for darker shade
            v3 = max(0.25, v * 0.7)
            
            # Convert back to RGB
            for hue, sat, val in [(h1, s1, v1), (h2, s2, v2), (h3, s3, v3)]:
                r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
                rgb_255 = np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)
                
                # Double-check: skip if accidentally grayscale
                if self._is_grayscale(rgb_255):
                    continue
                    
                d15_colors.append(rgb_255)
        
        # Remove near-duplicates
        d15_colors = self._remove_duplicate_colors(d15_colors, threshold=15)
        
        # If not enough colors, generate with more variation
        if len(d15_colors) < 15:
            if self.verbose:
                print(f"⚠️  Only {len(d15_colors)} unique colors, adding more variation...")
            d15_colors = self._generate_with_more_variation(hsv_base, d15_colors)
        
        # Ensure exactly 15 colors
        d15_colors = d15_colors[:15]
        
        if self.verbose:
            print(f"✅ Generated {len(d15_colors)} vibrant colors for D-15 test")
        
        return np.array(d15_colors, dtype=np.uint8)
    
    def _is_grayscale(self, rgb):
        """Check if RGB color is grayscale (low saturation)."""
        rgb_max = np.max(rgb)
        rgb_min = np.min(rgb)
        saturation = (rgb_max - rgb_min) / (rgb_max + 1e-6)
        brightness = np.mean(rgb)
        
        # Grayscale if: low saturation OR extreme brightness
        return saturation < 0.2 or brightness < 35 or brightness > 245
    
    def _generate_colorful_base_hues(self):
        """Generate 5 vibrant base hues across color wheel."""
        base_hues = [0, 0.17, 0.33, 0.5, 0.67]  # Red, Yellow, Green, Cyan, Blue
        hsv_base = []
        
        for h in base_hues:
            s = 0.8  # High saturation
            v = 0.7  # Medium brightness
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            rgb = np.array([int(r*255), int(g*255), int(b*255)], dtype=np.uint8)
            hsv_base.append((h, s, v, rgb))
        
        return hsv_base
    
    def _generate_with_more_variation(self, hsv_base, existing_colors):
        """
        Generate more colorful variations when needed.
        """
        d15_colors = existing_colors.copy()
        
        while len(d15_colors) < 15:
            for i in range(len(hsv_base)):
                if len(d15_colors) >= 15:
                    break
                    
                h, s, v, _ = hsv_base[i]
                
                # Create highly saturated variation
                h_var = (h + np.random.uniform(-0.05, 0.05)) % 1.0
                s_var = max(0.5, min(1.0, s + np.random.uniform(-0.1, 0.2)))
                v_var = max(0.3, min(0.9, v + np.random.uniform(-0.2, 0.2)))
                
                r, g, b = colorsys.hsv_to_rgb(h_var, s_var, v_var)
                rgb_255 = np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)
                
                # Only add if colorful and not duplicate
                if not self._is_grayscale(rgb_255):
                    is_duplicate = False
                    for existing in d15_colors:
                        dist = np.sqrt(np.sum((rgb_255.astype(float) - existing.astype(float)) ** 2))
                        if dist < 15:
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        d15_colors.append(rgb_255)
        
        return d15_colors
    
    def _remove_duplicate_colors(self, colors, threshold=10):
        """
        Remove near-duplicate colors based on RGB distance threshold.
        
        Args:
            colors: List of RGB colors
            threshold: Minimum RGB distance to consider colors as different
            
        Returns:
            List of unique colors
        """
        unique = []
        for color in colors:
            is_duplicate = False
            for existing in unique:
                dist = np.sqrt(np.sum((color.astype(float) - existing.astype(float)) ** 2))
                if dist < threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(color)
        return unique

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