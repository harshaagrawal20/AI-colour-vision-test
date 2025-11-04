#!/usr/bin/env python3
"""Quick test of D-15 shading functionality (OpenCV version)"""
import numpy as np
import cv2
from color_extractor import ColorExtractor

# Create a simple test image with 10 distinct colors
img_array = np.zeros((100, 100, 3), dtype=np.uint8)

# Fill with different colored blocks (BGR for OpenCV)
colors_bgr = [
    [0, 0, 255],      # Red
    [0, 255, 0],      # Green
    [255, 0, 0],      # Blue
    [0, 255, 255],    # Yellow
    [255, 0, 255],    # Magenta
    [255, 255, 0],    # Cyan
    [0, 0, 128],      # Dark Red
    [0, 128, 0],      # Dark Green
    [128, 0, 0],      # Dark Blue
    [0, 128, 128],    # Olive
]

patch_size = 20
for i, color in enumerate(colors_bgr):
    x = (i % 5) * patch_size
    y = (i // 5) * patch_size
    img_array[y:y+patch_size, x:x+patch_size] = color

print("Testing D-15 color generation...")

ce = ColorExtractor(n_colors=15)

# Test without shading (should extract 12 base colors)
print("\n1. Without D-15 shading (n_colors=15, use_d15_shading=False):")
try:
    colors_lab_no_shade, _, _ = ce.extract_dominant_colors(
        img_array, convert_to_lab=True, use_d15_shading=False
    )
    print(f"   ✓ Extracted {len(colors_lab_no_shade)} colors")
    print(f"   Color shape: {colors_lab_no_shade.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test with shading (should generate 15 colors with light/dark variants)
print("\n2. With D-15 shading (use_d15_shading=True):")
try:
    colors_lab_shade, _, _ = ce.extract_dominant_colors(
        img_array, convert_to_lab=True, use_d15_shading=True
    )
    print(f"   ✓ Extracted {len(colors_lab_shade)} colors")
    print(f"   Color shape: {colors_lab_shade.shape}")
    print(f"   Expected: 15 colors (10 base + 5 shade variants)")

    # Print luminance values to show variety
    print(f"\n   Luminance (L) values:")
    for i, lab in enumerate(colors_lab_shade):
        l_val = lab[0]
        print(f"      Color {i+1:2d}: L={l_val:6.2f}")

except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n✓ D-15 shading test complete!")
