"""
D-15 Color Vision Test Result Graph Generator
Creates circular diagram showing user's color arrangement pattern
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import io
from typing import List, Tuple


class D15GraphGenerator:
    def __init__(self, figsize=(10, 10)):
        """
        Initialize D-15 graph generator.
        
        Args:
            figsize: Figure size (width, height) in inches
        """
        self.figsize = figsize
        self.disc_radius = 0.08
        self.circle_radius = 0.8
        
    def generate_graph(
        self,
        user_order: List[int],
        all_colors_rgb: List[List[int]],
        reference_color_rgb: List[int] = None,
        show_confusion_lines: bool = True,
        title: str = "D-15 Color Vision Test Result"
    ) -> bytes:
        """
        Generate D-15 circular graph showing user's arrangement.
        
        Args:
            user_order: List of color indices showing which color was placed in each position
                       [color_at_pos1, color_at_pos2, ..., color_at_pos15]
                       e.g., [5, 12, 3, ...] means color #5 was placed at position 1
            all_colors_rgb: List of ALL 15 colors in their original order [[r,g,b], ...]
                           Index matches color index (not position)
            reference_color_rgb: RGB color of reference disc (shown at center)
            show_confusion_lines: Whether to show protan/deutan/tritan axes
            title: Graph title
            
        Returns:
            PNG image as bytes
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_aspect('equal')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        
        # Add title
        ax.text(0, 1.15, title, ha='center', va='top', 
                fontsize=16, fontweight='bold')
        
        # Calculate positions for 15 discs around circle
        n_discs = 15
        angles = np.linspace(0, 2 * np.pi, n_discs, endpoint=False)
        angles = angles - np.pi / 2  # Start at top (12 o'clock)
        
        positions = {}
        for i in range(n_discs):
            x = self.circle_radius * np.cos(angles[i])
            y = self.circle_radius * np.sin(angles[i])
            positions[i + 1] = (x, y)  # Positions numbered 1-15
        
        # Draw confusion axes (faint background lines) if requested
        # DISABLED - makes graph too busy
        # if show_confusion_lines:
        #     self._draw_confusion_axes(ax, positions)
        
        # Draw reference color at center (if provided)
        if reference_color_rgb:
            ref_circle = patches.Circle(
                (0, 0), 
                self.disc_radius * 1.5,
                facecolor=np.array(reference_color_rgb) / 255.0,
                edgecolor='black',
                linewidth=2,
                zorder=10
            )
            ax.add_patch(ref_circle)
            ax.text(0, 0, 'REF', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                   zorder=11)
        
        # Draw connecting path based on HUE ORDER, not position order
        # The colors are arranged by hue (0=ref, 1=darkest, 15=lightest)
        # We need to connect them in the order: which position has color 1, which has color 2, etc.
        
        print(f"\n🎨 D-15 Graph Path Generation:")
        print(f"  user_order (color indices at positions 1-15): {user_order}")
        
        # Build a map: color_index -> position_number
        # user_order[position-1] = color_index, so we need to invert this
        color_to_position = {}
        for position_num in range(1, n_discs + 1):
            color_index = user_order[position_num - 1]
            color_to_position[color_index] = position_num
            print(f"  Position {position_num} has color {color_index}")
        
        print(f"\n  Connecting path in HUE order:")
        # Connect colors in hue order: color 1 -> color 2 -> color 3 -> ... -> color 15
        # Skip color 0 (reference) since it's not in the arrangement
        path_points = []
        path_description = []
        for color_idx in range(1, n_discs + 1):  # Colors 1-15 in hue order
            if color_idx in color_to_position:
                position_num = color_to_position[color_idx]
                path_points.append(positions[position_num])
                path_description.append(f"Color{color_idx}@Pos{position_num}")
        
        print(f"  Path: {' → '.join(path_description)}")
        
        # Close the loop back to first color
        if len(path_points) > 0:
            path_points.append(path_points[0])
        
        path_array = np.array(path_points)
        ax.plot(path_array[:, 0], path_array[:, 1],
               color='#000000', linewidth=4, alpha=0.85,  # Dark black, thick, high opacity
               zorder=5, linestyle='-')
        
        # Draw color discs at each position
        for position_num in range(1, n_discs + 1):
            x, y = positions[position_num]
            
            # Get which color the user placed at this position
            # user_order[i] = color index placed at position (i+1)
            color_index = user_order[position_num - 1]  # 0-indexed list
            
            # Get the actual color RGB
            if 0 <= color_index < len(all_colors_rgb):
                color_rgb = np.array(all_colors_rgb[color_index]) / 255.0
            else:
                color_rgb = [0.5, 0.5, 0.5]  # Gray fallback
            # Get the actual color RGB
            if 0 <= color_index < len(all_colors_rgb):
                color_rgb = np.array(all_colors_rgb[color_index]) / 255.0
            else:
                color_rgb = [0.5, 0.5, 0.5]  # Gray fallback
            
            # Draw colored disc
            circle = patches.Circle(
                (x, y),
                self.disc_radius,
                facecolor=color_rgb,
                edgecolor='black',
                linewidth=2,
                zorder=10
            )
            ax.add_patch(circle)
            
            # Add position number label on disc
            ax.text(x, y, str(position_num), ha='center', va='center',
                   fontsize=11, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                   zorder=11)
            
            # Add position label outside circle
            label_x = x * 1.15
            label_y = y * 1.15
            ax.text(label_x, label_y, str(position_num),
                   ha='center', va='center',
                   fontsize=9, color='#333333',
                   zorder=3)
        
        # Add legend
        legend_text = (
            "How to read:\n"
            "• Numbers show arrangement order (1-15)\n"
            "• Line connects colors in user's order\n"
            "• Crossings indicate color confusion"
        )
        ax.text(-1.1, -1.05, legend_text,
               fontsize=9, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor='gray', alpha=0.9))
        
        # Save to bytes
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        
        return buf.getvalue()
    
    def _draw_confusion_axes(self, ax, positions):
        """
        Draw faint confusion axes for protan, deutan, tritan deficiencies.
        These help identify which type of color blindness pattern appears.
        
        Args:
            ax: Matplotlib axis
            positions: Dict of disc positions {disc_num: (x, y)}
        """
        # Protan axis (red-green confusion): connects discs along red-green axis
        # Typically affects discs: 1-4, 10-13
        protan_pairs = [(1, 13), (2, 12), (3, 11), (4, 10)]
        
        # Deutan axis (green-red confusion): similar to protan but different angles
        # Typically affects discs: 2-5, 11-14
        deutan_pairs = [(2, 14), (3, 13), (4, 12), (5, 11)]
        
        # Tritan axis (blue-yellow confusion): connects discs along blue-yellow axis
        # Typically affects discs: 6-9, 13-15, 1-3
        tritan_pairs = [(6, 15), (7, 1), (8, 2), (9, 3)]
        
        # Draw protan lines (red)
        for disc1, disc2 in protan_pairs:
            if disc1 in positions and disc2 in positions:
                x1, y1 = positions[disc1]
                x2, y2 = positions[disc2]
                ax.plot([x1, x2], [y1, y2], 
                       color='red', linewidth=0.5, alpha=0.15,
                       linestyle='--', zorder=1)
        
        # Draw deutan lines (green)
        for disc1, disc2 in deutan_pairs:
            if disc1 in positions and disc2 in positions:
                x1, y1 = positions[disc1]
                x2, y2 = positions[disc2]
                ax.plot([x1, x2], [y1, y2],
                       color='green', linewidth=0.5, alpha=0.15,
                       linestyle='--', zorder=1)
        
        # Draw tritan lines (blue)
        for disc1, disc2 in tritan_pairs:
            if disc1 in positions and disc2 in positions:
                x1, y1 = positions[disc1]
                x2, y2 = positions[disc2]
                ax.plot([x1, x2], [y1, y2],
                       color='blue', linewidth=0.5, alpha=0.15,
                       linestyle='--', zorder=1)
        
        # Add axis labels
        ax.text(1.05, 0.3, 'Protan', fontsize=8, color='red', alpha=0.3)
        ax.text(0.8, 0.6, 'Deutan', fontsize=8, color='green', alpha=0.3)
        ax.text(-0.5, -1.0, 'Tritan', fontsize=8, color='blue', alpha=0.3)


# Example usage
if __name__ == "__main__":
    # Test data
    user_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # Perfect order
    
    # Generate rainbow colors for testing
    colors_rgb = [
        [255, 0, 0],      # 1: Red
        [255, 127, 0],    # 2: Orange
        [255, 255, 0],    # 3: Yellow
        [127, 255, 0],    # 4: Yellow-Green
        [0, 255, 0],      # 5: Green
        [0, 255, 127],    # 6: Green-Cyan
        [0, 255, 255],    # 7: Cyan
        [0, 127, 255],    # 8: Cyan-Blue
        [0, 0, 255],      # 9: Blue
        [127, 0, 255],    # 10: Blue-Purple
        [255, 0, 255],    # 11: Magenta
        [255, 0, 127],    # 12: Magenta-Red
        [200, 100, 100],  # 13: Pink
        [150, 150, 200],  # 14: Lavender
        [200, 150, 100],  # 15: Tan
    ]
    
    reference_color = [128, 0, 0]  # Dark red
    
    generator = D15GraphGenerator()
    
    # Test 1: Perfect arrangement
    img_bytes = generator.generate_graph(
        user_order=user_order,
        colors_rgb=colors_rgb,
        reference_color_rgb=reference_color,
        show_confusion_lines=True,
        title="D-15 Test Result - Perfect Arrangement"
    )
    
    with open("d15_test_perfect.png", "wb") as f:
        f.write(img_bytes)
    print("✓ Generated: d15_test_perfect.png")
    
    # Test 2: Confused arrangement (simulating color blindness)
    confused_order = [1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]
    img_bytes = generator.generate_graph(
        user_order=confused_order,
        colors_rgb=colors_rgb,
        reference_color_rgb=reference_color,
        show_confusion_lines=True,
        title="D-15 Test Result - Color Confusion Pattern"
    )
    
    with open("d15_test_confused.png", "wb") as f:
        f.write(img_bytes)
    print("✓ Generated: d15_test_confused.png")
