"""
D-15 Color Vision Test Result Graph Generator
Creates circular diagram showing user's color arrangement pattern
Also generates hue spectrum error visualization
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import io
from typing import List, Tuple
import colorsys


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


    def generate_hue_error_visualization(
        self,
        user_order: List[int],
        all_colors_rgb: List[List[int]],
        title: str = "Color Hue Arrangement Analysis"
    ) -> bytes:
        """
        Generate a linear hue spectrum showing where colors should be vs where user placed them.
        
        Args:
            user_order: List of color indices showing which color was placed in each position
            all_colors_rgb: List of ALL colors in their original order
            title: Graph title
            
        Returns:
            PNG image as bytes
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Calculate hue for each color
        def rgb_to_hue(rgb):
            r, g, b = [x / 255.0 for x in rgb]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            return h * 360  # Convert to degrees
        
        # Get hues for colors 1-15 (skip reference at index 0)
        color_hues = []
        for i in range(1, min(16, len(all_colors_rgb))):
            hue = rgb_to_hue(all_colors_rgb[i])
            color_hues.append((i, hue, all_colors_rgb[i]))
        
        # Sort by hue for reference (correct order)
        sorted_by_hue = sorted(color_hues, key=lambda x: x[1])
        
        # --- TOP PANEL: CORRECT HUE ORDER (REFERENCE) ---
        ax1.set_xlim(0, 360)
        ax1.set_ylim(0, 2)
        ax1.set_title("✓ Correct Arrangement (by Hue Order)", fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel("Hue Angle (degrees)", fontsize=11)
        ax1.set_yticks([])
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Draw hue spectrum background (vibrant rainbow)
        for hue in range(0, 360, 5):
            r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 1.0, 1.0)
            ax1.axvspan(hue, hue + 5, facecolor=(r, g, b), alpha=0.95, zorder=0)
        
        # Plot colors in correct hue order
        for idx, (color_idx, hue, rgb) in enumerate(sorted_by_hue):
            color_normalized = np.array(rgb) / 255.0
            ax1.scatter(hue, 1, s=400, c=[color_normalized], edgecolors='black', linewidths=2, zorder=10)
            ax1.text(hue, 1.5, str(color_idx), ha='center', va='center', 
                    fontsize=9, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # --- BOTTOM PANEL: USER'S ARRANGEMENT ---
        ax2.set_xlim(0, 360)
        ax2.set_ylim(0, 2)
        ax2.set_title("❌ Your Arrangement (errors marked)", fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel("Hue Angle (degrees)", fontsize=11)
        ax2.set_yticks([])
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Draw hue spectrum background (vibrant rainbow)
        for hue in range(0, 360, 5):
            r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 1.0, 1.0)
            ax2.axvspan(hue, hue + 5, facecolor=(r, g, b), alpha=0.95, zorder=0)
        
        # Plot colors in user's order and mark errors
        correct_order = [x[0] for x in sorted_by_hue]  # Color indices in hue order
        errors_found = 0
        
        for position_num in range(1, len(user_order) + 1):
            color_idx = user_order[position_num - 1]
            
            # Find hue of this color
            color_info = next((x for x in color_hues if x[0] == color_idx), None)
            if not color_info:
                continue
                
            _, hue, rgb = color_info
            color_normalized = np.array(rgb) / 255.0
            
            # Check if this is correct position (comparing to sorted hue order)
            is_correct = (position_num - 1 < len(correct_order) and correct_order[position_num - 1] == color_idx)
            
            if is_correct:
                # Correct placement - green border
                ax2.scatter(hue, 1, s=400, c=[color_normalized], edgecolors='green', linewidths=3, zorder=10)
            else:
                # Wrong placement - red border
                ax2.scatter(hue, 1, s=400, c=[color_normalized], edgecolors='red', linewidths=3, zorder=10)
                errors_found += 1
                
                # Draw arrow showing where it should be
                correct_pos = correct_order.index(color_idx) if color_idx in correct_order else -1
                if correct_pos >= 0 and correct_pos < len(sorted_by_hue):
                    correct_hue = sorted_by_hue[correct_pos][1]
                    ax2.annotate('', xy=(correct_hue, 0.5), xytext=(hue, 0.5),
                               arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.6))
            
            # Label with color number
            ax2.text(hue, 1.5, str(color_idx), ha='center', va='center',
                    fontsize=9, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add summary text
        total_colors = len(user_order)
        correct_count = total_colors - errors_found
        accuracy = (correct_count / total_colors * 100) if total_colors > 0 else 0
        
        fig.text(0.5, 0.02, 
                f"Summary: {correct_count}/{total_colors} colors placed correctly ({accuracy:.1f}% accuracy) | "
                f"{errors_found} errors (red arrows show correct positions)",
                ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='green', linewidth=3, label='Correctly placed'),
            Patch(facecolor='white', edgecolor='red', linewidth=3, label='Incorrectly placed')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
        
        # Save to bytes
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        
        return buf.getvalue()


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
