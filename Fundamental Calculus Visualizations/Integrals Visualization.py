import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import quad # For calculating the true integral value
import os # Import the os module for path operations

# --- Configuration ---
FUNCTION = lambda x: x**2   # The function to visualize
INTERVAL_START = 0.0        # Start of the integration interval
INTERVAL_END = 2.0          # End of the integration interval
MAX_RECTANGLES = 100        # Maximum number of rectangles for approximation
ANIMATION_FRAMES = 100      # Number of frames for the animation
SAVE_ANIMATION = True       # Set to True to save the animation, False to just display
OUTPUT_FILENAME = 'integral_animation' # Base name for the output file (e.g., .gif or .mp4)

def animate_integrals():
    """
    Animates the concept of a definite integral by showing Riemann sums
    approaching the true area under the curve as the number of rectangles increases.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#f0f0f0') # Light gray background for the figure
    ax.set_facecolor('#ffffff') # White background for the plot area

    x_vals = np.linspace(INTERVAL_START - 0.5, INTERVAL_END + 0.5, 400)
    y_vals = FUNCTION(x_vals)

    # Plot the main function
    ax.plot(x_vals, y_vals, label=r'$f(x) = x^2$', color='#3498db', linewidth=2)

    # Shade the true area (initially transparent or less visible)
    x_fill = np.linspace(INTERVAL_START, INTERVAL_END, 400)
    y_fill = FUNCTION(x_fill)
    true_area_fill = ax.fill_between(x_fill, y_fill, color='#a2d9ce', alpha=0.3, label='True Area (Integral)')

    # Initialize rectangles for Riemann sum
    rects = []
    for _ in range(MAX_RECTANGLES):
        rect = plt.Rectangle((0, 0), 0, 0, fc='#e74c3c', ec='#c0392b', lw=0.8, alpha=0.6)
        ax.add_patch(rect)
        rects.append(rect)

    # Initialize text for approximated area and true integral
    approx_area_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, color='#e74c3c', verticalalignment='top')
    integral_text = ax.text(0.5, 0.05, '', transform=ax.transAxes, fontsize=14, color='#28b463', horizontalalignment='center')

    ax.set_title(r'Visualization of Integrals: Area under $f(x) = x^2$ from $x=' + str(INTERVAL_START) + '$ to $x=' + str(INTERVAL_END) + '$', fontsize=16)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(INTERVAL_START - 0.5, INTERVAL_END + 0.5)
    ax.set_ylim(0, FUNCTION(INTERVAL_END + 0.5) + 1)
    ax.set_aspect('auto')

    # Calculate the true integral using scipy for comparison
    true_integral_value, _ = quad(FUNCTION, INTERVAL_START, INTERVAL_END)


    def update(frame):
        """
        Update function for the animation.
        Increases the number of rectangles and updates the Riemann sum approximation.
        """
        # Determine the number of rectangles for the current frame
        # Ensure at least 1 rectangle, then scale up to MAX_RECTANGLES
        num_rectangles = int(1 + (MAX_RECTANGLES - 1) * (frame / (ANIMATION_FRAMES - 1) if ANIMATION_FRAMES > 1 else 0))
        
        # Calculate width of each rectangle
        dx = (INTERVAL_END - INTERVAL_START) / num_rectangles
        approx_area = 0.0

        # Update rectangles
        for i in range(MAX_RECTANGLES):
            if i < num_rectangles:
                # Right Riemann Sum: height is f(x_i_plus_1)
                x_i = INTERVAL_START + i * dx
                x_i_plus_1 = INTERVAL_START + (i + 1) * dx
                height = FUNCTION(x_i_plus_1) # Height based on right endpoint

                rects[i].set_xy((x_i, 0))
                rects[i].set_width(dx)
                rects[i].set_height(height)
                rects[i].set_alpha(0.6) # Make visible
                rects[i].set_edgecolor('#c0392b') # Ensure edge color is set

                approx_area += height * dx
            else:
                rects[i].set_alpha(0.0) # Hide unused rectangles

        approx_area_text.set_text(f'Rectangles: {num_rectangles}\nApproximated Area: {approx_area:.4f}')

        # Show true integral text when animation is near completion
        if frame > ANIMATION_FRAMES * 0.8: # Start revealing at 80% of frames
            integral_text.set_text(r'True Integral: $\int_{'+str(INTERVAL_START)+'}^{'+str(INTERVAL_END)+'} x^2 dx = ' + f'{true_integral_value:.4f}$')
            # Optionally, make the true area fill more opaque here
            # for collection in true_area_fill.collections:
            #    collection.set_alpha(0.5)

        # Collect all updated artists to be redrawn
        # Using a tuple for blit expects sequence, so combine rects with texts
        return rects + [approx_area_text, integral_text]

    ani = FuncAnimation(fig, update, frames=ANIMATION_FRAMES, blit=True, repeat=False, interval=50)
    
    # --- Save the animation ---
    if SAVE_ANIMATION:
        # Determine the save path relative to where the script is being executed.
        # This ensures the file is saved in the same directory as the script.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try to save as MP4 first as FFmpeg is generally more reliable.
        # Fallback to GIF if MP4 fails, which might use Pillow if ImageMagick is not found.
        mp4_full_path = os.path.join(script_dir, f'{OUTPUT_FILENAME}.mp4')
        gif_full_path = os.path.join(script_dir, f'{OUTPUT_FILENAME}.gif')

        print(f"Attempting to save animation to: {mp4_full_path}")
        try:
            ani.save(mp4_full_path, writer='ffmpeg', fps=30, dpi=150)
            print(f"Animation saved successfully as '{mp4_full_path}'!")
        except Exception as e_mp4:
            print(f"Could not save MP4 (Error: {e_mp4}). Trying GIF instead (requires ImageMagick or Pillow)...")
            try:
                ani.save(gif_full_path, writer='imagemagick', fps=15, dpi=100) # Pillow fallback likely for GIF
                print(f"Animation saved successfully as '{gif_full_path}'!")
            except Exception as e_gif:
                print(f"\n--- ERROR DURING SAVE ---")
                print(f"Could not save GIF either (Error: {e_gif}).")
                print(f"Please ensure FFmpeg (for MP4) or ImageMagick (for GIF) is installed and in your system's PATH.")
                print(f"Showing plot interactively instead.")
                plt.show() # Show plot if saving fails
        else:
            # If MP4 saving was successful, don't show the interactive plot by default
            pass
    else:
        # If SAVE_ANIMATION is False, just show the plot interactively
        plt.show()

if __name__ == '__main__':
    animate_integrals()
