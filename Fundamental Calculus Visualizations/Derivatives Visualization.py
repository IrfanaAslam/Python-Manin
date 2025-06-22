import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import os # Import the os module for path operations

# --- Configuration ---
FUNCTION = lambda x: x**2   # The function to visualize (currently x^2)
DERIVATIVE_POINT_X = 1.0    # The x-value where we find the derivative
H_START = 2.0               # Initial 'h' value for the secant line (distance from P to Q)
ANIMATION_FRAMES = 150      # Number of frames for the animation
SAVE_ANIMATION = True       # Set to True to save the animation, False to just display
OUTPUT_FILENAME = 'derivative_animation' # Base name for the output file (e.g., .gif or .mp4)

def animate_derivatives():
    """
    Animates the concept of a derivative by showing a secant line
    approaching a tangent line as 'h' (the distance between two points)
    approaches zero.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#f0f0f0') # Light gray background for the figure
    ax.set_facecolor('#ffffff') # White background for the plot area

    # Define x-values for plotting the function
    x_vals = np.linspace(DERIVATIVE_POINT_X - 2.5, DERIVATIVE_POINT_X + 2.5, 400)
    y_vals = FUNCTION(x_vals)

    # Plot the main function
    ax.plot(x_vals, y_vals, label=r'$f(x) = x^2$', color='#3498db', linewidth=2)

    # Mark the fixed point (P) for the derivative
    p_x = DERIVATIVE_POINT_X
    p_y = FUNCTION(p_x)
    ax.plot(p_x, p_y, 'o', color='#e74c3c', markersize=8, label=f'Point P ({p_x}, {p_y:.2f})')

    # Initialize movable point Q and the secant line
    q_x = p_x + H_START
    q_y = FUNCTION(q_x)
    # Initialize point_q with a list for data to avoid RuntimeError later
    point_q, = ax.plot([q_x], [q_y], 'o', color='#2ecc71', markersize=6, alpha=0.9, label='Point Q')
    secant_line, = ax.plot([], [], color='#9b59b6', linestyle='--', linewidth=1.5, label='Secant Line')
    
    # Initialize tangent line (initially hidden)
    tangent_line, = ax.plot([], [], color='#e74c3c', linestyle='-', linewidth=2, alpha=0.0, label='Tangent Line')

    # Initialize text for slope and derivative
    slope_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, color='#9b59b6', verticalalignment='top')
    derivative_text = ax.text(0.5, 0.05, '', transform=ax.transAxes, fontsize=14, color='#e74c3c', horizontalalignment='center')

    # Set plot title, labels, grid, and limits
    ax.set_title(r'Visualization of Derivatives: Slope of Tangent Line for $f(x) = x^2$ at $x=' + str(DERIVATIVE_POINT_X) + '$', fontsize=16)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(DERIVATIVE_POINT_X - 2.5, DERIVATIVE_POINT_X + 2.5)
    ax.set_ylim(0, FUNCTION(DERIVATIVE_POINT_X + 2.5) + 1)
    ax.set_aspect('auto')


    def update(frame):
        """
        Update function for the animation.
        Decreases 'h', moves point Q, updates the secant line and its slope.
        """
        # Calculate 'h' for the current frame, approaching 0 exponentially
        # The exponential decay makes the approach smoother and more visible
        h = H_START * (0.5 ** (frame / (ANIMATION_FRAMES / 5)))

        q_x_current = p_x + h
        q_y_current = FUNCTION(q_x_current)

        # Update point Q's position
        point_q.set_data([q_x_current], [q_y_current])

        # Calculate and update secant line
        if h != 0: # Avoid division by zero when h is exactly 0
            slope_secant = (q_y_current - p_y) / h
        else:
            # If h is effectively zero, use the true derivative for display
            slope_secant = 2 * p_x # True derivative for f(x) = x^2 at x=p_x

        # Calculate points for drawing the secant line extended across the plot
        line_x = np.array([ax.get_xlim()[0], ax.get_xlim()[1]])
        line_y = p_y + slope_secant * (line_x - p_x)
        secant_line.set_data(line_x, line_y)
        slope_text.set_text(f'Secant Slope: {slope_secant:.3f}\nh: {h:.3f}')

        # Reveal tangent line and derivative text towards the end of the animation
        if frame > ANIMATION_FRAMES * 0.8: # Start revealing at 80% of frames
            # Calculate the true tangent line (derivative of x^2 is 2x)
            true_slope = 2 * p_x
            tangent_line.set_alpha(1.0) # Make tangent line fully visible
            tangent_line_y = p_y + true_slope * (line_x - p_x)
            tangent_line.set_data(line_x, tangent_line_y)
            derivative_text.set_text(r'Derivative at $x=' + str(DERIVATIVE_POINT_X) + '$: ' + str(true_slope))
            slope_text.set_alpha(0.3) # Fade secant slope text to highlight tangent

        return point_q, secant_line, tangent_line, slope_text, derivative_text

    # Create the animation object. blit=True can sometimes cause issues with text updates
    # so we'll use blit=False for robustness during saving.
    ani = FuncAnimation(fig, update, frames=ANIMATION_FRAMES, blit=False, repeat=False, interval=50)
    
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
    animate_derivatives()
