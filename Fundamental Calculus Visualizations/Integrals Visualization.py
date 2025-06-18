import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import quad # For calculating the true integral value

# --- Configuration ---
FUNCTION = lambda x: x**2  # The function to visualize
INTERVAL_START = 0.0       # Start of the integration interval
INTERVAL_END = 2.0         # End of the integration interval
MAX_RECTANGLES = 100       # Maximum number of rectangles for approximation
ANIMATION_FRAMES = 100     # Number of frames for the animation

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

    # Shade the true area
    x_fill = np.linspace(INTERVAL_START, INTERVAL_END, 400)
    y_fill = FUNCTION(x_fill)
    # The `fill_between` returns a PolygonCollection, store it to update its visibility later
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
        num_rectangles = int(1 + (MAX_RECTANGLES - 1) * (frame / ANIMATION_FRAMES))

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

                approx_area += height * dx
            else:
                rects[i].set_alpha(0.0) # Hide unused rectangles

        approx_area_text.set_text(f'Rectangles: {num_rectangles}\nApproximated Area: {approx_area:.4f}')

        # Show true integral text when animation is near completion
        if frame > ANIMATION_FRAMES * 0.8:
            integral_text.set_text(r'True Integral: $\int_{'+str(INTERVAL_START)+'}^{'+str(INTERVAL_END)+'} x^2 dx = ' + f'{true_integral_value:.4f}$')
            # true_area_fill.set_alpha(0.5) # Reveal true area shading more clearly

        # Collect all updated artists to be redrawn
        return rects + [approx_area_text, integral_text]

    ani = FuncAnimation(fig, update, frames=ANIMATION_FRAMES, blit=True, repeat=False, interval=50)
    plt.show()

if __name__ == '__main__':
    animate_integrals()

