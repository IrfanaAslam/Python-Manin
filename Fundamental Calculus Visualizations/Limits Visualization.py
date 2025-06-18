import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# --- Configuration ---
FUNCTION = lambda x: x**2  # The function to visualize
LIMIT_POINT_X = 2.0        # The x-value the limit approaches
ANIMATION_FRAMES = 150     # Number of frames for the animation

def animate_limits():
    """
    Animates the concept of a limit by showing points approaching
    a specific x-value from both the left and the right, and how
    the function's y-values converge to the limit.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#f0f0f0') # Light gray background for the figure
    ax.set_facecolor('#ffffff') # White background for the plot area

    x_vals = np.linspace(LIMIT_POINT_X - 2.5, LIMIT_POINT_X + 2.5, 400)
    y_vals = FUNCTION(x_vals)

    # Plot the main function
    ax.plot(x_vals, y_vals, label=r'$f(x) = x^2$', color='#3498db', linewidth=2)

    # Mark the limit point and its function value
    limit_y = FUNCTION(LIMIT_POINT_X)
    ax.plot(LIMIT_POINT_X, limit_y, 'o', color='#e74c3c', markersize=8, label=f'Limit Point ({LIMIT_POINT_X}, {limit_y:.2f})')
    ax.axvline(LIMIT_POINT_X, color='#e74c3c', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(limit_y, color='#e74c3c', linestyle='--', linewidth=0.8, alpha=0.7)

    # Initialize movable points
    # Initialize point_left and point_right with lists for data to avoid RuntimeError later
    point_left, = ax.plot([], [], 'o', color='#2ecc71', markersize=6, alpha=0.9, label='Approaching from Left')
    point_right, = ax.plot([], [], 'o', color='#9b59b6', markersize=6, alpha=0.9, label='Approaching from Right')

    # Initialize text for coordinates
    text_left = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=10, color='#2ecc71', verticalalignment='top')
    text_right = ax.text(0.05, 0.90, '', transform=ax.transAxes, fontsize=10, color='#9b59b6', verticalalignment='top')
    limit_text = ax.text(0.5, 0.05, '', transform=ax.transAxes, fontsize=14, color='#e74c3c', horizontalalignment='center')

    # Initialize lines from points to axes
    line_left_x, = ax.plot([], [], ':', color='#2ecc71', linewidth=0.7, alpha=0.6)
    line_left_y, = ax.plot([], [], ':', color='#2ecc71', linewidth=0.7, alpha=0.6)
    line_right_x, = ax.plot([], [], ':', color='#9b59b6', linewidth=0.7, alpha=0.6)
    line_right_y, = ax.plot([], [], ':', color='#9b59b6', linewidth=0.7, alpha=0.6)


    ax.set_title(r'Visualization of Limits: $\lim_{x \to ' + str(LIMIT_POINT_X) + '} x^2$', fontsize=16)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(LIMIT_POINT_X - 2.5, LIMIT_POINT_X + 2.5)
    ax.set_ylim(0, FUNCTION(LIMIT_POINT_X + 2.5) + 1)
    ax.set_aspect('auto')


    def update(frame):
        """
        Update function for the animation.
        Moves the points and updates text/lines in each frame.
        """
        # Calculate the distance from the limit point for the current frame
        # We use an exponential approach to slow down as we get closer
        dist = 2.0 * (0.5 ** (frame / (ANIMATION_FRAMES / 5)))

        # Points approaching from left
        x_l = LIMIT_POINT_X - dist
        y_l = FUNCTION(x_l)
        point_left.set_data([x_l], [y_l]) # Pass as lists
        line_left_x.set_data([x_l, x_l], [ax.get_ylim()[0], y_l])
        line_left_y.set_data([ax.get_xlim()[0], x_l], [y_l, y_l])
        text_left.set_text(f'Left: x={x_l:.3f}, f(x)={y_l:.3f}')

        # Points approaching from right
        x_r = LIMIT_POINT_X + dist
        y_r = FUNCTION(x_r)
        point_right.set_data([x_r], [y_r]) # Pass as lists
        line_right_x.set_data([x_r, x_r], [ax.get_ylim()[0], y_r])
        line_right_y.set_data([ax.get_xlim()[0], x_r], [y_r, y_r])
        text_right.set_text(f'Right: x={x_r:.3f}, f(x)={y_r:.3f}')

        # Show limit text when animation is near completion
        if frame > ANIMATION_FRAMES * 0.8:
            limit_text.set_text(r'$\lim_{x \to ' + str(LIMIT_POINT_X) + '} x^2 = ' + str(limit_y) + '$')

        return point_left, point_right, text_left, text_right, limit_text, line_left_x, line_left_y, line_right_x, line_right_y

    ani = FuncAnimation(fig, update, frames=ANIMATION_FRAMES, blit=True, repeat=False, interval=50)
    plt.show()

if __name__ == '__main__':
    animate_limits()

