import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# --- Configuration ---
# Define the periodic function to approximate (e.g., a square wave)
# This function defines one period from -L to L
def square_wave(x, L=np.pi):
    """
    Defines a square wave function over one period [-L, L].
    Returns 1 for x in (0, L], -1 for x in [-L, 0].
    """
    return np.where((x > 0) & (x <= L), 1.0, -1.0)

# Period of the function (2L)
PERIOD = 2 * np.pi
L_HALF_PERIOD = np.pi # L = Period / 2

# Number of harmonic terms to add (max_n_terms = how many odd harmonics for square wave)
MAX_HARMONIC_TERMS = 50 # Corresponds to N_max in 2N-1 or 2N+1 series
ANIMATION_FRAMES = MAX_HARMONIC_TERMS # One frame per harmonic added

# --- Fourier Series Calculation Functions ---

def fourier_series_square_wave(x, n_terms, L=L_HALF_PERIOD):
    """
    Calculates the Fourier series approximation for a square wave.
    The square wave is f(x) = 1 for 0 < x <= L, and -1 for -L <= x < 0.
    The series only contains odd sine terms.
    a_0 = 0 (average value over a full period -L to L)
    a_n = 0 (since it's an odd function)
    b_n = (2/L) * integral from 0 to L of f(x)sin(n*pi*x/L) dx
    For square wave: b_n = (2/L) * [ integral from 0 to L of 1*sin(n*pi*x/L) dx ]
                     b_n = (4 / (n * pi)) for odd n, and 0 for even n.
    """
    # Initialize the sum to 0 (a_0 is 0 for this specific square wave centered at 0)
    fs_sum = np.zeros_like(x, dtype=float)

    # We only sum odd terms for a square wave
    # n_terms here means how many *odd* terms to include.
    # So if n_terms = 1, we use n=1. If n_terms = 2, we use n=1, 3.
    for i in range(1, n_terms + 1):
        n = 2 * i - 1 # This generates odd numbers: 1, 3, 5, ...
        # Fourier coefficient b_n for the square wave
        b_n = 4 / (n * np.pi)
        fs_sum += b_n * np.sin(n * np.pi * x / L)
    return fs_sum

# --- Animation Setup ---

def animate_fourier_series():
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#f0f0f0') # Light gray background for the figure
    ax.set_facecolor('#ffffff') # White background for the plot area

    # Define the x-range for plotting over a few periods
    x_plot = np.linspace(-1.5 * PERIOD, 1.5 * PERIOD, 1000)
    
    # Generate original square wave values for plotting
    # We need to handle the periodicity for visualization over multiple periods
    # The square_wave function is defined for one period [-L, L], which is [-pi, pi] here.
    # We can use numpy's remainder operator (%) to map any x to within one period.
    # For a periodic function f(x) with period T, f(x) = f(x % T)
    # However, square_wave is defined for -L to L. So, map x_plot to [-L, L] first.
    
    y_original_mapped = []
    for x_val in x_plot:
        # Map x_val to the range [-L, L]
        # Equivalent to x_val_mod = (x_val + L) % PERIOD - L
        x_val_mod = np.fmod(x_val + L_HALF_PERIOD, PERIOD) - L_HALF_PERIOD
        y_original_mapped.append(square_wave(x_val_mod, L_HALF_PERIOD))
    y_original = np.array(y_original_mapped)


    # Plot the original function
    ax.plot(x_plot, y_original, label='Original Square Wave', color='#2c3e50', linewidth=2.5, linestyle='--')

    # Initialize the Fourier series approximation line
    fourier_line, = ax.plot([], [], label='Fourier Series Approximation', color='#e74c3c', linewidth=2)

    # Initialize text for current number of terms
    terms_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    ax.set_title('Fourier Series Decomposition of a Square Wave', fontsize=18)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlim(x_plot.min(), x_plot.max())
    ax.set_ylim(-1.5, 1.5) # Set fixed y-limits for better visualization
    ax.set_aspect('auto')


    def update(frame):
        """
        Update function for the animation.
        Calculates and plots the Fourier series approximation for the current number of terms.
        """
        # The 'frame' directly corresponds to the number of *odd* terms to include.
        # This means for frame=0, 1 term (n=1); frame=1, 2 terms (n=1, 3), etc.
        num_terms_current = frame + 1

        # Calculate Fourier series approximation for the current number of terms
        y_fourier = fourier_series_square_wave(x_plot, num_terms_current, L_HALF_PERIOD)
        fourier_line.set_data(x_plot, y_fourier)

        # Update the terms text
        terms_text.set_text(f'Harmonic Terms: {num_terms_current} (up to n={2 * num_terms_current - 1})')

        return fourier_line, terms_text

    ani = FuncAnimation(fig, update, frames=ANIMATION_FRAMES, blit=True, repeat=False, interval=100)
    plt.show()

if __name__ == '__main__':
    animate_fourier_series()
