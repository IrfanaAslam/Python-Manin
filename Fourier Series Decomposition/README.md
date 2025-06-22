# **Fourier Series Decomposition Visualization**

This repository contains a Python script (`Fourier_Series_Decomposition.py`) that animates the decomposition of a square wave into a sum of its harmonic components. This visualization aims to demonstrate the fundamental concept of Fourier analysis and its power in representing periodic signals.

## **Project Overview**

The core idea of Fourier series is that any periodic function can be represented as an infinite sum of sine and cosine functions. This project visually illustrates this by starting with a simple approximation and progressively adding more harmonic terms to show how the sum converges to the original function.

### **Visualized Concept:**

* **Fourier Series Approximation**: The animation focuses specifically on a **square wave**, showing how its smooth sine wave components are summed together. As more higher-order harmonic terms are included, the approximation gets increasingly accurate, demonstrating the convergence of the Fourier series. The **Gibbs phenomenon** (overshoots at discontinuities) is also naturally visible at the edges of the square wave.

## **Features**

* **Interactive Animation**: A dynamic plot that animates the build-up of the Fourier series approximation.
* **Clear Convergence**: Visually demonstrates how adding more harmonic terms improves the approximation of the original function.
* **Animation Saving**: The script automatically attempts to save the generated animation as a **GIF (`.gif`)** or **MP4 (`.mp4`)** file in the same directory as the script.
* **Configurable Parameters**: Adjust the maximum number of harmonic terms and animation speed directly in the script.

## **How to Run**

To run this visualization, you'll need Python installed on your system along with a few common scientific computing libraries and an external tool for saving animations.

### **Prerequisites**

* Python 3.x
* `matplotlib`
* `numpy`
* **For saving animations (one of these is usually sufficient):**
    * **ImageMagick:** (Recommended for GIF output). Ensure it's installed and in your system's PATH.
    * **FFmpeg:** (Recommended for MP4 output). Ensure it's installed and in your system's PATH.
    * *(If neither is found, `matplotlib` will attempt to use `Pillow` for GIF saving, which is often installed with `matplotlib`.)*

### **Installation**

1.  **Install the required Python libraries:**
    ```bash
    pip install matplotlib numpy
    ```
2.  **Install an animation writer (if you don't have them):**
    * **For ImageMagick (GIFs):** Download from [imagemagick.org](https://imagemagick.org/) (ensure it's added to PATH during Windows install).
    * **For FFmpeg (MP4s):** Download from [ffmpeg.org](https://ffmpeg.org/) (ensure it's added to PATH).
    * *On macOS (with Homebrew):* `brew install imagemagick ffmpeg`
    * *On Linux (Debian/Ubuntu):* `sudo apt-get install imagemagick ffmpeg`

### **Execution**

To run the animation, execute the Python script from your terminal within the script's directory:

```bash
python Fourier_Series_Decomposition.py
A new window will appear displaying the animated Fourier series decomposition. The script will also attempt to save the animation as fourier_square_wave.gif (or .mp4) in the same folder. Check your terminal output for messages regarding saving success or errors.

Project Structure
Fourier_Series_Decomposition.py: The main Python script containing the animation logic and Fourier series calculation for a square wave.
README.md: This file.
fourier_square_wave.gif (or .mp4): The generated animation file (after running the script).
Customization
You can modify the following variables at the top of the Fourier_Series_Decomposition.py script to experiment with the visualization:

MAX_HARMONIC_TERMS: Control the maximum number of odd harmonic terms included in the approximation.
ANIMATION_FRAMES: Set the number of frames for the animation, which affects its speed and smoothness.
PERIOD and L_HALF_PERIOD: Adjust the period of the square wave.
Note: If you wish to visualize a different periodic function, you will need to:

Define a new function similar to square_wave.
Crucially, update the fourier_series_square_wave function (or create a new fourier_series_your_function) to calculate the correct Fourier coefficients (a_0, a_n, b_n) for your new function, as these coefficients are specific to each function.
Contributing
Feel free to fork this repository, open issues, or submit pull requests. Contributions to add new functions, improve the animation, or enhance the code are highly welcome!

License
This project is licensed under the MIT License - see the LICENSE file for details (you might want to add a LICENSE file if you don't have one).

Credits
Developed by Irfana (irfanaaslam69@gmail.com)
Inspired by the elegant mathematics of Fourier analysis.
<!-- end list -->
