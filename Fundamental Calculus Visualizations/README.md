# Fundamental Calculus Visualizations

This project provides animated visualizations of core concepts in fundamental calculus: limits, derivatives (slope of the tangent line), and integrals (area under the curve). Built using Python's matplotlib library, these animations aim to offer a clear and intuitive understanding of these essential mathematical principles.

## Project Overview

Understanding calculus often benefits greatly from visual aids. This project brings key ideas to life through dynamic graphs, helping to bridge the gap between abstract mathematical concepts and their geometric interpretations.

### Visualized Concepts:

1. **Limits**: Demonstrates how the value of a function approaches a specific point from both the left and the right.  
2. **Derivatives**: Illustrates the concept of the derivative as the slope of a tangent line, showing the secant line approaching the tangent line as the interval shrinks.  
3. **Integrals**: Explains the definite integral as the area under a curve, visualized through Riemann sums (rectangular approximations) becoming increasingly accurate as the number of rectangles increases.

## Features

* **Interactive Animations**: Dynamic plots that animate the core calculus concepts.  
* **Clear Labeling**: All plots are well-labeled with function names, axes, and relevant values.  
* **Animation Saving**: Each script automatically attempts to save the generated animation as a **GIF (.gif)** or **MP4 (.mp4)** file in the same directory as the script.  
* **Customizable Functions**: Easily change the function being visualized in each script.  
* **Configurable Animation Speed**: Adjust the number of frames to control animation smoothness and duration.

## How to Run

To run these visualizations, you'll need Python installed on your system along with a few common scientific computing libraries and external tools for saving animations.

### Prerequisites

* Python 3.x  
* matplotlib  
* numpy  
* scipy (specifically for the integral visualization)  
* **For saving animations (one of these is usually sufficient):**  
  * **ImageMagick:** (Recommended for GIF output). Ensure it's installed and in your system's PATH.  
  * **FFmpeg:** (Recommended for MP4 output). Ensure it's installed and in your system's PATH.  
  * *(If neither is found, matplotlib will attempt to use Pillow for GIF saving, which is often installed with matplotlib.)*

### Installation

1. **Install the required Python libraries:**  
   pip install matplotlib numpy scipy

2. **Install an animation writer (if you don't have them):**  
   * **For ImageMagick (GIFs):** Download from [imagemagick.org](https://imagemagick.org/) (ensure it's added to PATH during Windows install).  
   * **For FFmpeg (MP4s):** Download from [ffmpeg.org](https://ffmpeg.org/) (ensure it's added to PATH).  
   * *On macOS (with Homebrew):* brew install imagemagick ffmpeg  
   * *On Linux (Debian/Ubuntu):* sudo apt-get install imagemagick ffmpeg

### Execution

Each visualization is contained in its own Python script. You can run them individually from your terminal within their respective directories:

* **Limits Visualization:**  
  python Limits Visualization.py

* **Derivatives Visualization:**  
  python Derivatives Visualization.py

* **Integrals Visualization:**  
  python Integrals Visualization.py

Upon execution, a new window will appear displaying the animated visualization. The script will also attempt to save the animation (e.g., limit\_animation.gif, derivative\_animation.mp4, etc.) in the same folder. Check your terminal output for messages regarding saving success or errors.

## Project Structure

* Limits Visualization.py: Python script for animating the concept of limits.  
* Derivatives Visualization.py: Python script for animating the concept of derivatives.  
* Integrals Visualization.py: Python script for animating the concept of definite integrals.  
* README.md: This file.  
* limit\_animation.gif (or .mp4): The generated animation file for limits.  
* derivative\_animation.gif (or .mp4): The generated animation file for derivatives.  
* integral\_animation.gif (or .mp4): The generated animation file for integrals.

## Customization

You can easily modify the variables at the top of each Python script to visualize different functions or adjust animation parameters:

* **For Limits Visualization.py:** FUNCTION, LIMIT\_POINT\_X, ANIMATION\_FRAMES.  
* **For Derivatives Visualization.py:** FUNCTION, DERIVATIVE\_POINT\_X, H\_START, ANIMATION\_FRAMES.  
* **For Integrals Visualization.py:** FUNCTION, INTERVAL\_START, INTERVAL\_END, MAX\_RECTANGLES, ANIMATION\_FRAMES.

## Contributing

Feel free to fork this repository, open issues, or submit pull requests. Any contributions to improve the clarity, functionality, or add new visualizations are welcome\!

## License

This project is licensed under the MIT License \- see the LICENSE file for details (you might want to add a LICENSE file if you don't have one).

## Credits

* Developed by Irfana Aslam ([irfanaaslam69@gmail.com](mailto:irfanaaslam69@gmail.com))  
* Inspired by the beauty of calculus and the power of matplotlib.