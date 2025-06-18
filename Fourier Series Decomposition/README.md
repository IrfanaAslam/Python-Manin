# **Fourier Series Decomposition Visualization**

This repository contains a Python script (fourier\_series.py) that animates the decomposition of a complex periodic function into a sum of simpler sine and cosine waves (its harmonic components). This visualization aims to demonstrate the fundamental concept of Fourier analysis and its power in representing periodic signals.

## **Project Overview**

The core idea of Fourier series is that any periodic function can be represented as an infinite sum of sine and cosine functions. This project visually illustrates this by starting with a simple approximation and progressively adding more harmonic terms to show how the sum converges to the original function.

### **Visualized Concept:**

* **Fourier Series Approximation**: The animation focuses on a square wave, showing how its smooth sine wave components are summed together. As more higher-order harmonic terms are included, the approximation gets increasingly accurate, demonstrating the convergence of the Fourier series. The Gibbs phenomenon (overshoots at discontinuities) is also naturally visible.

## **Features**

* **Interactive Animation**: A dynamic plot that animates the build-up of the Fourier series.  
* **Clear Convergence**: Visually demonstrates how adding more harmonic terms improves the approximation of the original function.  
* **Customizable Function**: The script is set up to easily change the periodic function being visualized (currently a square wave).  
* **Configurable Parameters**: Adjust the number of harmonic terms and animation speed directly in the script.

## **How to Run**

To run this visualization, you'll need Python installed on your system along with a few common scientific computing libraries.

### **Prerequisites**

* Python 3.x  
* matplotlib  
* numpy

### **Installation**

1. **Install the required Python libraries:**  
   pip install matplotlib numpy

### **Execution**

To run the animation, execute the Python script from your terminal:

python fourier\_series\_Decomposition.py

A new window will appear, displaying the animated Fourier series decomposition.

## **Project Structure**

* fourier\_series.py: The main Python script containing the animation logic and Fourier series calculation for a square wave.  
* README.md: This file.

## **Customization**

You can modify the following variables at the top of the fourier\_series.py script to experiment with the visualization:

* FUNCTION: Change the lambda function to define a different periodic function.  
* PERIOD: Adjust the period of your chosen function.  
* MAX\_HARMONIC\_TERMS: Control the maximum number of harmonic terms included in the approximation.  
* ANIMATION\_FRAMES: Set the number of frames for the animation, which affects its speed and smoothness.

If you change the FUNCTION, you will also need to update the fourier\_series\_square\_wave function to calculate the correct Fourier coefficients (a\_0, a\_n, b\_n) for your new function.

## **Contributing**

Feel free to fork this repository, open issues, or submit pull requests. Contributions to add new functions, improve the animation, or enhance the code are highly welcome\!

## **License**

This project is licensed under the MIT License \- see the LICENSE file for details (you might want to add a LICENSE file if you don't have one).

## **Credits**

* Developed by Irfana ([irfanaaslam69@gmail.com](mailto:irfanaaslam69@gmail.com))  
* Inspired by the elegant mathematics of Fourier analysis.