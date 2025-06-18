# **Fundamental Calculus Visualizations**

This project provides animated visualizations of core concepts in fundamental calculus: limits, derivatives (slope of the tangent line), and integrals (area under the curve). Built using Python's matplotlib library, these animations aim to offer a clear and intuitive understanding of these essential mathematical principles.

## **Project Overview**

Understanding calculus often benefits greatly from visual aids. This project brings key ideas to life through dynamic graphs, helping to bridge the gap between abstract mathematical concepts and their geometric interpretations.

### **Visualized Concepts:**

1. **Limits**: Demonstrates how the value of a function approaches a specific point from both the left and the right.  
2. **Derivatives**: Illustrates the concept of the derivative as the slope of a tangent line, showing the secant line approaching the tangent line as the interval shrinks.  
3. **Integrals**: Explains the definite integral as the area under a curve, visualized through Riemann sums (rectangular approximations) becoming increasingly accurate as the number of rectangles increases.

## **Features**

* **Interactive Animations**: Dynamic plots that animate the core calculus concepts.  
* **Clear Labeling**: All plots are well-labeled with function names, axes, and relevant values.  
* **Customizable Functions**: Easily change the function being visualized in each script.  
* **Configurable Animation Speed**: Adjust the number of frames to control animation smoothness and duration.

## **How to Run**

To run these visualizations, you'll need Python installed on your system along with a few common scientific computing libraries.

### **Prerequisites**

* Python 3.x  
* matplotlib  
* numpy  
* scipy (specifically for the integral visualization)

### **Installation** 

1. **Install the required Python libraries:**  
   pip install matplotlib numpy scipy

### **Execution**

Each visualization is contained in its own Python script. You can run them individually from your terminal:

* **Limits Visualization:**  
  python limits.py

* **Derivatives Visualization:**  
  python derivatives.py

* **Integrals Visualization:**  
  python integrals.py

Upon execution, a new window will pop up displaying the animated visualization.

## **Project Structure**

* limits.py: Python script for animating the concept of limits.  
* derivatives.py: Python script for animating the concept of derivatives.  
* integrals.py: Python script for animating the concept of definite integrals.  
* README.md: This file.

## **Customization**

You can easily modify the FUNCTION, LIMIT\_POINT\_X, DERIVATIVE\_POINT\_X, INTERVAL\_START, INTERVAL\_END, and ANIMATION\_FRAMES variables at the top of each Python script to visualize different functions or adjust animation parameters.

## **Contributing**

Feel free to fork this repository, open issues, or submit pull requests. Any contributions to improve the clarity, functionality, or add new visualizations are welcome\!

## **Credits**

* Developed by Irfana Aslam  
* Inspired by the beauty of calculus and the power of matplotlib.