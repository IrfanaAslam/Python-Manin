# **Process Dynamics and Control: Data-Driven Analysis**

This repository contains a Python script (Process\_Dynamics\_and\_Control\_Data\_Driven\_Analysis.py) that addresses various problems related to process dynamics and control using data-driven analysis. The script covers data inspection and cleaning, numerical integration for cost estimation, curve fitting and root finding for titration data, and solving ordinary differential equations (ODEs) for tank draining simulations.

## **Project Overview**

This script serves as a comprehensive solution to an engineering assignment, demonstrating practical applications of numerical methods in analyzing process data. It specifically tackles challenges in a pH neutralization system and a tank draining scenario.

## **Key Sections and Functionality**

The script is structured into four main questions, each addressing a distinct aspect of data analysis and numerical methods:

### **Q1: Data Inspection and Cleaning**

* **Flowrate Correction**: Identifies and corrects negative flowrate readings due to sensor offset.  
* **Data Visualization**: Plots inlet pH, outlet pH, pH difference, and corrected flowrate over time.  
* **System Status Analysis**: Determines total operating and adjusting times of the pH neutralization system.

### **Q2: Integration and Cost Estimation**

* **Chemical Dose Rate**: Calculates the required chemical dosing mass flowrate R(t) based on inlet pH and a target pH.  
* **Total Chemical Mass**: Numerically integrates R(t) to find the total chemical mass added over a period.  
* **Daily Dosing Cost**: Computes the total cost based on a tiered pricing scheme for daily chemical consumption.

### **Q3: Curve Fitting and Root Finding**

* **Piecewise Model**: Fits a piecewise model (linear, logistic, constant) to titration experimental pH data.  
* **pH Prediction**: Provides a function to estimate pH values based on the fitted piecewise model for given base volumes.  
* **Equivalence Point**: Uses numerical methods to find the volume of base required to reach a neutral pH (equivalence point).  
* **Derivative Plot**: Visualizes the numerical derivative of the pH curve to highlight the equivalence point.

### **Q4: ODEs \- Torricelli's Law Tank Draining**

* **Tank Draining Simulation**: Implements Heun's method (a numerical ODE solver) to simulate liquid height in a draining tank over time, based on Torricelli's Law.  
* **Time Step Analysis**: Investigates the impact of different time step sizes on simulation accuracy.  
* **Analytical Comparison**: Compares numerical drain times to the theoretical analytical solution and calculates percentage errors.  
* **Accuracy Optimization**: Iteratively reduces the time step size until a desired percentage error threshold is met.  
* **Error Analysis Plot**: Generates a log-log plot of percentage error versus time step size to demonstrate convergence behavior.

## **How to Run**

To run this Python script, you'll need Python installed on your system along with a few common scientific computing libraries.

### **Prerequisites**

* Python 3.x  
* matplotlib  
* numpy  
* scipy  
* eng1014 module (This module is explicitly imported in the script. Ensure comp\_trap\_vector and heun functions from eng1014 are available in your Python environment or in the same directory as the script. If eng1014 is a custom module, it must be provided alongside the script.)

### **Required Data Files**

The script expects the following CSV files to be present in the specified path or the same directory:

* pH\_neutralisation\_Sept.csv (for Q1 and Q2)  
* strong\_acid\_strong\_base.csv (While the data for Q3 is hardcoded in the provided script, the context implies it might originate from this file, so it's good to keep in mind for future externalization.)

**Note**: The current script hardcodes the path for pH\_neutralisation\_Sept.csv as r"C:\\Users\\hp\\OneDrive\\Desktop\\Keerthana\\pH\_neutralisation\_Sept.csv". You might need to update this path to match your local file structure.

### **Installation**

1. **Install the required Python libraries:**  
   pip install matplotlib numpy scipy

   Ensure the eng1014.py file or module is accessible to your Python environment (e.g., in the same directory as the main script).

### **Execution**

To run the entire analysis, simply execute the Python script from your terminal:

python Process\_Dynamics\_and\_Control\_Data\_Driven\_Analysis.py

The script will print various results to the console and generate several matplotlib plots in separate windows.

## **Customization**

You can modify the parameters for each question (e.g., k, h0, h\_min for ODEs; target\_pH for integration) by editing the respective sections within the script.

## **Contributing**

Feel free to fork this repository, open issues, or submit pull requests. Any contributions that enhance the analysis, improve code efficiency, or add new features are welcome\!

## **License**

This project is licensed under the MIT License \- see the LICENSE file for details (you might want to add a LICENSE file if you don't have one).

## **Credits**

* Developed by Irfana ([irfanaaslam69@gmail.com](mailto:irfanaaslam69@gmail.com))  
* Part of a Process Dynamics and Control coursework assignment.