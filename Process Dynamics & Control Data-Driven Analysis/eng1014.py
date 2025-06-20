# eng1014.py
# ENG1014 Numerical Methods Module
# Authors: ENG1014 Team
# Last modified: 08/04/2025

import numpy as np

# Good Programming Practices
def ft2m(ft):
    """
    Converts values in feet to meters.

    Args:
        ft: a value or an array of numbers to convert to meters

    Returns:
        m: f, but converted to meters
    """
    m = ft * 0.3045

    return m

def lb2kg(lb):
    """
    Converts values in pounds to kilograms.

    Args:
        lb: a value or an array of numbers to convert to kilograms

    Returns:
        kg: lb, but converted to kilograms
    """
    kg = lb * 0.454

    return kg

# Fitting Curves to Data
def linreg(x, y):
    """
    Performs linear regression on datasets x and y.

    Args:
        x: An array of numbers. Linear independent dataset. 
        y: An array of numbers. Linear dependent dataset.

    Returns:
        a0: Constant in y = a1*x + a0
        a1: Gradient in y = a1*x + a0
        r2: Coefficient of determination
    """
    
    # Determining best regression coefficients
    n = len(x)
    Sx = sum(x)
    Sy = sum(y)
    Sxx = sum(x**2)
    Sxy = sum(x*y)
    a1 = (n*Sxy - Sx*Sy)/(n*Sxx - Sx**2)
    a0 = np.mean(y) - a1*np.mean(x)

    # Determining r^2 value
    St = sum((y - np.mean(y))**2)
    Sr = sum((y - a0 - a1*x)**2)
    r2 = (St - Sr)/St

    return a0, a1, r2

# Systems of Linear Equations
def naive_gauss(A, b):
    """
    Uses naive Gaussian elimination to solve a system of linear equations represented
    as the matrix equation Ax=b.

    Limitation: will not work when the system has infinite or no solutions.

    Args:
        A: A 2D array containing the coefficients
        b: A 1D array or 2D row/column vector containing the solutions

    Returns:
        x: A 1D array containing the unknowns
    """
    # INPUT VALIDATION
    # check A has 2 dimensions
    ndim_A = np.ndim(A)
    if ndim_A != 2:
        raise ValueError("Error: A must be 2D")

    # check A has the same number of rows and columns
    m_A, n_A = np.shape(A)
    if m_A != n_A:
        raise ValueError("Error: A must be a square matrix")

    # check b has 1 or 2 dimensions
    ndim_b = np.ndim(b)
    if ndim_b != 1 and ndim_b != 2:
        raise ValueError("Error: b must be 1D or a 2D row/column vector")

    # if b has 2 dimensions, b is a row/column vector
    if ndim_b == 2: 
        m_b, n_b = np.shape(b)
        if m_b != 1 and n_b != 1:
            raise ValueError("Error: b must be 1D or a 2D row/column vector")
    
    # check b has the same number of elements as there are rows/columns in A
    p = np.size(b)
    if p != n_A:
        raise ValueError("Error: b must have the same number of elements as there are rows/columns in A")

    # START ALGORITHM
    # reshape b into a px1 2D array
    b_col = np.reshape(b, (p,1))
    # create Aug by concatenating A and b_col and converting the data type to floats
    Aug = np.astype(np.hstack([A,b_col]), float)
    
    # pre-allocate x as a 1D array of 0s with a data type of floats
    x = np.zeros(n_A, dtype = float)

    # FORWARD ELIMINATION
    # loop through columns from the first to the second last
    for c in range(n_A-1):

        # loop through rows from first row below the main diagonal to the last
        for r in range(c+1,n_A):
            # determine normalisation factor
            factor = Aug[r,c]/Aug[c,c]
    
            # replace row r with row r – factor × row c
            Aug[r,:] -= factor*Aug[c,:]

    # BACK SUBSTITUTION
    # solve the last row
    x[n_A-1] = Aug[n_A-1,-1]/Aug[n_A-1,n_A-1]
    
    # loop through rows from second last row to the first
    for r in range(n_A-2,-1,-1):
        # determine value of x_r
        x[r] = (Aug[r,-1] - Aug[r,r+1:n_A] @ x[r+1:n_A]) / Aug[r,r]
    
    return x

def gauss(A, b):
    """
    Uses Gaussian elimination and partial pivoting to solve a system of 
    linear equations represented as the matrix equation Ax=b.

    Limitations: 
    1. will not work when the system has infinite or no solutions.

    Args:
        A: A 2D array containing the coefficients
        b: A 1D array or 2D row/column vector containing the solutions

    Returns:
        x: A 1D array containing the unknowns
    """
    # INPUT VALIDATION
    # check A has 2 dimensions
    ndim_A = np.ndim(A)
    if ndim_A != 2:
        raise ValueError("Error: A must be 2D")

    # check A has the same number of rows and columns
    m_A, n_A = np.shape(A)
    if m_A != n_A:
        raise ValueError("Error: A must be a square matrix")

    # check b has 1 or 2 dimensions
    ndim_b = np.ndim(b)
    if ndim_b != 1 and ndim_b != 2:
        raise ValueError("Error: b must be 1D or a 2D row/column vector")

    # if b has 2 dimensions, b is a row/column vector
    if ndim_b == 2: 
        m_b, n_b = np.shape(b)
        if m_b != 1 and n_b != 1:
            raise ValueError("Error: b must be 1D or a 2D row/column vector")
    
    # check b has the same number of elements as there are rows/columns in A
    p = np.size(b)
    if p != n_A:
        raise ValueError("Error: b must have the same number of elements as there are rows/columns in A")

    # reshape b into a px1 2D array
    b_col = np.reshape(b, (p,1))
    # create Aug by concatenating A and b_col and converting the data type to floats
    Aug = np.astype(np.hstack([A,b_col]), float)
    
    # pre-allocate x as a 1D array of 0s with a data type of floats
    x = np.zeros(n_A, dtype = float)

    # FORWARD ELIMINATION
    # loop through columns from the first to the second last
    for c in range(n_A-1):
        # PARTIAL PIVOTING
        # check if pivot is 0
        if Aug[c,c] == 0:
            # find index of max element below pivot
            index = np.argmax(np.abs(Aug[c+1:n_A,c]))

            # adjust index for A instead of subset
            rowswap = c + index + 1

            # swap rows
            Aug[[c,rowswap],:] = Aug[[rowswap,c],:]

        # loop through rows from first row below the main diagonal to the last
        for r in range(c+1,n_A):
            # determine normalisation factor
            factor = Aug[r,c]/Aug[c,c]
    
            # replace row r with row r – factor × row c
            Aug[r,:] -= factor*Aug[c,:]

    # BACK SUBSTITUTION
    # solve the last row
    x[n_A-1] = Aug[n_A-1,-1]/Aug[n_A-1,n_A-1]
    
    # loop through rows from second last row to the first
    for r in range(n_A-2,-1,-1):
        # determine value of x_r
        x[r] = (Aug[r,-1] - Aug[r,r+1:n_A]@x[r+1:n_A]) / Aug[r,r]
    
    return x
    

# Root Finding
def bisection(f, xl, xu, precision):
    """
    Finds a root of the function f(x) within the interval [xl, xu] using the bisection method.

    Args:
        f: Lambda function to be solved
        xl: Lower limit of the initial guess/bracket
        xu: Upper limit of the initial guess/bracket
        precision: Stopping criteria determined by the user
    
    Returns:
        root: The root of the equation
        iter: The number of iterations taken to find the root
    """
    # Checking if bounds are appropriate
    if f(xl) * f(xu) > 0:
        raise ValueError('Bounds are not appropriate for bisection method')

    # Estimate first iteration of xr and initialize iteration count
    iter = 1
    xr = (xu + xl)/2

    # Check if f(xr) is close enough to zero
    while abs(f(xr)) > precision:
        # Checking subinterval for root
        if f(xl) * f(xr) > 0:
            xl = xr
        else:
            xu = xr
        
        # Recalculate xr and update iteration count
        iter += 1
        xr = (xl + xu) / 2

    # The final xr value is the root
    root = xr

    return root, iter

def falseposition(f, xl, xu, precision):
    """
    Finds a root of the function f(x) within the interval [xl, xu] using the false position method.

    Args:
        f: Lambda function to be solved
        xl: Lower limit of the initial guess/bracket
        xu: Upper limit of the initial guess/bracket
        precision: Stopping criteria determined by the user
    
    Returns:
        root: The root of the equation
        iter: The number of iterations taken to find the root
    """
    # Checking if bounds are appropriate
    if f(xl) * f(xu) > 0:
        raise ValueError('Bounds are not appropriate for false position method')

    # Estimate first iteration of xr and initialize iteration count
    iter = 1
    xr = (f(xu)*xl - f(xl)*xu) / (f(xu) - f(xl))

    # Check if f(xr) is close enough to zero
    while abs(f(xr)) > precision:
        # Checking subinterval for root
        if f(xl) * f(xr) < 0:
            xu = xr
        else:
            xl = xr

        # Recalculate xr and update iteration count
        iter += 1
        xr = (f(xu)*xl - f(xl)*xu)/(f(xu) - f(xl))

    # The final xr value is the root
    root = xr

    return root, iter

def modisecant(f, xi, pert, precision):
    """
    Finds a root of the function f(x) using the modified secant method.

    Args:
        f: Lambda function to be solved
        xi: The initial guess
        pert: A small perturbation fraction
        precision: Stopping criteria determined by the user
    
    Returns:
        root: The root of the equation
        iter: The number of iterations taken to find the root
    """
    # Estimate first iteration of xi1 and initialize iteration count
    iter = 1
    xi1 = xi - pert*f(xi) / (f(xi + pert) - f(xi))

    # Iteration for modified secant method starts
    while abs(f(xi1)) > precision:
        # Increment the iteration count by 1
        iter +=  1
        
        # Updating variables
        xi = xi1
        
        # Recalculating xi1
        xi1 = xi - pert*f(xi) / (f(xi + pert) - f(xi))             

    # The final xi1 value is the root
    root = xi1

    return root, iter

def newraph(f, df, xi, precision):
    """
    Finds a root of the function f(x) using the Newton-Raphson method.

    Args:
        f: Lambda function to be solved
        df: Derivative of the function to be solved
        xi: The initial guess / the next guess x_i_+_1
        precision: Stopping criteria determined by the user
    
    Returns:
        root: The root of the equation
        iter: The number of iterations taken to find the root
    """
    # Estimate first iteration of xi1 and initialize iteration count
    iter = 1
    xi1 = xi - f(xi) / df(xi)

    # Iteration for Newton-Raphson method starts
    while abs(f(xi1)) > precision:
        # Increment the iteration count by 1
        iter += 1

        # Updating variables
        xi = xi1
        
        # Recalculating xi1
        xi1 = xi - f(xi) / df(xi);            

    # The final xi1 value is the root
    root = xi1

    return root, iter

def secant(f, xi, xi_1, precision):
    """
    Finds a root of the function f(x) using the secant method.

    Args:
        f: Lambda function to be solved
        xi: The initial guess
        xi_1: The initial guess x_i_-_1
        precision: Stopping criteria determined by the user
    
    Returns:
        root: The root of the equation
        iter: The number of iterations taken to find the root
    """
    # Estimate first iteration of xi1 and initialize iteration count
    iter = 1
    xi1 = xi - f(xi)*(xi - xi_1) / (f(xi) - f(xi_1))

    # Iteration for secant method starts
    while abs(f(xi1)) > precision:
        # Increment the iteration count by 1
        iter +=  1
        
        # Updating variables
        xi_1 = xi
        xi = xi1
        
        # Recalculating xi1
        xi1 = xi - f(xi)*(xi - xi_1) / (f(xi) - f(xi_1));                      

    # The final xi1 value is the root
    root = xi1

    return root, iter

# Numerical Integration
def comp_trap_vector(x, y):
    """
    Function to perform composite trapezoidal rule.

    Args:
        x: Independent variable
        y: Dependent variable

    Returns:
        I: Integral value
    """
    # Ensure x and y are not empty
    if np.logical_or(len(x) < 2, len(y) < 2):
        raise ValueError("x and y must have at least two points.")
    
    # Ensure x and y have the same length
    if len(x) != len(y):
        raise ValueError("x and y must be of the same length.")
    
    I = 0.0
    # Evaluating integral
    for i in range(1, len(x)):
        I +=  (x[i] - x[i-1]) * (y[i-1] + y[i])/2
    
    return I

def comp_trap(f, a, b, n):
    """
    Function to perform composite trapezoidal rule.

    Args:
        f: Lambda function of equation 
        a: Starting integral limit
        b: Ending integral limit
        n: Number of points 

    Returns:
        I: Integral value
    """
    h = (b - a) / (n - 1)
    x = np.linspace(a, b, n)

    # Evaluating integral
    middleSum = sum(map(f, x[1:-1]))
    I = h/2 * (f(a) + 2*middleSum + f(b))

    return I

def comp_simp13_vector(x,y):
    """
    Function to perform composite Simpson's 1/3 rule.

    Args:
        x: Independent variable
        y: Dependent variable

    Returns:
        I: Integral value
    """
    # Ensure x and y are not empty
    if np.logical_or(len(x) < 2, len(y) < 2):
        raise ValueError("x and y must have at least two points.")
    
    # Ensure x and y have the same length
    if len(x) != len(y):
        raise ValueError("x and y must be of the same length.")

    # Width and number of points
    h = x[1] - x[0]
    n = len(x)

    # Error checking
    if (n < 3) or (np.mod(n,2) == 0):
        raise ValueError('Inappropriate number of points for integration.')
    
    # Evaluating integral
    evenSum = 4 * sum(y[1:-1:2])
    oddSum = 2 * sum(y[2:-1:2])

    I = h/3* (y[0] + evenSum + oddSum + y[-1])

    return I

def comp_simp13(f, a, b, n):
    """
    Function to perform composite Simpson's 1/3 rule.
    
    Args:
        f: Lambda function of equation 
        a: Starting integral limit
        b: Ending integral limit
        n: Number of points 

    Returns:
        I: integral value
    """
    # Error checking
    if (n < 3) or (np.mod(n,2) == 0):
        raise ValueError('Inappropriate number of points for integration.')
    
    # Creating x vector and determining width
    h = (b - a) / (n - 1)
    x = np.linspace(a, b, n)

    # Evaluating integral
    evenSum = 4 * sum(map(f, x[1:-1:2]))
    oddSum = 2 * sum(map(f, x[2:-1:2]))

    I = h/3 * (f(a) + evenSum + oddSum + f(b))

    return I

def simp38_vector(x, y):
    """
    Function to perform a single application of Simpson's 3/8 rule.

    Args:
        x: Independent variable
        y: Dependent variable

    Returns:
        I: Integral value
    """
    # Ensure x and y are not empty
    if np.logical_or(len(x) < 2, len(y) < 2):
        raise ValueError("x and y must have at least two points.")
    
    # Ensure x and y have the same length
    if len(x) != len(y):
        raise ValueError("x and y must be of the same length.")

    # Creating x vector and determining width
    h = x[1] - x[0]

    # Evaluating integral
    I = 3 * h/8 *(y[0] + 3*sum(y[1:3]) + y[-1])

    return I

def simp38(f, a, b):
    """
    Function to perform a single application of Simpson's 3/8 rule.

    Args:
        f: Lambda function of equation 
        a: Starting integral limit
        b: Ending integral limit

    Returns:
        I: Integral value
    """
    # Creating x vector and determining width
    x = np.linspace(a, b, 4)
    h = x[1] - x[0]

    # Evaluating integral
    I = 3 * h/8 * (f(a) + 3*sum(map(f, x[1:3])) + f(b))
    
    return I

# ODEs
def euler(dydt, tSpan, y0, h):
    """
    Function that uses Euler's method to solve an ODE.

    Args:
        dydt: Lambda function of the ODE, f(t,y)
        tSpan: [<initial value>, <final value>] of independent variable
        y0: Initial value of dependent variable
        h: Step size
    
    Returns:
        t: Vector of independent variable
        y: Vector of solution for dependent variable
    """

    # Create t as a vector 
    t = np.arange(tSpan[0], tSpan[1], h)

    # Add an additional t so that the range goes up to tspan[1]
    t = np.append(t, tSpan[1])

    # Preallocate y to improve efficiency
    y = np.ones(len(t)) * y0

    # Implement Euler's method
    for i in range(len(t) - 1):
        if i == len(t) - 2:  # Adjust step size for the last step
            h = t[-1] - t[-2]
        y[i + 1] = y[i] + h * dydt(t[i], y[i])

    return t, y

def heun(dydt, tSpan, y0, h):
    """
    Function that uses Heun's method to solve an ODE.

    Args:
        dydt: Lambda function of the ODE, f(t,y)
        tSpan: [<initial value>, <final value>] of independent variable
        y0: Initial value of dependent variable
        h: Step size
    
    Returns:
        t: Vector of independent variable
        y: Vector of solution for dependent variable
    """

    # Create t as a vector 
    t = np.arange(tSpan[0], tSpan[1], h)

    # Add an additional t so that the range goes up to tspan[1]
    t = np.append(t, tSpan[1])

    # Preallocate y to improve efficiency
    y = np.ones(len(t)) * y0

    # Implement Heun's method
    for i in range(len(t) - 1):
        if i == len(t) - 2:  # Adjust step size for the last step
            h = t[-1] - t[-2]
        yPred = y[i] + h * dydt(t[i], y[i])
        y[i + 1] = y[i] + h * (dydt(t[i], y[i]) + dydt(t[i + 1], yPred)) / 2

    return t, y

def midpoint(dydt, tSpan, y0, h):
    """
    Function that uses the midpoint method to solve an ODE.

    Args:
        dydt: Lambda function of the ODE, f(t,y)
        tSpan: [<initial value>, <final value>] of independent variable
        y0: Initial value of dependent variable
        h: Step size
    
    Returns:
        t: Vector of independent variable
        y: Vector of solution for dependent variable
    """

    # Create t as a vector 
    t = np.arange(tSpan[0], tSpan[1], h)

    # Add an additional t so that the range goes up to tspan[1]
    t = np.append(t, tSpan[1])

    # Preallocate y to improve efficiency
    y = np.ones(len(t)) * y0

    # Implement Midpoint method
    for i in range(len(t) - 1):
        if i == len(t) - 2:  # Adjust step size for the last step
            h = t[-1] - t[-2]
        yHalf = y[i] + (h / 2) * dydt(t[i], y[i])
        tHalf = t[i] + h / 2
        y[i + 1] = y[i] + h * dydt(tHalf, yHalf)

    return t, y