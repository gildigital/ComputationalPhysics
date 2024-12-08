# pylint: disable=invalid-name, redefined-outer-name, trailing-whitespace, line-too-long, ambiguous-variable-name

"""SecondOrderRungeKutta solves a first-order ODE using the 2nd-order Runge-Kutta method (midpoint method)."""

import numpy as np
import matplotlib.pyplot as plt

class SecondOrderRungeKutta:
    """SecondOrderRungeKutta class for solving ODEs using the 2nd-order Runge-Kutta method."""
    def __init__(self, func, a, b, N, x0, enablePlot=True):
        """
        Initializes the solver.
        
        ## Keyword arguments
        --------------------------
        func : callable
            The function to approximate (right-hand side of the ODE).
        a : |float|
            Start of the interval.
        b : |float|
            End of the interval.
        N : |int|
            Number of steps.
        x0 : |float|
            Initial condition.
        """
        self.func = func # The function to approximate (f(x, t))
        self.a = a # Start of the interval
        self.b = b # End of the interval
        self.N = N # Number of steps
        self.h = (b - a) / N # Step size
        self.x0 = x0 # Initial condition
        self.enablePlot = enablePlot # Enable plotting
    
    def solve(self):
        """
        Solves the ODE using the 2nd-order Runge-Kutta method (midpoint method).
        
        ## Returns
        ----------------
        tpoints : |np.ndarray|
            Array of time points.
        xpoints : |np.ndarray|
            Array of approximated x values.
        """
        tpoints = np.arange(self.a, self.b, self.h)
        xpoints = []
        x = self.x0
        
        for t in tpoints:
            xpoints.append(x)
            k1 = self.func(x, t) # Slope at the beginning of the interval
            x_mid = x + 0.5 * self.h * k1 # Estimate x at the midpoint
            k2 = self.func(x_mid, t + 0.5 * self.h) # Slope at the midpoint
            x += self.h * k2 # Update x using the midpoint slope
        
        return np.array(tpoints), np.array(xpoints)
    
    def plot(self, tpoints, xpoints):
        """
        Plots the results of the ODE approximation.
        
        ## Keyword arguments
        --------------------------
        tpoints : numpy.ndarray
            Array of time points.
        xpoints : numpy.ndarray
            Array of approximated x values.
        """
        plt.plot(tpoints, xpoints, label="2rk")
        plt.xlabel("t")
        plt.ylabel("x(t)")
        if self.enablePlot:
            print("Showing Euler's 2nd (2rk) plot...")
            plt.show()
