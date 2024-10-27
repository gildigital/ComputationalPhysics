# pylint: disable=invalid-name, redefined-outer-name, trailing-whitespace, line-too-long

"""
Sourced from USD Phys 371, Fall 24, ODE.pdf
Euler's method is a 1st order Runge-Kutta method.
"""

import numpy as np
import matplotlib.pyplot as plt

class EulersMethod:
    """
    EulersMethod class for solving ODEs using Euler's method (1st-order Runge-Kutta).
    """
    def __init__(self, func, a, b, N, x0, enablePlot=True):
        """
        Initializes the Euler method solver.
        
        <H4>Keyword arguments</H4>
        --------------------------
        func : callable
            The function to approximate (right-hand side of the ODE).
        a : float
            Start of the interval.
        b : float
            End of the interval.
        N : int
            Number of steps.
        x0 : float
            Initial condition.
        """
        self.func = func  # The function to approximate (f(x, t))
        self.a = a        # Start of the interval
        self.b = b        # End of the interval
        self.N = N        # Number of steps
        self.h = (b - a) / N  # Step size
        self.x0 = x0      # Initial condition
        self.enablePlot = enablePlot # Enable plotting
    
    def solve(self):
        """
        Solves the ODE using Euler's method (1st-order Runge-Kutta method).
        
        Returns:
        tpoints : numpy.ndarray
            Array of time points.
        xpoints : numpy.ndarray
            Array of approximated x values.
        """
        tpoints = np.arange(self.a, self.b, self.h)
        xpoints = []
        x = self.x0
        
        for t in tpoints:
            xpoints.append(x)
            x += self.h * self.func(x, t)  # Euler's update: x(t+h) = x(t) + h*f(x, t)
        
        return np.array(tpoints), np.array(xpoints)
    
    def plot(self, tpoints, xpoints):
        """
        Plots the results of the ODE approximation.
        
        <H4>Keyword arguments</H4>
        --------------------------
        tpoints : numpy.ndarray
            Array of time points.
        xpoints : numpy.ndarray
            Array of approximated x values.
        """
        plt.plot(tpoints, xpoints, label="1rk")
        plt.xlabel("t")
        plt.ylabel("x(t)")
        if self.enablePlot:
            print("Showing Euler's (1rk) plot...")
            plt.show()

