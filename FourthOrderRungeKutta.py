# 4thOrderRungeKutta.py

import numpy as np
import matplotlib.pyplot as plt

class FourthOrderRungeKutta:
    def __init__(self, func, a, b, N, x0, enablePlot=True, runName="4rk"):
        """
        Initializes the solver.
        
        Parameters:
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
        enablePlot : bool
            If True, enables plotting after the solution is computed.
        """
        self.func = func  # The function to approximate (f(x, t))
        self.a = a        # Start of the interval
        self.b = b        # End of the interval
        self.N = N        # Number of steps
        self.h = (b - a) / N  # Step size
        self.x0 = x0      # Initial condition
        self.enablePlot = enablePlot # Enable plotting
        self.runName = runName
    
    def solve(self):
        """
        Solves the ODE using the 4th-order Runge-Kutta method (RK4).
        
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
            k1 = self.func(x, t)
            k2 = self.func(x + 0.5 * self.h * k1, t + 0.5 * self.h)
            k3 = self.func(x + 0.5 * self.h * k2, t + 0.5 * self.h)
            k4 = self.func(x + self.h * k3, t + self.h)
            x += (self.h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)  # RK4 update step
        
        return np.array(tpoints), np.array(xpoints)
    
    def plot(self, tpoints, xpoints):
        """
        Plots the results of the ODE approximation.
        
        Parameters:
        tpoints : numpy.ndarray
            Array of time points.
        xpoints : numpy.ndarray
            Array of approximated x values.
        """
        plt.plot(tpoints, xpoints, label=self.runName)
        plt.xlabel("t")
        plt.ylabel("x(t)")
        if self.enablePlot:
            print("Showing 4th Order RK plot...")
            plt.show()
