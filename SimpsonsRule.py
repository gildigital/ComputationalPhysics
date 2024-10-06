# Approximate the area of functions using Simpson's 1/3rd rule.

import numpy as np
import matplotlib.pyplot as plt

class SimpsonsRule:
    def __init__(self, func, a, b, N, enablePlot=False):
        """
        Initializes the Simpson's Rule solver.
        
        Parameters:
        func : callable
            The function to integrate.
        a : float
            The lower bound of the integration interval.
        b : float
            The upper bound of the integration interval.
        N : int
            The number of steps (must be even for Simpson's Rule).
        enablePlot : bool
            If True, enable plotting of the function and the approximate area.
        """
        self.func = func
        self.a = a
        self.b = b
        self.N = N if N % 2 == 0 else N + 1  # Ensure N is even
        self.h = (b - a) / self.N  # Step size
        self.enablePlot = enablePlot
        self.validateInput()
    
    def validateInput(self):
        """
        Ensures that N is even. Adjusts N if necessary.
        """
        if self.N % 2 != 0:
            print(f"Warning: N was odd. Adjusted to {self.N + 1} for Simpson's Rule.")
            self.N += 1  # Adjust to an even number if needed
    
    def solve(self):
        """
        Applies Simpson's 1/3rd Rule to compute the numerical integral.
        
        Returns:
        result : float
            The approximated integral value.
        """
        h = self.h
        xpoints = np.linspace(self.a, self.b, self.N + 1)  # Evenly spaced points in the interval
        fpoints = self.func(xpoints)  # Function values at the xpoints
        
        # Simpson's Rule formula
        result = h / 3 * (fpoints[0] + 4 * np.sum(fpoints[1:-1:2]) + 2 * np.sum(fpoints[2:-2:2]) + fpoints[-1])
        return result
    
    def plot(self):
        """
        (Optional) Plots the function and the approximated area.
        """
        if not self.enablePlot:
            return
        
        xpoints = np.linspace(self.a, self.b, self.N + 1)
        fpoints = self.func(xpoints)
        
        plt.plot(xpoints, fpoints, label="Function f(x)")
        plt.fill_between(xpoints, fpoints, alpha=0.3, label="Approximated Area")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Simpson's Rule Approximation")
        plt.legend()
        plt.show()