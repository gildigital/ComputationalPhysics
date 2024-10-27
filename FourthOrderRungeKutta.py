# pylint: disable=invalid-name, redefined-outer-name, trailing-whitespace, line-too-long

"""This module provides a class for solving ODEs using the 4th-order Runge-Kutta method."""

import numpy as np
import matplotlib.pyplot as plt


class FourthOrderRungeKutta:
    """
    FourthOrderRungeKutta class for solving ODEs using the 4th-order Runge-Kutta
    """
    def __init__(self, func, a, b, N, x0, enablePlot=True, runName="4rk"):
        """
        Initializes the solver.
        
        <H4>Keyword arguments</H4>
        --------------------------
        func : |callable|
            The function to approximate (right-hand side of the ODE).
        a : |float|
            Start of the interval.
        b : |float|
            End of the interval.
        N : |int|
            Number of steps.
        x0 : |np.ndarray|
            Initial condition.
        enablePlot : |bool|
            If True, enables plotting after the solution is computed.
        """
        self.func = func # The function to approximate (f(x, t))
        self.a = a # Start of the interval
        self.b = b # End of the interval
        self.N = N # Number of steps
        self.h = (b-a) / N # Step size
        self.x0 = np.array(x0) # Initial condition (state vector [x0, v0])
        self.enablePlot = enablePlot # Enable plotting
        self.runName = runName

    def solve(self):
        """
        Solves the ODE using the 4th-order Runge-Kutta method (RK4).
        
        <H4>Returns</H4>
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
            xpoints.append(x.copy())  # Append the current state vector
            k1 = self.h*self.func(x, t)  # Multiply by h to scale the derivative
            k2 = self.h*self.func(x + 0.5*k1, t + 0.5*self.h)
            k3 = self.h*self.func(x + 0.5*k2, t + 0.5*self.h)
            k4 = self.h*self.func(x + k3, t + self.h)
            x += (1.0 / 6.0) * (k1 + 2*k2 + 2*k3 + k4)  # RK4 update step
        
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
        plt.plot(tpoints, xpoints[:, 0], label=f'{self.runName}-Displacement')
        plt.plot(tpoints, xpoints[:, 1], label=f'{self.runName}-Velocity', linestyle='--')
        plt.xlabel("Time $t$")
        plt.ylabel("State Variables")
        plt.legend()
        if self.enablePlot:
            plt.show()
