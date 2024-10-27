# pylint: disable=invalid-name, redefined-outer-name, trailing-whitespace, line-too-long

"""Approximate the area of functions using Simpson's 1/3rd rule."""

import numpy as np

from SimpsonsRule import SimpsonsRule


def f(x):
    """Define the function to integrate."""
    return np.sin(x)

# Set up the bounds and the number of steps
a = 0  # Start of the interval
b = np.pi  # End of the interval (one full sine wave)
N = 100  # Number of subdivisions

# Create an instance of the SimpsonsRule class
simpsonsAreaSolver = SimpsonsRule(f, a, b, N, enablePlot=True)

# Solve the integral
result = simpsonsAreaSolver.solve()
print(f"The approximate integral of sin(x) from {a} to {b} is: {result}")

# Plot the result
simpsonsAreaSolver.plot()
