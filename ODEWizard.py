from FourthOrderRungeKutta import FourthOrderRungeKutta
from SecondOrderRungeKutta import SecondOrderRungeKutta
from EulersMethod import EulersMethod
import numpy as np
import matplotlib.pyplot as plt

# Define a different function to apparoximate
def f(x, t):
    return -x**3 + np.sin(t)

# Parameters for the solver
a = 0.0     # Start of the interval
b = 10.0    # End of the interval
N = 15      # Number of steps
x0 = 0.0    # Initial condition

# Most precise solution
rk4Approximator = FourthOrderRungeKutta(f, a, b, 10000, x0, False, "Actual")
tpoints, xpoints = rk4Approximator.solve()
rk4Approximator.plot(tpoints, xpoints)

# Create an instance of the FourthOrderRungeKutta class
rk4Approximator = FourthOrderRungeKutta(f, a, b, N, x0, False)
tpoints, xpoints = rk4Approximator.solve()
rk4Approximator.plot(tpoints, xpoints)

# Create an instance of the SecondOrderRungeKutta class
rk2Approximator = SecondOrderRungeKutta(f, a, b, N, x0, False)
tpoints, xpoints = rk2Approximator.solve()
rk2Approximator.plot(tpoints, xpoints)

# Create an instance of the EulerMethod class
rk1Approximator = EulersMethod(f, a, b, N, x0, False)
tpoints, xpoints = rk1Approximator.solve()
rk1Approximator.plot(tpoints, xpoints)

plt.title("Numerical Solutions of $-x^3+sin(t)$")
plt.legend()
plt.show()