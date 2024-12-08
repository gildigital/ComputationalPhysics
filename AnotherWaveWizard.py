"""This module provides a class for solving Ted's project 3."""

import numpy as np

from FourthOrderRungeKutta import FourthOrderRungeKutta


class AnotherWaveWizard:
    """
    This class solves Ted's project 3.
    
    <H4>Problems</H4>
    -----------------
    1.  Write functions that represent the rhs of the system of ODEs:
            dy1/dx=y2
            dy2/dx=-k^2*y1
    2.  Implement the 4th-order Runge-Kutta method to solve the system of ODEs.
    """

    def __init__(self, k=1.0):
        """
        Initialize AnotherWaveWizard with a wave number.
        
        Parameters:
        k : |float, optional|
            The wave number, which defines the wavelength.
            Defaults to 1.0.
        """
        self.k = k

    def system_of_ode(self, y, x): # x is added for compatibility with FourthOrderRungeKutta, pylint: disable=unused-argument
        """
        Define the system of ODEs for the wave equation.
        
        <H4>Keyword arguments</H4>
        --------------------------
        y : |np.ndarray|
            Array of dependent variables.
        x : |float|
            Not used.
        
        <H4>Returns</H4>
        ----------------
        |np.ndarray| : Array of derivatives: [dy1_dx, dy2_dx].
        """
        y1, y2 = y
        dy1_dx = y2
        dy2_dx = -self.k**2*y1
        return np.array([dy1_dx, dy2_dx])

    def find_fundamental_k(self, L, tolerance, initial_conditions, n=1, num_steps=100):
        """
        Find the fundamental wave number k for a fixed-free string that satisfies
        the boundary condition y(x=L) ~ 0 within a specified tolerance.
        
        Parameters:
        L : |float|
            Length of the string.
        tolerance : |float|
            Tolerance for boundary condition at x = L.
        initial_conditions : |np.ndarray|
            Initial conditions [y1(0), y2(0)].
        n : |int|
            Mode number for the fundamental. Defaults to 1.
        num_steps : |int, optional|
            Number of steps for the RK4 solver. Defaults to 100.
        
        Returns:
        |float| : The fundamental wave number k that meets the boundary condition.
        """
        # Define initial bracket for k around the expected fundamental mode for a fixed-free string
        k_min = 0.9*(2*n - 1)*np.pi/(2*L)
        k_max = 1.1*(2*n - 1)*np.pi/(2*L)

        # Initialize the RK4 solver
        rk_solver = FourthOrderRungeKutta(
            func=self.system_of_ode,
            a=0,
            b=L,
            N=num_steps,
            x0=initial_conditions,
            enablePlot=False
        )

        # Bisection method to find k that satisfies |y(x=L)| < tolerance
        while k_max - k_min > 1e-6: # Continue until k converges within a small range
            k_mid = (k_min + k_max) / 2.0
            self.k = k_mid # Update k in the system of ODEs
            _, y_vals = rk_solver.solve() # Solve the system with the current k
            
            y_L = y_vals[-1, 0] # Displacement at x = L

            # Check if the boundary condition meets the tolerance
            if abs(y_L) < tolerance:
                return k_mid # Found the wave number that meets tolerance
            elif y_L > 0:
                k_max = k_mid
            else:
                k_min = k_mid

        # Return the midpoint as the best approximation for k
        return (k_min + k_max) / 2.0

# pylint: disable=invalid-name

# Parameters for the wave problem
L = 1 # Length of the string
wavelength = 4.255321
initial_y1 = 0 # y(x=0) = 0
initial_y2 = 1.0 # dy/dx at x=0 = a, let's say a = 1.0
initial_conditions = [initial_y1, initial_y2] # Initial state vector [y1, y2]

k_value = 2*np.pi/wavelength # Example wavelength wavelength=1, so k = 2*pi/wavelength

# Initialize AnotherWaveWizard with k_value
wave_wizard = AnotherWaveWizard(k=k_value)

# Define the interval and step count for RK4
x_start = 0
x_end = L
num_steps = 100

# Initialize FourthOrderRungeKutta with the system_of_ode function from AnotherWaveWizard
rk_solver = FourthOrderRungeKutta(
    func = wave_wizard.system_of_ode, # Pass system_of_ode directly
    a = x_start,
    b = x_end,
    N = num_steps,
    x0 = initial_conditions
)

# Solve the system and plot results
x_vals, y_vals = rk_solver.solve()
rk_solver.plot(x_vals, y_vals)


# Problem 5: Find the fundamental wave number for a fixed-free string
# Parameters for the fundamental mode
L = 1.0 # Length of the string
tolerance = L / 1000 # Boundary condition tolerance at x = L
initial_conditions = [0, 1.0] # Example initial conditions: y(0) = 0, dy/dx(0) = 1

# Initialize AnotherWaveWizard
wave_solver = AnotherWaveWizard()

# Find the fundamental wave number k that satisfies the boundary condition
k_fundamental = wave_solver.find_fundamental_k(L, tolerance, initial_conditions)
lambda_fundamental = 2 * np.pi / k_fundamental

print(f"Fundamental Mode: k = {k_fundamental:.6f}, wavelength = {lambda_fundamental:.6f}")

# Problem 6: Find the next two modes for a fixed-free string

# Initialize the wave solver
wave_solver = AnotherWaveWizard()

# Parameters for the problem
L = 1.0 # Length of the string
tolerance = L / 1000 # Tolerance for the boundary condition
initial_conditions = [0, 1.0] # Initial conditions: y(0) = 0, dy/dx(0) = 1

# Calculate numerical results for higher harmonics
k_harmonics = []
lambda_harmonics = []

for n in [2, 3]: # First two higher harmonics
    # Adjust the initial guess brackets based on theoretical k values
    k_min = 0.9*(2*n - 1)*np.pi/(2*L)
    k_max = 1.1*(2*n - 1)*np.pi/(2*L)

    # Use the bisection method to find k_n
    wave_solver.k = (k_min + k_max)/2
    k_n = wave_solver.find_fundamental_k(L, tolerance, initial_conditions, n)
    lambda_n = 2*np.pi/k_n

    # Store the results
    k_harmonics.append(k_n)
    lambda_harmonics.append(lambda_n)

    print(f"Harmonic {n}: k = {k_n:.6f}, wavelength = {lambda_n:.6f}")

# Theoretical values for comparison
k_theoretical = [(2*n - 1)*np.pi/(2*L) for n in [2, 3]]
lambda_theoretical = [4*L/(2*n - 1) for n in [2, 3]]

for n, k_theory, lambda_theory in zip([2, 3], k_theoretical, lambda_theoretical):
    print(f"Theoretical values for Harmonic {n}: k = {k_theory:.6f}, wavelength = {lambda_theory:.6f}")

# Compute fractional discrepancies with a list comprehension for each pair of wave numbers
# and theoretical values; zip() is used to iterate over both lists simultaneously,
# and we use the formula for fractional discrepancy over each pair produced by zip().
fractional_discrepancies = [
    # Recall: dx/x will give the fractional discrepancy
    abs((k_num - k_theory)/k_theory)
    for k_num, k_theory in zip(k_harmonics, k_theoretical)
]

# Display the fractional discrepancies by zipping the harmonic numbers and the discrepancies
for n, frac_disc in zip([2, 3], fractional_discrepancies):
    print(f"Fractional discrepancy for Harmonic {n}: {frac_disc:.6%}")