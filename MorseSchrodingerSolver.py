# pylint: disable=invalid-name, redefined-outer-name, trailing-whitespace, line-too-long, ambiguous-variable-name

"""Solver for the Schrodinger equation with the Morse potential using 4th-order Runge-Kutta."""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from FourthOrderRungeKutta import FourthOrderRungeKutta
from SimpsonsRule import SimpsonsRule


class MorseSchrodingerSolver:
    """
    MorseSchrodingerSolver solves the time-independent Schrodinger equation for
    the Morse potential using the 4th-order Runge-Kutta method.
    """
    def __init__(self, D, a, x0, mu, hbar=1.0):
        """
        Initialize the solver.

        ## Keyword arguments
        --------------------------
        D : |float|
            Dissociation energy.
        a : |float|
            Width parameter.
        x0 : |float|
            Equilibrium bond length.
        mu : |float|
            Reduced mass of the particle.
        hbar : |float, optional|
            Planck's reduced constant. Defaults to 1.0.
        """
        self.D = D # Dissociation energy
        self.a = a # Width parameter
        self.x0 = x0 # Equilibrium bond length
        self.mu = mu # Reduced mass
        self.hbar = hbar
        self.current_energy = None

    def morse_potential(self, x):
        """
        Compute the Morse potential V(x).
        
        ## Keyword arguments
        --------------------------
        x : |float|
            Position.
        
        ## Returns
        ----------------
        |float| : Value of the Morse potential at x.
        """
        return self.D*(np.exp(-2*self.a*(x-self.x0)) - 2*np.exp(-self.a*(x-self.x0)))

    def schrodinger(self, state, x):
        """
        Define the Schrodinger equation as a system of first-order ODEs.
        
        ## Keyword arguments
        --------------------------
        state : |arraylike|
            Current state of the system [psi, dpsi/dx].
        x : |float|
            Position value.
        
        ## Returns
        ----------------
        |np.ndarray| : Derivatives [dpsi, d2psi] w.r.t. position.
        """
        psi, dpsi = state
        V_x = self.morse_potential(x)
        d2psi = (2.0*self.mu/self.hbar**2)*(V_x-self.current_energy)*psi
        return np.array([dpsi, d2psi])

    def solve_schrodinger_equation(self, x_range, E, psi0):
        """
        Solve the Schrodinger equation using RK4.
        
        ## Keyword arguments
        --------------------------
        x_range : |arraylike|
            Array of position values for solving.
        E : |float|
            Energy of the system.
        psi0 : |list|
            Initial values [psi(x_start), dpsi/dx(x_start)].
        
        ## Returns
        ----------------
        |np.ndarray| : Wavefunction values.
        """
        self.current_energy = E
        rk4_solver = FourthOrderRungeKutta(
            self.schrodinger,
            x_range[0],
            x_range[-1],
            len(x_range),
            psi0,
            enablePlot=False
        )
        _, solution = rk4_solver.solve()
        return solution[:, 0] # Return the wavefunction

    def normalize_wavefunction(self, x, psi):
        """
        Normalize the wavefunction using Simpson's 1/3 Rule.
        
        ## Keyword arguments
        --------------------------
        x : |arraylike|
            Position array.
        psi : |arraylike|
            Wavefunction values.
        
        ## Returns
        ----------------
        |np.ndarray| : Normalized wavefunction.
        """
        def psi_squared(xpoints): # pylint: disable=unused-argument
            """
            ## Keyword arguments
            --------------------------
            xpoints : |arraylike|
                Placeholder to match the Simpson's Rule function signature.    

            ## Returns
            ----------------
            |np.ndarray| : The square of the wavefunction as part of the integrand for normalization
            """
            
            return psi**2

        # Prevent overflow by clipping extreme values
        psi = np.clip(psi, -1e5, 1e5)
        
        simpson_solver = SimpsonsRule(
            psi_squared,
            x[0],
            x[-1],
            len(x)-1,
            enablePlot=False
        )
        norm = np.sqrt(simpson_solver.solve())
        
        return psi/norm if norm != 0 else psi
    
    def shoot(self, E, x_range, psi0):
        """
        Shooting method to find eigenvalues
        """
        psi = self.solve_schrodinger_equation(x_range, E, psi0)
        return psi[-1]
    
    def find_eigenstate(self, n, x_range, energy_range=None, num_points=5000):
        """
        Find the nth eigenstate using the shooting method
        """
        if energy_range is None:
            energy_range = [-self.D, 0]  # Default search range
            
        energies = np.linspace(energy_range[0], energy_range[1], num_points)
        shooting_results = [self.shoot(E, x_range, [0.0, 1.0]) for E in energies]
        
        # We want to find the eigenvalue E_n such that the wavefunction
        # psi(x) approaches zero as x approaches infinity. So we look for
        # the zero crossing closest to zero in the shooting results.
        # Zero crossings indicate that the wavefunction is diverging and is
        # indicated by a change in sign of the wavefunction. 
        # The np.signbit function returns:
        #     True for negative numbers (1), 
        #     False for positive numbers (0). 
        # The np.diff function calculates the difference between consecutive elements in the array. 
        # A change in sign will result in a True value (1) in the diff array. 
        # The np.where function returns the indices of the True values and we select the first element [0].
        zero_crossings = np.where(np.diff(np.signbit(shooting_results)))[0]
        
        # Find the zero crossing closest to zero
        if len(zero_crossings) > n:
            idx = zero_crossings[n]
            E_n = energies[idx]
            
            # Use the found energy to solve the Schrodinger equation
            # and get the wavefunction
            psi = self.solve_schrodinger_equation(x_range, E_n, [0.0, 1.0])
            
            # Normalize the wavefunction
            psi = self.normalize_wavefunction(x_range, psi)
            
            return E_n, psi
        else:
            raise ValueError(f"No eigenstate found for n={n} in the given energy range")


# Parameters
D = 10.0 # Dissociation energy
a = 1.0 # Width parameter
x0 = 0.0 # Equilibrium bond length
mu = 1.0 # Reduced mass

# Initialize the solver
solver = MorseSchrodingerSolver(D, a, x0, mu)

# Solve the Schrodinger equation
x_range = np.linspace(-1, 1, 2000)
E = -5.0 # Energy
psi0 = [0.0, 1.0] # Initial wavefunction and derivative
psi = solver.solve_schrodinger_equation(x_range, E, psi0)

# Normalize the wavefunction
psi_normalized = solver.normalize_wavefunction(x_range, psi)

# Plot the result
plt.plot(x_range, psi_normalized, label=r"Wavefunction $\psi(x)$")
plt.title("Wavefunction for the Morse Potential")
plt.xlabel(r"$x$")
plt.ylabel(r"$\psi(x)$")
plt.legend()
plt.grid(True)
plt.show()

# Problem 14 and 15

# Parameters for the Morse potential
D = 100  # Dissociation energy
a = 0.7  # Width parameter
x0 = 0.0  # Equilibrium bond length
mu = 1.0  # Reduced mass
hbar = 1.0  # Planck's constant

# Initialize the solver
solver = MorseSchrodingerSolver(D, a, x0, mu, hbar)

# Define the position range and initial conditions
x_range = np.linspace(-5, 5, 1000)
psi0 = [0.0, 1.0]  # Initial wavefunction and derivative
energy_range = [-D, 0]  # Energy range for eigenvalue search

# Compute the first eight eigenstates
eigenvalues = []
wavefunctions = []

# Check if the results file exists
if os.path.exists("morse_eigenstates.pkl"):
    # File exists, prompt the user
    recalculate = input("The results already exist. Recalculate? [y,n]: ").strip().lower()
    
    if recalculate == 'y':
        # Recalculate eigenstates
        eigenvalues = []
        wavefunctions = []
        for n in range(8):
            try:
                E_n, psi_n = solver.find_eigenstate(n, x_range, energy_range, num_points=1000)
                eigenvalues.append(E_n)
                wavefunctions.append(psi_n)
                print(f"Eigenstate {n}: E = {E_n:.4f}")
            except ValueError as e:
                print(f"Error finding eigenstate {n}: {e}")
        
        # Save results
        with open("morse_eigenstates.pkl", "wb") as f:
            pickle.dump({"eigenvalues": eigenvalues, "wavefunctions": wavefunctions}, f)
        print("Eigenvalues and wavefunctions saved successfully!")
    else:
        # Load precomputed eigenstates
        with open("morse_eigenstates.pkl", "rb") as f:
            data = pickle.load(f)
            eigenvalues = data["eigenvalues"]
            wavefunctions = data["wavefunctions"]
        print("Eigenvalues and wavefunctions loaded successfully!")
else:
    # File does not exist, calculate eigenstates
    eigenvalues = []
    wavefunctions = []
    for n in range(8):
        try:
            E_n, psi_n = solver.find_eigenstate(n, x_range, energy_range, num_points=1000)
            eigenvalues.append(E_n)
            wavefunctions.append(psi_n)
            print(f"Eigenstate {n}: E = {E_n:.4f}")
        except ValueError as e:
            print(f"Error finding eigenstate {n}: {e}")
    
    # Save results
    with open("morse_eigenstates.pkl", "wb") as f:
        pickle.dump({"eigenvalues": eigenvalues, "wavefunctions": wavefunctions}, f)
    print("Eigenvalues and wavefunctions saved successfully!")

# Plot the Morse potential and the eigenstates
plt.figure(figsize=(10, 8))
V_x = solver.morse_potential(x_range)

# Plot the Morse potential
plt.plot(x_range, V_x, 'k-', label="Morse Potential $V(x)$")

# Overlay the wavefunctions
for n, (E_n, psi_n) in enumerate(zip(eigenvalues, wavefunctions)):
    plt.plot(x_range, psi_n + E_n, label=f"n={n}, E={E_n:.4f}")

plt.title("Morse Potential and Wavefunctions")
plt.ylim(-100,0)
plt.xlim(-2,2)
plt.xlabel(r"$x$")
plt.ylabel(r"$V(x)$ and $\psi_n(x)$")
plt.legend()
plt.grid(True)
plt.show()
