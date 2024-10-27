# pylint: disable=invalid-name, redefined-outer-name, trailing-whitespace, line-too-long, ambiguous-variable-name

"""Solver for the quantum harmonic oscillator using the shooting method."""

import numpy as np
import matplotlib.pyplot as plt

from FourthOrderRungeKutta import FourthOrderRungeKutta
from SimpsonsRule import SimpsonsRule


class QuantumHarmonicOscillator:
    """
    QuantumHarmonicOscillator solves the quantum harmonic oscillator using the shooting method
    and plots the energy eigenstates and probability densities.
    """
    def __init__(self, m=1.0, omega=1.0, hbar=1.0):
        """
        Initialize the quantum harmonic oscillator.

        <H4>Keyword arguments</H4>
        --------------------------
            m : |float, optional|
                Defaults to 1.0.
            omega : |float, optional|
                Defaults to 1.0.
            hbar : |float, optional|
                Defaults to 1.0.
        """
        self.m = m
        self.omega = omega
        self.hbar = hbar
        self.current_energy = None
        """Initialize the energy to None."""
        
    def potential(self, x):
        """
        Calculate the potential energy V(x) = (1/2)*m*(omega)^2*x^2
        """
        return 0.5*self.m*self.omega**2*x**2
    
    def exact_energy(self, n):
        """
        Calculate exact energy eigenvalue for nth state
        """
        return (n+0.5) * self.hbar*self.omega
    
    def schrodinger(self, state, x):
        """
        Define the Schrodinger equation for RK4
        
        <H4>Keyword arguments</H4>
        --------------------------
        state : |arraylike|
            Current state of the system [psi, dpsi/dx], where 
            <ol>
                <li>psi: wavefunction</li>
                <li>dpsi/dx: derivative of wavefunction</li>
            </ol>
        x : |float|
            Position value.
        
        <H4>Returns</H4>
        ----------------
        |np.ndarray| Derivatives [dpsi, d2psi] w.r.t. position.
        
        """
        
        # Unpack the state vector, [y_1, y_2] = [psi, dpsi/dx]
        psi, dpsi = state
        
        # Calculate the second derivative of psi
        # d2psi = (2m/hbar^2) * (V(x)-E) * psi
        # Notice, V(x) is a function of x, and E is a constant.
        d2psi = (2.0*self.m/self.hbar**2) * (self.potential(x)-self.current_energy) * psi
        return np.array([dpsi, d2psi])
    
    def solve_schrodinger_equation(self, x_range, E, psi0):
        """
        Solve the Schrodinger equation using RK4
        """
        self.current_energy = E
        
        # Add one more point to account for RK4's behavior
        N = len(x_range)
        x_extended = np.linspace(x_range[0], x_range[-1], N + 1)
        
        rk4_solver = FourthOrderRungeKutta(
            self.schrodinger,
            x_extended[0],
            x_extended[-1],
            N,
            psi0,
            enablePlot=False,
            runName="QHO"
        )
        
        # Ignore the time points, only keep the solution
        _, solution = rk4_solver.solve()
        
        # Interpolate the solution back to the original x_range
        psi = np.interp(x_range, x_extended[:-1], solution[:, 0])
        return psi
    
    def normalize_wavefunction(self, x, psi):
        """
        Normalize the wavefunction using Simpson's 1/3rd Rule
        """
        def psi_squared(x_points): # pylint: disable=unused-argument
            """The square of the wavefunction as part of the integrand for normalization"""
            return psi**2
        
        simpson_solver = SimpsonsRule(
            psi_squared,
            x[0],
            x[-1],
            len(x)-1,
            enablePlot=False
        )
        
        norm = np.sqrt(simpson_solver.solve())
        return psi / norm if norm != 0 else psi
    
    def shoot(self, E, x_range, psi0):
        """
        Shooting method to find eigenvalues
        """
        psi = self.solve_schrodinger_equation(x_range, E, psi0)
        return psi[-1]
    
    def find_eigenstate(self, n, x_range, energy_range=None, num_points=1000):
        """
        Find the nth eigenstate using the shooting method
        """
        if energy_range is None:
            exact_E = self.exact_energy(n)
            energy_range = [0.9 * exact_E, 1.1 * exact_E]
            
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
        if len(zero_crossings) > 0:
            idx = zero_crossings[np.argmin(np.abs(np.array(shooting_results)[zero_crossings]))]
            E_n = energies[idx]
            
            # Use the found energy to solve the Schrodinger equation
            # and get the wavefunction
            psi = self.solve_schrodinger_equation(x_range, E_n, [0.0, 1.0])
            
            # Normalize the wavefunction
            psi = self.normalize_wavefunction(x_range, psi)
            
            return E_n, psi
        else:
            raise ValueError(f"No eigenstate found for n={n} in the given energy range")
    
    def plot_states(self, n_states=4, x_limit=4, num_points=1000):
        """
        Plot multiple eigenstates and the potential
        """
        x = np.linspace(-x_limit, x_limit, num_points)
        V = self.potential(x)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
        
        # Plot potential energy
        ax1.plot(x, V, 'k-', label='V(x)')
        
        # Plot eigenstates and their energy levels
        colors = plt.cm.rainbow(np.linspace(0, 1, n_states))
        for n, color in enumerate(colors):
            try:
                E_n, psi = self.find_eigenstate(n, x)
                
                # Plot wavefunction
                ax1.plot(x, psi + E_n, color=color, 
                        label=f'n={n}, E={E_n:.3f}')
                
                # Plot probability density
                ax2.plot(x, psi**2, color=color,
                        label=f'|$\psi${n}|^2')
                
                # Print comparison with exact energy
                exact_E = self.exact_energy(n)
                print(f"State n={n}:")
                print(f"Calculated E = {E_n:.6f}")
                print(f"Exact E = {exact_E:.6f}")
                print(f"Error = {abs(E_n - exact_E):.6f}")
            
            except ValueError as e:
                print(f"Failed to find state n={n}: {e}")
                continue
        
        ax1.set_ylabel('Energy / Wavefunction')
        ax1.set_title('Quantum Harmonic Oscillator: Wavefunctions and Potential')
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_xlabel('Position (x)')
        ax2.set_ylabel('Probability Density')
        ax2.set_title('Probability Densities')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# Create and plot the quantum harmonic oscillator
qho = QuantumHarmonicOscillator()
qho.plot_states(n_states=4)