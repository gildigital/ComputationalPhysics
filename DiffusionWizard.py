# pylint: disable=invalid-name, redefined-outer-name, trailing-whitespace, line-too-long, ambiguous-variable-name

import numpy as np
import matplotlib.pyplot as plt


class DiffusionWizard:
    r"""
    DiffusionWizard solves the 1D diffusion equation for heat conduction in a rod.
    
    The class implements the Forward Time Centered Space (FTCS) method to solve:
    \frac{\partial T}{\partial t} = D \frac{\partial^2 T}{\partial x^2}
    
    ## Attributes
    --------------------------
    L : |float|
        Length of the rod (meters)
    D : |float| 
        Thermal diffusivity (m^2/s)
    dx : |float|
        Spatial step size (meters)
    dt : |float|
        Time step size (seconds)
    T_hot : |float|
        Temperature at hot end (C)
    T_cold : |float|
        Temperature at cold end (C)
    T_initial : |float|
        Initial temperature (C)
    """

    def __init__(self, L=1.0, D=0.5, dx=0.01, T_hot=50, T_cold=0, T_initial=20):
        """
        Initialize the DiffusionWizard solver.

        ## Keyword arguments  
        --------------------------
        L : |float|
            Length of rod in meters. Defaults to 1.0.
        D : |float|
            Thermal diffusivity in m²/s. Defaults to 0.5.
        dx : |float|
            Spatial step size in meters. Defaults to 0.01.
        T_hot : |float|
            Hot reservoir temperature in C. Defaults to 50.
        T_cold : |float| 
            Cold reservoir temperature in C. Defaults to 0.
        T_initial : |float|
            Initial rod temperature in C. Defaults to 20.
        """
        self.L = L
        self.D = D
        self.dx = dx
        self.T_hot = T_hot 
        self.T_cold = T_cold
        self.T_initial = T_initial
        
        # Derived parameters
        self.nx = int(L/dx) + 1 # Number of spatial points
        self.dt = dx**2/(2*D)*0.50 # Time step for stability
        self.alpha = D*self.dt/dx**2 # Diffusion parameter

        # Check stability condition
        if self.alpha >= 0.5:
            raise ValueError(f'Stability condition not met: alpha = {self.alpha}')

        # Initialize temperature arrays
        self.x = np.linspace(0, L, self.nx)
        self.T = np.full(self.nx, T_initial) # Current temperature
        self.T_new = np.zeros(self.nx) # Next time step

        # Set boundary conditions
        self.T[0] = T_hot
        self.T[-1] = T_cold

    def step(self):
        """
        Perform one time step update using FTCS method.

        ## Returns
        ----------------
        |float| : Maximum temperature change during this step.
        """
        # Apply FTCS update for interior points
        for i in range(1, self.nx-1):
            self.T_new[i] = self.T[i] + self.alpha*(
                self.T[i+1] - 2*self.T[i] + self.T[i-1]
            )

        # Apply boundary conditions
        self.T_new[0] = self.T_hot
        self.T_new[-1] = self.T_cold

        # Calculate maximum change
        max_change = np.max(np.abs(self.T_new - self.T))

        # Update temperature array
        self.T = self.T_new.copy()

        return max_change

    def solve(self, t_final, tolerance=1e-6):
        """
        Solve until final time or steady state is reached.
        
        ## Keyword arguments
        --------------------------
        t_final : |float|
            Final time in seconds
        tolerance : |float|
            Convergence tolerance for steady state
        
        ## Returns
        ----------------
        |list| : Times at which solutions were stored
        |list| : Temperature profiles at those times
        """
        n_steps = int(t_final/self.dt)

        # Lists to store results at intervals
        times = [0]
        profiles = [self.T.copy()]

        # Evolution loop
        for n in range(n_steps):
            max_change = self.step()

            # Store results periodically
            if n % (n_steps//10) == 0:
                times.append((n+1)*self.dt)
                profiles.append(self.T.copy())

            # Check for steady state
            if max_change < tolerance:
                print(f"Steady state reached at t = {(n+1)*self.dt:.3f} s")
                break

        return times, profiles

    def plot_evolution(self, times, profiles):
        """
        Plot temperature profiles at different times.
        
        ## Keyword arguments
        --------------------------
        times : |list|
            List of times
        profiles : |list|
            List of temperature profiles
        """
        plt.figure(figsize=(10, 6))
        
        # Plot analytical steady state
        x = np.linspace(0, self.L, 100)
        T_steady = -50*x + 50
        plt.plot(x, T_steady, 'k--', label='Analytical steady state')
        
        # Plot numerical solutions
        for t, T in zip(times, profiles):
            plt.plot(self.x, T, '-', label=f't = {t:.1f} s')
        
        plt.xlabel('Position (m)')
        plt.ylabel(r'Temperature ($\degree$ C)')
        plt.ylim(0, 60)
        plt.title('Temperature Evolution in $1 m$ Steel Rod')
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Simulation parameters
    L = 1.0 # Rod length (m)
    D = 1.17e-5 # Thermal diffusivity of steel (m^2/s)
    dx = 0.05 # Spatial step (m)
    t_final = 10000.0 # Total simulation time (s)
    
    # Initialize solver
    solver = DiffusionWizard(
        L=L,
        D=D,
        dx=dx,
        T_hot=50, # Hot end (C)
        T_cold=0, # Cold end (C)
        T_initial=20 # Initial temperature (C)
    )
    
    # Solve and get results
    times, profiles = solver.solve(t_final=t_final)
    
    # Plot results
    solver.plot_evolution(times, profiles)
    
    # Plot analytical steady state solution
    x = np.linspace(0, L, 100)
    T_steady = -50*x + 50
    
    # Calculate error compared to steady state
    final_profile = profiles[-1]
    steady_state = -50*solver.x + 50
    max_error = np.max(np.abs(final_profile - steady_state))
    print(rf"Maximum deviation from analytical solution: {max_error:.6f} C")
