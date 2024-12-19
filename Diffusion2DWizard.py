# pylint: disable=invalid-name, redefined-outer-name, trailing-whitespace, line-too-long, ambiguous-variable-name

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Diffusion2DWizard:
    """Diffusion2DWizard solves the 2D diffusion equation using FTCS method."""
    
    def __init__(self, D, L=2.0, dx=0.05, A=1.0):
        """
        Initialize the 2D diffusion solver.

        ## Keyword arguments
        --------------------
        D : |float|
            Diffusion coefficient in cm^2/s
        L : |float|
            Domain size in cm. Defaults to 2.0
        dx : |float|
            Spatial step in cm. Defaults to 0.05
        A : |float|
            Initial concentration. Defaults to 1.0
        """
        self.D = D
        self.L = L
        self.dx = dx
        self.A = A

        # Calculate time step for stability (2D needs factor of 4)
        self.dt = (dx**2)/(4*D)*0.9 # 0.9 factor for safety
        self.alpha = D*self.dt/dx**2

        # Check stability condition
        if self.alpha >= 1/4:
            raise ValueError(f'Stability condition not met: alpha = {self.alpha}')

        # Setup grid
        self.nx = int(L/dx)
        self.x = np.linspace(-L/2, L/2, self.nx)
        self.y = np.linspace(-L/2, L/2, self.nx)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Initialize concentration arrays
        self.xi = np.zeros((self.nx, self.nx))
        self.xi_new = np.zeros((self.nx, self.nx))

        # Set initial delta function at center
        center = self.nx//2
        self.xi[center, center] = self.A

    def step(self):
        """
        Perform one time step update using FTCS method.

        ## Returns
        ----------
        |float| : Maximum concentration change
        """
        for i in range(1, self.nx-1):
            for j in range(1, self.nx-1):
                self.xi_new[i,j] = self.xi[i,j] + self.alpha * (
                    self.xi[i+1,j] + self.xi[i-1,j] + 
                    self.xi[i,j+1] + self.xi[i,j-1] - 4*self.xi[i,j]
                )

        # Calculate maximum change
        max_change = np.max(np.abs(self.xi_new - self.xi))

        # Update concentration array
        self.xi = self.xi_new.copy()

        # Enforce boundary conditions
        self.xi[0,:] = self.xi[-1,:] = self.xi[:,0] = self.xi[:,-1] = 0

        return max_change

    def solve(self, t_final, save_interval=10):
        """
        Solve until final time, saving profiles at intervals.

        ## Keyword arguments
        --------------------
        t_final : |float|
            Final time in seconds
        save_interval : |int|
            Number of intermediate profiles to save

        ## Returns
        ----------
        |list| : Times at which profiles were saved
        |list| : Concentration profiles at those times
        """
        n_steps = int(t_final/self.dt)
        save_steps = max(1, n_steps//save_interval)
        
        print(f"n_steps: {n_steps}")
        print(f"dt: {self.dt}")
        print(f"save_steps: {save_steps}")
        
        times = [0]
        profiles = [self.xi.copy()]

        for n in range(n_steps): # pylint: disable=unused-variable
            # Note: Perform the time step update, we don't need to store the return value,
            #       but we do need to call step() to update self.xi
            _ = self.step() # Using _ to indicate we're ignoring the return value
            
            if n % save_steps == 0:
                times.append((n+1)*self.dt)
                profiles.append(self.xi.copy())

        return times, profiles

    def plot_profile(self):
        """Plot current concentration profile."""
        plt.figure(figsize=(8, 6))
        plt.contourf(self.X, self.Y, self.xi, levels=20)
        plt.colorbar(label='Concentration')
        plt.xlabel('x (cm)')
        plt.ylabel('y (cm)')
        plt.title(rf'D = {self.D} $cm^2/s$')
        plt.axis('equal')
        plt.show()
        
    def plot_evolution(self, times, profiles, num_plots=4):
        """
        Plot concentration profiles at different times in a grid.
        
        ## Keyword arguments
        --------------------
        times : |list|
            List of times
        profiles : |list|
            List of concentration profiles
        num_plots : |int|
            Number of profiles to show
        """
        # Select evenly spaced profiles to plot
        indices = np.linspace(0, len(times)-1, num_plots, dtype=int)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.ravel()
        
        for idx, ax in zip(indices, axes):
            im = ax.contourf(self.X, self.Y, profiles[idx], levels=20)
            ax.set_title(f't = {times[idx]:.3f} s')
            ax.set_xlabel('x (cm)')
            ax.set_ylabel('y (cm)')
            ax.set_aspect('equal')
            
            # Add colorbar with same height as plot
            # https://how2matplotlib.com/positioning-the-colorbar-in-matplotlib.html
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            
        plt.suptitle(rf'Diffusion Evolution (D = {self.D} $cm^2/s$)')
        plt.tight_layout()
        plt.show()
        
    def plot_cross_sections(self, times, profiles, num_plots=4):
        """
        Plot concentration cross-sections along x and y axes.
        
        ## Keyword arguments
        --------------------
        times : |list|
            List of times
        profiles : |list|
            List of concentration profiles
        num_plots : |int|
            Number of profiles to show
        """
        indices = np.linspace(0, len(times)-1, num_plots, dtype=int)
        center = self.nx//2
        
        plt.figure(figsize=(12, 5))
        
        # X-axis cross section
        plt.subplot(121)
        for idx in indices:
            plt.plot(self.x, profiles[idx][:,center], 
                    label=f't = {times[idx]:.3f} s')
        plt.xlabel('x (cm)')
        plt.ylabel('Concentration')
        plt.title('X-axis Cross Section')
        plt.legend()
        plt.grid(True)
        
        # Y-axis cross section
        plt.subplot(122)
        for idx in indices:
            plt.plot(self.y, profiles[idx][center,:], 
                    label=f't = {times[idx]:.3f} s')
        plt.xlabel('y (cm)')
        plt.ylabel('Concentration')
        plt.title('Y-axis Cross Section')
        plt.legend()
        plt.grid(True)
        
        plt.suptitle(rf'Concentration Cross-sections (D = {self.D} $cm^2/s$)')
        plt.tight_layout()
        plt.show()
        
    ###########################
    ### Part II Problem 6   ###
    ### compute_averages()  ###
    ### analyze_evolution() ###
    ### plot_averages()     ###
    ###########################
    
    def compute_averages(self, profile):
        """
        Compute spatial averages for a given concentration profile.
        
        ## Keyword arguments                        
        --------------------
        profile : |np.ndarray|
            2D array of concentration values
            
        ## Returns
        ----------
        |tuple| : (avg_x, avg_y, avg_r2)
            Average x position, y position, and r^2
        """
        # Total concentration (for normalization)
        #       \xi_{tot} = \int \int xi(x, y) dx dy
        total_conc = np.sum(profile)*self.dx*self.dx
        
        if total_conc < 1e-10: # Avoid division by zero
            return 0.0, 0.0, 0.0
        
        # Compute average x and y positions
        #   <x> = \int \int x xi(x, y) dx dy / \xi_{tot}
        #   <y> = \int \int y xi(x, y) dx dy / \xi_{tot}
        avg_x = np.sum(self.X*profile)*self.dx*self.dx/total_conc
        avg_y = np.sum(self.Y*profile)*self.dx*self.dx/total_conc
        
        # Calculate r^2 = x^2 + y^2
        #   MSD: <r^2> = \int \int r^2 xi(x, y) dx dy / \xi_{tot}
        r2 = self.X**2 + self.Y**2
        avg_r2 = np.sum(r2*profile)*self.dx*self.dx/total_conc
        
        return avg_x, avg_y, avg_r2

    def analyze_evolution(self, times, profiles): # pylint: disable=unused-argument
        """
        Analyze the time evolution of average quantities.
        
        ## Keyword arguments                                                            
        --------------------
        times : |list|
            List of time points. Note: Not used in this function.
        profiles : |list|
            List of concentration profiles
            
        ## Returns
        ----------
        |tuple| : (avg_x_t, avg_y_t, avg_r2_t)
            Time series of average x, y positions and r^2
        """
        avg_x_t = []
        avg_y_t = []
        avg_r2_t = []
        
        for profile in profiles:
            avg_x, avg_y, avg_r2 = self.compute_averages(profile)
            avg_x_t.append(avg_x)
            avg_y_t.append(avg_y)
            avg_r2_t.append(avg_r2)
        
        return np.array(avg_x_t), np.array(avg_y_t), np.array(avg_r2_t)

    def plot_averages(self, times, profiles):
        """
        Plot the time evolution of average position and MSD.
        
        ## Keyword arguments
        --------------------
        times : |list|
            List of time points
        profiles : |list|
            List of concentration profiles
        """
        # Analyze evolution computes average x, y positions and r^2
        avg_x_t, avg_y_t, avg_r2_t = self.analyze_evolution(times, profiles)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot average position components
        ax1.plot(times, avg_x_t, 'b-', label='⟨x⟩')
        ax1.plot(times, avg_y_t, 'r--', label='⟨y⟩')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (cm)')
        ax1.set_title('Average Position vs Time')
        ax1.grid(True)
        ax1.legend()
        
        # Plot MSD
        ax2.plot(times, avg_r2_t, 'g-')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel(r'$\langle r^2 \rangle \ (cm^2)$')
        ax2.set_title('Mean Square Displacement vs Time')
        ax2.grid(True)
        
        plt.suptitle(rf'$D = {self.D}\ cm^2/s$')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Case 1: Oxygen in air
    D_air = 0.18 # cm^2/s
    solver_air = Diffusion2DWizard(D=D_air)
    print("Solving for oxygen in air...")
    times_air, profiles_air = solver_air.solve(t_final=2.0)
    solver_air.plot_evolution(times_air, profiles_air)
    solver_air.plot_cross_sections(times_air, profiles_air)
    
    # Case 2: Oxygen in water
    D_water = 2e-5 # cm^2/s
    solver_water = Diffusion2DWizard(D=D_water)
    print("Solving for oxygen in water...")
    times_water, profiles_water = solver_water.solve(t_final=1000.0, save_interval=20)
    solver_water.plot_evolution(times_water, profiles_water)
    solver_water.plot_cross_sections(times_water, profiles_water)
    
    # Part II Problem 6
    # Case 1: Oxygen in air
    print("Plotting averages for oxygen in air...")
    solver_air.plot_averages(times_air, profiles_air)

    # Case 2: Oxygen in water
    print("Plotting averages for oxygen in water...")
    solver_water.plot_averages(times_water, profiles_water)
