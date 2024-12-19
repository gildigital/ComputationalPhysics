# pylint: disable=invalid-name, redefined-outer-name, trailing-whitespace, line-too-long, ambiguous-variable-name

import numpy as np
import matplotlib.pyplot as plt


class RandomWalkWizard:
    """
    RandomWalkWizard simulates 2D random walks for multiple particles.
    
    Simulates N_p particles taking N_s random walk steps. Each step has:
    - Fixed step size dr
    - Random angle theta
    - Position updates: x += dr*cos(theta), y += dr*sin(theta)
    """
    def __init__(self, N_p=1000, N_s=10000, dr=1.0):
        """
        Initialize the random walk simulation.
        
        ## Keyword arguments
        --------------------------
        N_p : |int|
            Number of particles. Defaults to 1000.
        N_s : |int|
            Number of steps per particle. Defaults to 10000.
        dr : |float|
            Step size. Defaults to 1.0.
        """
        self.N_p = N_p # Number of particles
        self.N_s = N_s # Number of steps
        self.dr = dr # Step size
        
        # Initialize position arrays [particles, steps]
        self.x = np.zeros((N_p, N_s+1)) # +1 to include initial position
        self.y = np.zeros((N_p, N_s+1))
        
        # Array for random angles at each step
        self.theta = np.zeros((N_p, N_s)) # Only N_s since initial position is fixed
        
    def generate_steps(self):
        """
        Generate random angles for all particles and steps.
        Angles are uniform in [0, 2pi].
        """
        self.theta = np.random.uniform(0, 2*np.pi, size=(self.N_p, self.N_s))
        
    def simulate(self):
        """
        Perform the random walk simulation for all particles.
        Updates x and y arrays with particle trajectories.
        """
        # Generate random angles first
        self.generate_steps()
        
        # Perform steps for all particles
        for n in range(self.N_s):
            # Update positions using vectorized operations
            self.x[:, n+1] = self.x[:, n] + self.dr*np.cos(self.theta[:, n])
            self.y[:, n+1] = self.y[:, n] + self.dr*np.sin(self.theta[:, n])
            
    def compute_msd(self):
        """
        Compute mean squared displacement vs step number.
        
        ## Returns
        ----------------
        |tuple| : (msd, rmsd)
            Arrays of mean squared displacement and root mean squared displacement
            vs step number
        """
        # Calculate r^2 = x^2+y^2 relative to starting point
        x_disp = self.x - self.x[:, 0:1] # Displacement from start
        y_disp = self.y - self.y[:, 0:1]
        r2 = x_disp**2 + y_disp**2
        
        # Average over particles for each step number
        msd = np.mean(r2, axis=0)
        rmsd = np.sqrt(msd)
        
        return msd, rmsd
    
    def fit_power_law(self, msd):
        """
        Fit MSD data to power law of form <r^2> = Cn^alpha.
        Uses log transformation to perform linear fit.
        
        ## Keyword arguments
        --------------------------
        msd : |np.ndarray|
            Mean square displacement data
            
        ## Returns
        ----------------
        |tuple| : (alpha, C)
            Best fit power law exponent and coefficient
        """
        # Convert to log space (excluding n=0)
        n = np.arange(1, len(msd)) # Step numbers, skip n=0
        log_msd = np.log(msd[1:]) # log(<r^2>), skip n=0
        log_n = np.log(n) # log(n)
        
        # Fit line to log-transformed data
        # log(<r^2>) = log(C) + alpha*log(n)
        coef = np.polyfit(log_n, log_msd, 1)
        alpha = coef[0]
        C = np.exp(coef[1])
        
        return alpha, C
    
    def plot_2d_histograms(self, steps=[10, 1000, 5000, 10000], bin_counts=[10, 100, 1000]):
        """
        Create 2D histograms of particle positions at specified steps.
        
        ## Keyword arguments
        --------------------
        steps : |list|
            Step numbers to plot
        bin_counts : |list|
            Number of bins to use in each direction
        """
        rows = len(bin_counts)
        cols = len(steps)
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        
        # Find global range for consistent plotting
        x_min, x_max = np.min(self.x), np.max(self.x)
        y_min, y_max = np.min(self.y), np.max(self.y)
        
        for i, bins in enumerate(bin_counts):
            for j, step in enumerate(steps):
                ax = axes[i,j]
                h = ax.hist2d(self.x[:, step], self.y[:, step], 
                            bins=bins, range=[[x_min, x_max], [y_min, y_max]])
                plt.colorbar(h[3], ax=ax)
                ax.set_title(f'n={step}, bins={bins}')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                
        plt.suptitle('Particle Distributions at Different Steps and Bin Resolutions')
        plt.tight_layout()
        plt.show()

    def plot_cross_sections(self, step_number, bins=100):
        """
        Plot x=0 and y=0 cross-sections of particle distribution.
        
        ## Keyword arguments
        --------------------
        step_number : |int|
            Step at which to plot cross sections
        bins : |int|
            Number of bins for histogram
        """
        # Create 2D histogram
        x_positions = self.x[:, step_number]
        y_positions = self.y[:, step_number]
        
        # Get histogram data
        H, x_edges, y_edges = np.histogram2d(x_positions, y_positions, bins=bins)
        
        # Get bin centers
        x_centers = (x_edges[:-1]+x_edges[1:])/2
        y_centers = (y_edges[:-1]+y_edges[1:])/2
        
        # Get central slices with integer division
        central_x_bin = bins//2
        central_y_bin = bins//2
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot I(x, y=0)
        ax1.plot(x_centers, H[:, central_y_bin])
        ax1.set_xlabel('x')
        ax1.set_ylabel('I(x, y=0)')
        ax1.grid(True)
        
        # Plot I(x=0, y)
        ax2.plot(y_centers, H[central_x_bin, :])
        ax2.set_xlabel('y')
        ax2.set_ylabel('I(x=0, y)')
        ax2.grid(True)
        
        plt.suptitle(f'Distribution Cross-sections at step {step_number}')
        plt.show()
        
    def compute_2d_histogram(self, step_number=None, bins=100):
        """
        Compute a 2D histogram (intensity I(x, y)) of particle positions.

        ## Keyword arguments
        --------------------
        step_number : |int|
            The step number to use for the histogram. If None, uses the last defined step.
        bins : |int|
            Number of bins for the histogram in each direction.

        ## Returns
        ----------
        |tuple| : (hist, x_edges, y_edges)
            2D histogram data and bin edges along x and y axes.
        """
        # If no step is specified, use the last defined step
        if step_number is None:
            step_number = self.N_s

        # Extract positions at the specified step
        x_positions = self.x[:, step_number]
        y_positions = self.y[:, step_number]

        # Compute the 2D histogram
        hist, x_edges, y_edges = np.histogram2d(x_positions, y_positions, bins=bins)
        return hist, x_edges, y_edges

# Test different particle numbers
N_s = 10000
particle_counts = [10, 100, 1000, 10000]
dr = 1.0

results = []
for N_p in particle_counts:
    # Initialize and run simulation
    print(f"Running simulation with {N_p} particles...")
    rw = RandomWalkWizard(N_p=N_p, N_s=N_s, dr=dr)
    rw.simulate()
    msd, rmsd = rw.compute_msd()
    results.append((msd, rmsd))

# Plot results
n = np.arange(N_s + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot MSD
for i, N_p in enumerate(particle_counts):
    ax1.plot(n, results[i][0], label=f'N_p = {N_p}')
ax1.set_xlabel('Step number n')
ax1.set_ylabel(r'$<r^2>$')
ax1.set_title('Mean Square Displacement')
ax1.legend()
ax1.grid(True)

# Plot RMSD
for i, N_p in enumerate(particle_counts):
    ax2.plot(n, results[i][1], label=f'N_p = {N_p}')
ax2.set_xlabel('Step number n')
ax2.set_ylabel(r'$\sqrt{<r^2>}$')
ax2.set_title('Root Mean Square Displacement')
ax2.legend()
ax2.grid(True)

plt.suptitle('Random Walk Statistics vs Number of Particles')
plt.tight_layout()
plt.show()

for N_p in particle_counts:
    rw = RandomWalkWizard(N_p=N_p, N_s=N_s, dr=dr)
    rw.simulate()
    msd, rmsd = rw.compute_msd()
    alpha, C = rw.fit_power_law(msd) # Power law fit
    print(f"N_p = {N_p}:")
    print(f"  alpha = {alpha:.4f}")
    print(f"  C = {C:.4f}")

# Execute the 2D histogram plotting
# Run simulation with 10000 particles
rw = RandomWalkWizard(N_p=10000, N_s=10000, dr=1.0)
rw.simulate()
rw.plot_2d_histograms()

# Execute the cross-section plotting
# Initialize with 10000 particles
rw = RandomWalkWizard(N_p=10000, N_s=10000, dr=1.0)
rw.simulate()

# Plot cross sections at the same times used for histograms
steps = [10, 1000, 5000, 10000]
for step in steps:
    print(f"\nPlotting cross sections for step {step}")
    rw.plot_cross_sections(step)
    