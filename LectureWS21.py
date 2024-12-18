import numpy as np
import matplotlib.pyplot as plt

# pylint: disable=invalid-name
# Parameters
Np = 1000 # Number of particles
Ns = 100 # Number of steps
dx = 1.0 # Step size
dt = 1.0 # Time step

# Step 1: Create arrays to store positions of particles at each step
positions = np.zeros((Np, Ns+1))  # Shape: (Np, Ns+1)

# Array to store squared distances of particles from the origin at each step
squared_distances = np.zeros((Np, Ns+1)) # Shape: (Np, Ns+1)

# Step 2: Array for Mean Squared Displacement (MSD)
# MSD is calculated across all particles at each time step
msd = np.zeros(Ns+1) # Shape: (Ns+1,)

# Explanation of dimensions and sizes:

# - positions: A 2D array with dimensions (Np, Ns+1), storing positions for
#              Np particles over Ns steps (+1 for initial step).

# - squared_distances: Same shape as positions, storing x^2 values for each
#                      particle at each step.

# - msd: A 1D array with Ns+1 elements, representing the mean squared displacement
#        at each step.

# Random walk simulation
for step in range(1, Ns+1):
    # Each particle moves randomly left (-dx) or right (+dx)
    random_steps = np.random.choice([-dx, dx], size=Np)
    positions[:, step] = positions[:, step-1] + random_steps
    squared_distances[:, step] = positions[:, step]**2

# Calculate MSD
msd = np.mean(squared_distances, axis=0)

# Display the results

# Plot the mean squared displacement
plt.plot(np.arange(Ns+1)*dt, msd)
plt.xlabel('Time (t)')
plt.ylabel('Mean Squared Displacement (MSD)')
plt.title('Mean Squared Displacement vs Time')
plt.grid()
plt.show()
