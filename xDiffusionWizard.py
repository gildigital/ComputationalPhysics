# pylint: disable=invalid-name, redefined-outer-name, trailing-whitespace, line-too-long, ambiguous-variable-name

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 10 # Total time for simulation
D = 0.5 # Diffusion coefficient
A = 1.0 # Initial concentration at x=0
L = np.sqrt(2*D*T)*(10) # Half-length of the spatial domain
dx = 0.05 # Spatial step size
dt = dx**2/(2*D)*0.50 # Time step size

alpha = D*dt/dx**2

if dt > dx**2/(2*D):
    raise ValueError(f'Stability condition not met: dt = {dt} > {dx**2/(2*D)}')

if alpha >= 0.5:
    raise ValueError('Stability condition not met: alpha =', alpha)

# Grid setup
nx = int(L/dx) + 1 # Number of spatial points (covers 0 to L)
nt = int(T/dt) # Number of time steps

# Initialize arrays
x = np.linspace(0, L, nx)
xi = np.zeros(nx) # Concentration array at time step n
xi_new = np.zeros(nx) # Concentration array at time step n+1
xi[nx//2] = A # Initial condition: peak at x=0 (center of the grid)

# List to store <x> and <x^2> as functions of time
mean_x = []
mean_x2 = []

# Time evolution using FTCS
for n in range(nt):
    # Apply FTCS update for each interior spatial point
    for i in range(1, nx-1):
        xi_new[i] = xi[i] + alpha*(xi[i+1] - 2*xi[i] + xi[i-1])

    # Boundary conditions
    xi_new[0] = 0
    xi_new[-1] = 0

    # Update xi for the next time step
    xi = xi_new.copy()
    
    # Calculate <x> and <x^2> at the current time step
    current_mean_x = np.sum(x*xi)*dx
    current_mean_x2 = np.sum(x**2*xi)*dx
    
    # Store these values in the lists
    mean_x.append(current_mean_x)
    mean_x2.append(current_mean_x2)

    # Plot at select intervals
    if n % (nt//100) == 0: # Plot every [10, 10%], [100, 1%], [1000,0.1%] of total time
        plt.plot(x, xi)

plt.xlabel('Position x')
plt.ylabel(r'Concentration $\xi$')
plt.title('Diffusion of initial peak over time')
plt.show()

# Plot <x> and <x^2> as functions of time
time = np.linspace(0, T, nt)
plt.figure()
plt.plot(time, mean_x, label=r'$\langle x \rangle$')
plt.xlabel('Time')
plt.ylabel('Mean Values')
plt.ylim(mean_x[0]-0.1, mean_x[0]+0.1)
plt.title(r'Evolution of $\langle x \rangle$ over time')
plt.legend()
plt.show()

plt.figure()
plt.plot(time, mean_x2, label=r'$\langle x^2 \rangle$')
plt.xlabel('Time')
plt.ylabel('Mean Values')
plt.title(r'Evolution of $\langle x^2 \rangle$ over time')
plt.legend()
plt.show()
