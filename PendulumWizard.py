# pylint: disable=invalid-name, redefined-outer-name, trailing-whitespace, line-too-long, ambiguous-variable-name

"""This script simulates the motion of a nonlinear pendulum using the 4th-order Runge-Kutta method."""

import numpy as np
import matplotlib.pyplot as plt

from FourthOrderRungeKutta import FourthOrderRungeKutta

# Global module constants
l = 1.0   # length of the pendulum in meters
g = 9.81  # gravitational acceleration in m/s^2
theta_0 = np.pi / 4  # initial angle in radians, converts to 45 degrees
omega_0 = 0.0  # initial angular velocity (rad/s), starting from rest
num_oscillations = 2  # number of oscillations to simulate
M = 1.0  # mass of the weight on the pendulum
I = M*l**2  # moment of inertia of the system

# Time parameters
T = 2*np.pi*np.sqrt(l/g)  # period of the pendulum
t_total = num_oscillations*T  # total time for the simulation
step_size = 1000  # Resolution of dt
dt = t_total/step_size  # time step


def pendulum_ode(state, t):
    """
    Function for the equations of motion for the nonlinear pendulum.
    
    ## Keyword arguments
    --------------------------
    state : |arraylike|
        Current state of the system [theta, omega], where 
        <ol>
            <li>theta: angle</li>
            <li>omega: angular velocity</li>
        </ol>
    t : |float|
        NOTE: Time value, not used in this function but required by the solver.
    
    ## Returns
    ----------------
    |np.ndarray| Derivatives [d_theta, d_omega] w.r.t. time.
    """
    theta, omega = state
    d_theta = omega
    d_omega = -(g / l) * np.sin(theta)
    return np.array([d_theta, d_omega])

# 4th Order RK solver instance
rk_solver = FourthOrderRungeKutta(pendulum_ode, 0, t_total, step_size, [theta_0, omega_0], enablePlot=False)

# Solving the ODE
time_values, solution = rk_solver.solve()

# Extract theta and omega from the solution
theta_values = solution[:, 0]
omega_values = solution[:, 1]

def tot_E(I, omega, M, theta, g, l):
    """
    Computes the total energy of the system
    
    ## Keyword arguments
    --------------------------
    I : |float|
        Moment of inertia of the system.
    omega : |np.ndarray|
        Angular velocity values.
    M : |float|
        Mass of the weight.
    theta : |np.ndarray|
        Angle values.
    g : |float|
        Gravitational acceleration.
    l : |float|
        Length of the pendulum.
    
    ## Returns
    ----------------
    |np.ndarray| Total energy values or the Hamiltonian.
    """
    KE = 0.5*I*omega**2 # Kinetic energy
    PE = M*g*l*(1-np.cos(theta)) # Potential energy
    return KE+PE # Total energy

# Energy computation
E_values = tot_E(I, omega_values, M, theta_values, g, l)
E_0 = E_values[0]  # initial energy

# Compute fractional energy difference
frac_E_diff = (E_values-E_0) / E_0
avg_frac_E_diff = np.mean(frac_E_diff)

# Define the threshold for energy conservation
threshold = 1e-8
outside_threshold = not np.all(np.abs(frac_E_diff) <= threshold)

if outside_threshold:
    print("Fractional Energy did not remain within threshold")
else:
    print("All Fractional Energy values remained in threshold")

# Plot results
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot theta vs time on the first y-axis. This should show the system oscillating
ax1.plot(time_values, theta_values, 'b-', label='Theta (radians)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angle (radians)', color='b')
ax1.tick_params('y', colors='b')
ax1.set_ylim(min(theta_values)*(1+0.1), max(theta_values)*(1+0.1))
ax1.set_xlim(min(time_values), max(time_values))

# Create a second y-axis to plot fractional energy difference
ax2 = ax1.twinx()
ax2.plot(time_values, frac_E_diff, 'r-', label='Fractional Energy Difference')
ax2.set_ylabel('Fractional Energy Difference', color='r')
ax2.tick_params('y', colors='r')
ax2.set_ylim(min(frac_E_diff), max(frac_E_diff))
# plt.annotate(f'Fractional Energy Difference: {avg_frac_E_diff:.2e}', 
#              xy=(1, 1), xycoords='axes fraction', fontsize=12, color='red',
#              xytext=(-10, -10), textcoords='offset points',  
#              ha='right', va='top',  
#              bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))
plt.title('Nonlinear Pendulum Motion with Energy Conservation')
ax1.grid(True)
plt.show()
