import numpy as np
import matplotlib.pyplot as plt
from FourthOrderRungeKutta import FourthOrderRungeKutta

# Constants
l = 1.0   # length of the pendulum in meters
g = 9.81  # gravitational acceleration in m/s^2
M = 1.0   # mass of the weight on the pendulum
I = M * l**2  # moment of inertia of the system

# Time parameters
T = 2 * np.pi * np.sqrt(l / g)  # period of the pendulum
num_oscillations = 2  # number of oscillations to simulate
t_total = num_oscillations * T  # total time for the simulation
step_size = 1000  # number of steps in the simulation
dt = t_total / step_size  # time step

# Function for the equations of motion for the nonlinear pendulum
def pendulum_ode(state, t):
    """
    state: [theta, omega]
    t: time (unused here since the pendulum is time-independent, but passed for generality)
    
    Returns the derivatives dtheta/dt and domega/dt.
    """
    theta, omega = state
    d_theta = omega
    d_omega = -(g / l) * np.sin(theta)
    return np.array([d_theta, d_omega])

# Analytical small-angle approximation solution: theta_approx(t)
def small_angle_approx(theta0, t):
    omega0 = np.sqrt(g / l)
    return theta0 * np.cos(omega0 * t)

# Range of small initial angles (step 9 requirement)
small_angles = np.linspace(0.01, 0.1, 5)  # range of small initial angles (in radians)

# Plot configuration
fig, ax1 = plt.subplots(figsize=(10, 6))

# Simulate for each small angle
for theta_0 in small_angles:
    omega_0 = 0.0  # initial angular velocity (rad/s), starting from rest

    # 4th Order RK solver instance for each initial condition
    rk_solver = FourthOrderRungeKutta(pendulum_ode, 0, t_total, step_size, [theta_0, omega_0], enablePlot=False)

    # Solving the ODE
    time_values, solution = rk_solver.solve()

    # Extract theta and omega from the solution
    theta_values = solution[:, 0]

    # Compute the small-angle analytical solution
    theta_approx_values = small_angle_approx(theta_0, time_values)

    # Plot the numerical and analytical solutions
    ax1.plot(time_values, theta_values, label=f'Numerical $\\theta_0={theta_0:.2f}$', linestyle='-')
    ax1.plot(time_values, theta_approx_values, label=f'Analytical $\\theta_0={theta_0:.2f}$', linestyle='--')

# Final plot configuration
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Angle (radians)')
ax1.set_xlim(0, t_total)
ax1.set_ylim(-max(small_angles), max(small_angles))
plt.title('Nonlinear vs Small-Angle Approximation for Various Initial Angles')
ax1.legend(loc='upper right')
ax1.grid(True)
plt.show()

# Now implement the energy conservation analysis as well.

# Energy computation function
def tot_E(I, omega, M, theta, g, l):
    KE = 0.5 * I * omega**2
    PE = M * g * l * (1 - np.cos(theta))
    return KE + PE

# Initialize variables to store energy differences for each initial angle
energy_diff = []

# Re-run the simulation for each small angle to compute energy differences
for theta_0 in small_angles:
    omega_0 = 0.0  # initial angular velocity

    # RK solver for each angle
    rk_solver = FourthOrderRungeKutta(pendulum_ode, 0, t_total, step_size, [theta_0, omega_0], enablePlot=False)

    # Solve the ODE
    time_values, solution = rk_solver.solve()

    # Extract theta and omega
    theta_values = solution[:, 0]
    omega_values = solution[:, 1]

    # Compute energy values
    E_values = tot_E(I, omega_values, M, theta_values, g, l)
    E_0 = E_values[0]  # initial energy

    # Compute fractional energy difference
    frac_E_diff = (E_values - E_0) / E_0

    # Store the mean of the absolute fractional energy difference
    avg_frac_E_diff = np.mean(np.abs(frac_E_diff))
    energy_diff.append(avg_frac_E_diff)

    # Print or display fractional energy difference
    print(f"Initial angle: {theta_0:.2f} rad, Avg fractional energy difference: {avg_frac_E_diff:.2e}")

# Plotting the fractional energy difference for each small angle
plt.figure(figsize=(8, 5))
plt.plot(small_angles, energy_diff, 'o-', color='r', label='Energy Conservation Error')
plt.xlabel('Initial Angle (radians)')
plt.ylabel('Avg Fractional Energy Difference')
plt.title('Energy Conservation Error vs Initial Angle')
plt.grid(True)
plt.legend()
plt.show()
