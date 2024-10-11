import numpy as np
import matplotlib.pyplot as plt
from FourthOrderRungeKutta import FourthOrderRungeKutta

class LinearOscillatorWizard:
    def __init__(self, omega0, omega_d, f0, beta, x0, v0, t0, t_end, dt):
        """
        Initialize the LinearOscillatorWizard class.

        Parameters:
        omega0 : float
            Natural frequency of the oscillator.
        omega_d : float
            Driving frequency.
        f0 : float
            Amplitude of the driving force.
        beta : float
            Damping coefficient.
        x0 : float
            Initial displacement.
        v0 : float
            Initial velocity.
        t0 : float
            Initial time.
        t_end : float
            End time for the simulation.
        dt : float
            Time step for the Runge-Kutta method.
        """
        self.omega0 = omega0
        self.omega_d = omega_d
        self.f0 = f0
        self.beta = beta
        self.x0 = x0
        self.v0 = v0
        self.t0 = t0
        self.t_end = t_end
        self.dt = dt

    def coupled_ode_system(self, state, t):
        """
        Represents the coupled first-order ODE system.

        Parameters:
        state : list or np.ndarray
            Current state of the system [x, v], where x is displacement and v is velocity.
        t : float
            Current time.

        Returns:
        list
            Derivatives [dx/dt, dv/dt] as defined by the coupled ODEs.
        """
        x, v = state
        dxdt = v
        dvdt = -2 * self.beta * v - self.omega0**2 * x + self.f0 * np.sin(self.omega_d * t)
        return np.array([dxdt, dvdt])

    def run_simulation(self):
        """
        Runs the simulation for the oscillator and plots the results.
        """
        # Initial state [x(0), v(0)]
        initial_state = np.array([self.x0, self.v0])

        # Define a function for RK4 that takes x and t and returns the state update
        def func(state, t):
            return self.coupled_ode_system(state, t)

        # Create an instance of FourthOrderRungeKutta to solve the system
        rk4_solver = FourthOrderRungeKutta(func, self.t0, self.t_end, int((self.t_end - self.t0) / self.dt), initial_state, enablePlot=False)
        
        # Solve using RK4
        time_values, solution = rk4_solver.solve()

        # Extract displacement and velocity
        x_values = solution[:, 0]
        v_values = solution[:, 1]

        # Plot results
        self.plot_results(time_values, x_values, v_values)

    def plot_results(self, time_values, x_values, v_values):
        """
        Plot the displacement and velocity over time.

        Parameters:
        time_values : np.ndarray
            Array of time points.
        x_values : np.ndarray
            Displacement values over time.
        v_values : np.ndarray
            Velocity values over time.
        """
        fig, ax1 = plt.subplots()

        # Plot displacement
        ax1.plot(time_values, x_values, 'b-', label='Displacement $x(t)$')
        ax1.set_xlabel('Time $t$ (s)')
        ax1.set_ylabel('Displacement $x(t)$', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)

        # Create a second y-axis for velocity
        ax2 = ax1.twinx()
        ax2.plot(time_values, v_values, 'r--', label='Velocity $v(t)$')
        ax2.set_ylabel('Velocity $v(t)$', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.title('Damped Oscillator: Displacement and Velocity')
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Parameters for the oscillator
    omega0 = 1.0  # Natural frequency
    omega_d = 1.0  # Driving frequency
    f0 = 0.5  # Amplitude of the driving force
    beta = 0.1  # Damping coefficient
    x0 = 0.0  # Initial displacement
    v0 = 0.0  # Initial velocity
    t0 = 0.0  # Start time
    t_end = 50.0  # End time
    dt = 0.01  # Time step

    # Create an instance of the LinearOscillatorWizard and run the simulation
    oscillator = LinearOscillatorWizard(omega0, omega_d, f0, beta, x0, v0, t0, t_end, dt)
    oscillator.run_simulation()
