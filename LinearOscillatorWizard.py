import numpy as np
import matplotlib.pyplot as plt
from FourthOrderRungeKutta import FourthOrderRungeKutta

# LinearOscillatorWizard solves the damped driven oscillator problem using the 4th-order Runge-Kutta method.


class LinearOscillatorWizard:
    def __init__(self, omega0, omega_d, f0, beta, x0, v0, t0, t_end, dt, verify=True):
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
        verify : bool
            If True, verify that the displacement graph matches the particular solution.
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
        self.verify = verify

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

    def particular_solution(self, t):
        """
        Computes the particular solution for the damped driven oscillator.

        Parameters:
        t : float
            Time value at which to evaluate the particular solution.

        Returns:
        float
            The particular solution at time t.
        """
        amplitude = self.f0 / np.sqrt((self.omega0**2 - self.omega_d**2)**2 + (2 * self.beta * self.omega_d)**2)
        phase_shift = np.arctan2(2 * self.beta * self.omega_d, self.omega0**2 - self.omega_d**2)
        return amplitude * np.cos(self.omega_d * t + phase_shift)  
      
    def verify_particular_solution(self, time_values, x_values):
        """
        Verifies that the displacement graph matches the particular solution.
        We plot the numerical solution and the particular solution over the last 50 time units
        as that's when the numerical solution should reach a steady state and resemble the 
        particular solution.

        Parameters:
        time_values : np.ndarray
            Array of time points.
        x_values : np.ndarray
            Displacement values over time.
        """
        # Compute the particular solution at each time point
        particular_solution_values = np.array([self.particular_solution(t) for t in time_values])

        # Create a mask to select time values between 450 and 500
        mask = (time_values >= max(time_values) - 50) & (time_values <= max(time_values))

        # Plot the particular solution and the numerical solution
        plt.plot(time_values[mask], x_values[mask], 'b-', label='Numerical Solution')
        plt.plot(time_values[mask], particular_solution_values[mask], 'r--', label='Particular Solution')
        print(time_values)
        plt.xlabel('Time $t$')
        plt.ylabel('Displacement $x(t)$')
        plt.title('Verification of Particular Solution')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

    def run_simulation(self):
        """
        Runs the simulation for the oscillator and plots the results.
        """
        # Initial state [x(0), v(0)]
        initial_state = np.array([self.x0, self.v0])

        # Define a function for RK4 that takes state [x, v] and t and returns the state update
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

        # If verification is enabled, plot the particular solution alongside the numerical solution
        if self.verify:
            if self.t_end > 50:
                self.verify_particular_solution(time_values, x_values)
            else:
                print("Verification is only available for t_end > 50.")

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

        mask = (time_values >= 0) & (time_values <= 50)

        # Plot displacement
        ax1.plot(time_values[mask], x_values[mask], 'b-', label='Displacement $x(t)$')
        ax1.set_xlabel('Time $t$ (s)')
        ax1.set_ylabel('Displacement $x(t)$', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)

        # Create a second y-axis for velocity
        ax2 = ax1.twinx()
        ax2.plot(time_values[mask], v_values[mask], 'r--', label='Velocity $v(t)$')
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
    omega0 = 0.5   # Natural frequency (increased)
    omega_d = 0.618  # Driving frequency (close to resonance)
    f0 = 1.0        # Amplitude of the driving force
    beta = 0.1     # Damping coefficient (decreased for weak damping)
    x0 = 0.0        # Initial displacement
    v0 = 0.0        # Initial velocity
    t0 = 0.0        # Start time
    t_end = 500    # End time
    dt = 0.01       # Time step

    # Create an instance of the LinearOscillatorWizard and run the simulation
    oscillator = LinearOscillatorWizard(omega0, omega_d, f0, beta, x0, v0, t0, t_end, dt, verify=True)
    oscillator.run_simulation()
