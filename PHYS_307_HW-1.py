import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# Problem 1: Projectile Motion
# (a) Compute the horizontal range of the projectile in terms of v_i and theta_i
# (b) Find the launch angle, theta_0 that would maximize the area under the trajectory curve.
# - Feel free to leave as an algebraic equation that you can solve for theta_0 in terms of given information
# (c) Outline the computation for part (b). 
# (d) Write down the equation of motion for a projectile subject to resistive force of the form F_r = -b*v, where F_r and v are vectors.
# - How do you expect the trajectory to change? No need to solve the diff. eqs.; we will numerically approach this.

# (a)

class Projectile:
    def __init__(self, v_i, theta_i, x_i=0, y_i=0, g=9.8):
        """
        Initialize the projectile with initial velocity: v_i [m/s], angle: theta_i [degrees], and 
        initial position: x_i, y_i [m]. 
        """
        self.v_i = v_i
        self.theta_i = np.radians(theta_i)  # Convert to radians
        self.x_i = x_i
        self.y_i = y_i
        self.g = g
        
        # Initial velocity components
        self.v_ix = self.v_i * np.cos(self.theta_i)
        self.v_iy = self.v_i * np.sin(self.theta_i)

    def compute_trajectory(self, num_points=100):
        """
        Compute the x and y positions of the projectile's trajectory.
        First, calculates the maximum flight time. Then, calculates the final x and y positions.

        Returns the final x and y positions.
        """
        # Time of flight calculation
        t_max = 2 * self.v_iy / self.g
        t = np.linspace(0, t_max, num=num_points) # dimension of time, up to the max time

        # Calculate trajectory
        x_f = self.x_i + self.v_ix * t
        y_f = self.y_i + self.v_iy * t - 1/2 * self.g * t**2
        
        return x_f, y_f

    def compute_range(self):
        """
        Compute the horizontal range of the projectile.
        """
        return (self.v_i ** 2 * np.sin(2 * self.theta_i)) / self.g

    def compute_area(self, num_points=100):
        """
        Compute the area under the trajectory curve using numerical integration method: Simpson's rule.

        Returns the area.
        """
        x_f, y_f = self.compute_trajectory(num_points=num_points)
        area = simpson(y_f, x=x_f)  # Using Simpson's rule for integration
        return area

    def maximize_area(self, theta_min=0, theta_max=90, num_angles=100):
        """
        Find the launch angle that maximizes the area under the trajectory curve.
        """
        angles = np.linspace(theta_min, theta_max, num=num_angles)
        max_area = 0
        best_angle = 0

        for angle in angles:
            self.theta_i = np.radians(angle)
            self.v_ix = self.v_i * np.cos(self.theta_i)
            self.v_iy = self.v_i * np.sin(self.theta_i)
            area = self.compute_area()
            
            if area > max_area:
                max_area = area
                best_angle = angle

        return best_angle, max_area

    def plot_trajectory(self):
        """
        Plot the projectile's trajectory.
        """
        x_f, y_f = self.compute_trajectory()
        plt.plot(x_f, y_f, label=f'Trajectory at {np.degrees(self.theta_i):.2f} degrees')
        plt.xlabel('Horizontal Distance (m)')
        plt.ylabel('Vertical Distance (m)')
        plt.title('Projectile Motion')
        plt.grid(True)
        plt.legend()
        plt.show()


# Initialize the projectile with initial velocity and angle
v_i = 20
theta_i = 10

projectile = Projectile(v_i, theta_i)

print(f"The initial launch angle is: {theta_i} degrees")

# Compute and print the range
print(f"Horizontal range: {projectile.compute_range():.2f} meters")
# Compute and print the area
print(f"Initial area: {projectile.compute_area():.2f} square meters")

# Plot the initial trajectory
projectile.plot_trajectory()

# Find the launch angle that maximizes the area under the trajectory
optimal_angle, max_area = projectile.maximize_area()
print(f"Optimal launch angle to maximize area: {optimal_angle:.2f} degrees")
print(f"Maximum area under the trajectory: {max_area:.2f} square meters")

# Plot the trajectory for the optimal angle
projectile.theta_i = np.radians(optimal_angle)  # Set the angle to the optimal one
projectile.plot_trajectory()






