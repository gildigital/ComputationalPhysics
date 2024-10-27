# pylint: disable=invalid-name, redefined-outer-name, trailing-whitespace, line-too-long, ambiguous-variable-name

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

class Projectile:
    def __init__(self, v_i, theta_i, x_i=0, y_i=0, g=9.8):
        """
        Initialize the projectile with initial velocity: v_i [m/s], angle: theta_i [degrees], and 
        initial position: x_i [m], y_i [m]. 
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
        """
        # Time of flight calculation
        t_max = 2 * self.v_iy / self.g
        t = np.linspace(0, t_max, num=num_points)  # Time array

        # Calculate trajectory
        x_f = self.x_i + self.v_ix * t
        y_f = self.y_i + self.v_iy * t - 0.5 * self.g * t**2
        
        return x_f, y_f

    def compute_range(self):
        """
        Compute the horizontal range of the projectile.
        """
        return (self.v_i ** 2 * np.sin(2 * self.theta_i)) / self.g

    def compute_area(self, num_points=100):
        """
        Compute the area under the trajectory curve using Simpson's rule.
        """
        x_f, y_f = self.compute_trajectory(num_points=num_points)
        area = simpson(y_f, x=x_f)
        return area

    def maximize_area(self, theta_min=0, theta_max=90, num_angles=100):
        """
        Find the launch angle that maximizes the area under the trajectory curve.
        """
        angles = np.linspace(theta_min, theta_max, num=num_angles)
        max_area = 0
        best_angle = 0

        for angle in angles:
            theta_rad = np.radians(angle)
            v_ix = self.v_i * np.cos(theta_rad)
            v_iy = self.v_i * np.sin(theta_rad)

            # Compute trajectory for this angle without modifying self
            t_max = 2 * v_iy / self.g
            t = np.linspace(0, t_max, num=100)
            x_f = self.x_i + v_ix * t
            y_f = self.y_i + v_iy * t - 0.5 * self.g * t**2
            area = simpson(y_f, x=x_f)
            
            if area > max_area:
                max_area = area
                best_angle = angle

        return best_angle, max_area

    def set_theta(self, theta_i):
        """
        Set a new launch angle and update velocity components.
        """
        self.theta_i = np.radians(theta_i)
        self.v_ix = self.v_i * np.cos(self.theta_i)
        self.v_iy = self.v_i * np.sin(self.theta_i)

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

    def print_equations_with_drag(self, b, m):
        print("Equations of motion with a resistive drag force F_r = -b v:")
        print(f"dv_x/dt = -({b} / {m}) * v_x")
        print(f"dv_y/dt = -g - ({b} / {m}) * v_y")


# Main program exists in a while loop that allows the user to run the program multiple times
# The program ends when the user enters 'q' or 'Q

# Initialize the projectile with initial velocity and angle
v_i = int(input("Enter the initial velocity in m/s [press enter for default: 20] :: ").strip() or "20")
theta_i = int(input("Enter the initial launch angle in degrees [press enter for default: 10] :: ").strip() or "10")

projectile = Projectile(v_i, theta_i)

print(f"\nThe initial launch angle is: {theta_i} degrees")

# Part (a): Compute and print the horizontal range and equation
print("\nPart (a):")
print("The horizontal range R is given by:")
print("R = (v_i^2 * sin(2 * theta_i)) / g")
# Compute and print the range
print(f"Horizontal range: {projectile.compute_range():.2f} meters")
# Compute and print the area
print(f"Initial area: {projectile.compute_area():.2f} square meters")

# Plot the initial trajectory
projectile.plot_trajectory()

print("\nParts (b) and (c):")
# Part (b): Find the launch angle that maximizes the area under the trajectory
# Part (c): Explain how (b) is done (and do it!)
# Find the launch angle that maximizes the area under the trajectory
print("\nWe will iteratively search for the launch angle that maximizes the area under the trajectory")
print("curve by computing the area for each angle in a specified range from 0 to 90 degrees. The area is")
print("calculated using Simpson's rule for numerical integration over the projectile's trajectory")

optimal_angle, max_area = projectile.maximize_area()
print(f"\nOptimal launch angle to maximize area: {optimal_angle:.2f} degrees")
print(f"Maximum area under the trajectory: {max_area:.2f} square meters")

# Update the projectile with the optimal angle
projectile.set_theta(optimal_angle)

# Compute and print the new range
print(f"New horizontal range: {projectile.compute_range():.2f} meters")

# Plot the trajectory for the optimal angle
projectile.plot_trajectory()

plt.show()

# Part (d): Equations with resistive force
print("\nPart (d):")
b = float(input("Enter the damping coefficient b [press enter for default: 0.1] :: ").strip() or "0.1")
m = float(input("Enter the mass m of the projectile in kg [press enter for default: 1.0] :: ").strip() or "1.0")
projectile.print_equations_with_drag(b, m)