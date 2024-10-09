import numpy as np
import matplotlib.pyplot as plt

# Import the custom SimpsonsRule class that uses Simpson's 1/3rd rule
# to approximate the area, or integral, of a function.
from SimpsonsRule import SimpsonsRule

# Problem 3: Damped Harmonic Oscillator with Exponentially Decaying Driving Force
# (a) Use (hardcoded) Simpson's Rule to compute the integral for x(t)
# (b) Evaluate the integral numerically and plot the solution for three sets of parameters

# DampedOscillator simulates a damped harmonic oscillator with an exponentially decaying driving force.
# Utilizing Green's functions to find the system response, x(t). The integral that calculates the displacement 
# and solves the ODE is a convolution of the Green's function and driving force. It is too difficult to do by hand. 
# The Simpson's 1/3rd rule, written by hand, is used to perform this integral as a precise approximation method.
class DampedOscillator:
    def __init__(self, mass, dampingConstant, springConstant, drivingForceAmplitude, alpha):
        """
        Initialize the DampedOscillator object's attributes.

        Use the self variable to represent the instance of the class and store the input parameters as attributes.

        Parameters:
        mass : float
            The mass of the oscillator (in kg).
        dampingConstant : float
            The damping coefficient (in N*s/m), also known as beta.
        springConstant : float
            The spring constant (in N/m).
        drivingForceAmplitude : float
            The amplitude of the driving force applied to the oscillator.
        alpha : float
            The exponential decay constant of the driving force.
        
        Attributes:
        naturalFrequency : float
            The natural frequency of the oscillator, derived from the mass and spring constant.
        """
        # Known parameters for the mechanical oscillator given as input during initialization of the 
        # DampedOscillator object.
        self.mass = mass
        self.springConstant = springConstant
        self.drivingForceAmplitude = drivingForceAmplitude
        self.alpha = alpha
        self.beta = dampingConstant  # Damping coefficient given in the 3 experiments
        
        # Derived parameters
        self.naturalFrequency = np.sqrt(self.springConstant / self.mass)  # Natural frequency

    def drivingForce(self, t):
        """
        Driving force function: f(t) = f0 * exp(-alpha * t)

        Parameters:
        t : float or array-like
            The time value or array of time values (in seconds) at which to compute the driving force.
        
        Returns:
        float or np.ndarray
            The driving force evaluated at the given time(s).
        """
        return self.drivingForceAmplitude * np.exp(-self.alpha * t)

    def computeEnergy(self, timeValues, displacementValues):
        """
        Using the energy function: E(t) = (m*v^2 + k*x^2) / 2 to acquire the energy at each time.

        Parameters:
        timeValues : array-like
            The time points (in seconds) at which the displacement was computed.
        displacementValues : array-like
            The computed displacement values x(t) corresponding to the time points.
        
        Returns:
        np.ndarray
            The computed energy values E(t) for the given time points.
        """
        # Calculate velocity using finite differences (central difference method)
        dt = np.abs(timeValues[1] - timeValues[0])  # Assuming uniform time steps
        velocityValues = np.zeros_like(displacementValues)
        
        # Central difference for velocity (except at the boundaries)
        velocityValues[1:-1] = (displacementValues[2:] - displacementValues[:-2]) / (2 * dt)
        # Forward difference for the first point
        velocityValues[0] = (displacementValues[1] - displacementValues[0]) / (2 * dt)
        # Backward difference for the last point
        velocityValues[-1] = (displacementValues[-1] - displacementValues[-2]) / (2 * dt)

        # Compute the energy at each time point
        kineticEnergy = 0.5 * self.mass * velocityValues**2
        potentialEnergy = 0.5 * self.springConstant * displacementValues**2
        totalEnergy = kineticEnergy + potentialEnergy

        return totalEnergy

    def computeDisplacement(self, timeValues):
        """
        computeDisplacement computes the displacement x(t) of the damped oscillator for a range of 
        time values. The displacement is calculated by convolving the Green's function G(t - t') with the
        driving force f(t'), using Simpson's 1/3rd Rule for numerical integration.

        Parameters:
        timeValues : array-like
            A list or array of time points (in seconds) at which to compute the displacement.
        
        Returns:
        np.ndarray
            The displacement values x(t) at the given time points.
        """
        # Define the Green's function (response function)
        def greenFunction(t):
            return np.exp(-self.beta * t) * np.sin(self.naturalFrequency * t)
        
        displacementValues = []
        for t in timeValues:
            tPrime = np.linspace(0, t, 100)
            self.drivingForce(tPrime)
            
            # Compute the integral using Simpson's 1/3rd rule
            # Instantiate a new instance of the class with initializing variables.
            # Pass a lambda as the integrand of SimpsonsRule, which is the product G(t-t')*f(t').
            simpsonsSolver = SimpsonsRule(lambda tPrime: self.drivingForce(tPrime) * greenFunction(t - tPrime), 0, t, 100)

            # Call the solve function of SimpsonsRule class to perform the approximation method
            displacementAtT = simpsonsSolver.solve()
            displacementValues.append(displacementAtT)
        
        return np.array(displacementValues)

    def plotSolution(self, timeValues, displacementValues, energyValues, label):
        """
        Plot x(t) and E(t) over time t.

        Parameters:
        timeValues : array-like
            The time points (in seconds) at which the displacement was computed.
        displacementValues : array-like
            The computed displacement values x(t) corresponding to the time points.
        energyValues : array-like
            The computed energy values E(t) corresponding to the time points.
        label : str
            A label for the plot, typically including the damping coefficient and alpha value.
        """
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Time $(s)$')
        ax1.set_ylabel('Displacement $x(t)$', color=color)
        ax1.plot(timeValues, displacementValues, color=color, label=f'Displacement ({label})')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)

        ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Energy $E(t)$', color=color)
        ax2.plot(timeValues, energyValues, color=color, linestyle='--', label=f'Energy ({label})')
        ax2.set_ylim(bottom=ax1.get_ylim()[0]) # Set the y-axis bottom limits to be the same as the first axis
        ax2.set_ylim(top=ax1.get_ylim()[1]) # Set the y-axis top limits to be the same as the first axis
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # To make sure labels do not overlap
        plt.title('Damped Oscillator Response and Energy')
        ax1.legend(bbox_to_anchor=(0.40, 1.0), loc='upper left')
        ax2.legend(bbox_to_anchor=(0.40, 0.9), loc='upper left')
  

    def runSimulation(self, timeValues):
        displacementValues = self.computeDisplacement(timeValues)
        energyValues = self.computeEnergy(timeValues, displacementValues)
        self.plotSolution(timeValues, displacementValues, energyValues, f'beta={self.beta}, α={self.alpha}')
        return displacementValues

# Run the simulation for three different sets of parameters and plot the results
if __name__ == "__main__":
    # Shared parameters for all oscillators
    mass = 1.0  # mass in kg
    springConstant = 1.0  # spring constant in N/m
    drivingForceAmplitude = 1.0  # amplitude of driving force
    time = np.linspace(0, 20, 500)  # time values for the simulations

    # 1. Set the parameters for the experiment.
    # 2. Instantiate a new instance of the DampedOscillator object, oscillator1
    # 3. Call the runSimulation method from the new instance with a list of times
    #    and store the output in displacement1.
    # 4. Repeat 1-3 for oscillator2 + displacement2 and oscillator3 + displacement3
                
    # First experiment: beta = 0.1(naturalFrequency), α = 0.3(naturalFrequency)
    dampingConstant1 = 0.1 * np.sqrt(springConstant / mass)
    alpha1 = 0.3 * np.sqrt(springConstant / mass)
    oscillator1 = DampedOscillator(mass, dampingConstant1, springConstant, drivingForceAmplitude, alpha1) 
    displacement1 = oscillator1.runSimulation(time)                           

    # Second experiment: beta = 0.2(naturalFrequency), α = 0.2(naturalFrequency)
    dampingConstant2 = 0.2 * np.sqrt(springConstant / mass)
    alpha2 = 0.2 * np.sqrt(springConstant / mass)
    oscillator2 = DampedOscillator(mass, dampingConstant2, springConstant, drivingForceAmplitude, alpha2)
    displacement2 = oscillator2.runSimulation(time)

    # Third experiment: beta = 0.3(naturalFrequency), α = 0.1(naturalFrequency)
    dampingConstant3 = 0.3 * np.sqrt(springConstant / mass)
    alpha3 = 0.1 * np.sqrt(springConstant / mass)
    oscillator3 = DampedOscillator(mass, dampingConstant3, springConstant, drivingForceAmplitude, alpha3)
    displacement3 = oscillator3.runSimulation(time)

    # Show the plot for all three experiments
    plt.show()
