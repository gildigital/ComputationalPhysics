import numpy as np
import matplotlib.pyplot as plt

# TODO: Replace python's simpson rule with hardcoded one
from scipy.integrate import simpson

# Problem 3: Damped Harmonic Oscillator with Exponentially Decaying Driving Force
# (a) Use Simpson's Rule to compute the integral for x(t)
# (b) Evaluate the integral numerically and plot the solution for three sets of parameters

class DampedOscillator:
    def __init__(self, mass, dampingConstant, springConstant, drivingForceAmplitude, alpha):
        """
        Initialize the DampedOscillator object's attributes: mass m, damping constant b, spring constant k, 
        driving force amplitude f0, and exponential decay constant alpha.

        Use the self variable to represent the instance of the class and store the input parameters as attributes.
        """
        # Known parameters for the mechanical oscillator given as input during initialization of the 
        # DampedOscillator object.
        self.mass = mass
        self.dampingConstant = dampingConstant
        self.springConstant = springConstant
        self.drivingForceAmplitude = drivingForceAmplitude
        self.alpha = alpha
        
        # Derived parameters
        self.naturalFrequency = np.sqrt(self.springConstant / self.mass)  # Natural frequency
        self.beta = self.dampingConstant / (2 * self.mass)  # Damping coefficient

    def drivingForce(self, t):
        """
        Driving force function: f(t) = f0 * exp(-alpha * t)
        """
        return self.drivingForceAmplitude * np.exp(-self.alpha * t)

    def computeDisplacement(self, timeValues):
        """
        Compute x(t) by numerically integrating using Simpson's rule.
        """
        # Define the Green's function (response function)
        greenFunction = lambda t: np.exp(-self.beta * t) * np.sin(self.naturalFrequency * t)
        
        displacementValues = []
        for t in timeValues:
            tPrime = np.linspace(0, t, 100)
            drivingForcePrime = self.drivingForce(tPrime)
            greenValues = greenFunction(t - tPrime)
            
            # Compute the integral using Simpson's rule from scipy
            # TODO: This will have to be replaced with hard coded Simpson's rule
            integrand = drivingForcePrime * greenValues
            displacementAtT = simpson(integrand, tPrime)
            displacementValues.append(displacementAtT)
        
        return np.array(displacementValues)

    def plotSolution(self, timeValues, displacementValues, label):
        """
        Plot x(t) over time t.
        """
        plt.plot(timeValues, displacementValues, label=label)
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement x(t)')
        plt.title('Damped Oscillator Response')
        plt.grid(True)

    def runSimulation(self, timeValues):
        """
        Run the simulation for specific time values and plot the result.
        """
        displacementValues = self.computeDisplacement(timeValues)
        self.plotSolution(timeValues, displacementValues, f'beta={self.beta}, α={self.alpha}')
        return displacementValues

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
plt.legend()
plt.show()
