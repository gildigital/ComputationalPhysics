# pylint: disable=invalid-name, redefined-outer-name, trailing-whitespace, line-too-long, ambiguous-variable-name

"""StandingWaveWizard solves standing wave problems using Simponson's Rule and visualizes the results."""
import numpy as np
import matplotlib.pyplot as plt

from SimpsonsRule import SimpsonsRule


# Problem Set 3: Standing Wave Wizard
class StandingWaveWizard:
    """StandingWaveWizard class for solving standing wave problems."""
    def __init__(self, length=1.0, waveProblem='fixed-free', numPoints=1000):
        """
        Initialize the wave problem solver.
        
        <H4>Keyword arguments</H4>
        --------------------------
        length : |float|
            Length of the string
        waveProblem : |str|
            Type of wave problem to solve ('fixed-free', or 'fixed-initial', etc...)
        numPoints : |int|
            Number of points for visualization
        """
        self.length = length
        self.x = np.linspace(0, length, numPoints)
        self.waveProblem = waveProblem
        self.waveSpeed = 1.0  # wave speed
        self.numPoints = numPoints
        
    def computeMode(self, modeNumber):
        """
        Compute the nth standing wave mode.
        """
        if self.waveProblem == 'fixed-free':
            k = (2*modeNumber-1) * np.pi/(2*self.length)
            return np.sin(k*self.x)
        elif self.waveProblem == 'fixed-initial':
            k = modeNumber*np.pi/self.length
            return np.sin(k*self.x)
        elif self.waveProblem == 'connected-strings':
            return self.computeConnectedStringMode(modeNumber)
    
    def computeConnectedStringMode(self, modeNumber):
        """
        Compute the standing wave mode for connected strings.
        This handles two strings with different wave speeds connected at the middle.
        """
        halfL = self.length/2  # Length of each half (String A and String B)
        xA = np.linspace(0, halfL, self.numPoints//2)
        xB = np.linspace(halfL, self.length, self.numPoints//2)

        # Phase speed of waves on String A and String B
        vA = self.waveSpeed
        vB = 2 * self.waveSpeed

        # Calculate frequencies based on the mode number and the two different phase speeds
        kA = modeNumber*np.pi/halfL
        kB = (modeNumber*np.pi/halfL)*vA/vB

        # Create the wave for each part of the string
        yA = np.sin(kA*xA)  # Standing wave on String A
        
        yB = np.sin(kB * (xB+halfL))  # Standing wave on String B, shifted to start at halfL
        
        # Concatenate both parts of the string to get the full wave
        return np.concatenate([yA, yB])

    
    def initialShape(self, x):
        """
        Compute the initial shape xi(x,t=0) for fixed-initial problem.
        """
        result = np.zeros_like(x)
        
        # Define regions according to problem
        mask1 = x < self.length/3
        mask2 = (x >= self.length/3) & (x <= 2*self.length/3)
        mask3 = x > 2 * self.length/3
        
        # Apply piecewise function
        result[mask1] = 3*x[mask1]/self.length
        result[mask2] = 1
        result[mask3] = -3*x[mask3]/self.length + 3
        
        return result
    
    def plotModes(self, numModes=3):
        """
        Plot standing wave modes or initial shape based on problem type.
        """
        if self.waveProblem == 'fixed-free':
            self.plotFixedFreeModes(numModes)
        elif self.waveProblem == 'connected-strings':
            self.plotConnectedStringModes(numModes)
        else:
            self.plotFixedInitialAnalysis()
    
    def plotFixedFreeModes(self, numModes):
        """
        Plot modes for fixed-free string problem.
        """
        fig, axes = plt.subplots(numModes, 1, figsize=(8, 2*numModes))
        fig.suptitle('Standing Wave Modes for Fixed-Free String')
        
        for i, ax in enumerate(axes):
            mode = i

            y = self.computeMode(mode)
            
            ax.plot(self.x, y, 'b-')
            ax.plot([0, self.length], [0, 0], 'k--', alpha=0.3)
            ax.set_ylim(-1.2, 1.2)
            ax.set_xticks(np.arange(0, self.length + 0.1, 0.1))
            
            ax.plot(0, 0, 'ro', label='Fixed end')
            ax.plot(self.length, y[-1], 'bo', label='Free end')
            
            if i == 0:
                ax.legend(loc='center right')
            
            ax.set_title(rf'Mode {mode}: k = {(2 * mode - 1)}$\pi$/2L')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def plotFixedInitialAnalysis(self):
        """
        Plot analysis for fixed-initial conditions problem.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle('Analysis of String with Initial Shape')
        
        # Plot initial shape
        ax1.plot(self.x, self.initialShape(self.x), 'b-', label='Initial shape')
        ax1.plot([0, self.length], [0, 0], 'k--', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('x/L')
        ax1.set_ylabel(r'$\xi$(x,0)')
        ax1.set_title('Initial Shape')
        ax1.legend()
        
        # Plot first few modes
        for n in range(1, 4):
            cn = self.computeCn(n)
            y = cn * self.computeMode(n)
            ax2.plot(self.x, y, label=f'Mode {n} (c{n}={cn:.3f})')
        
        ax2.plot([0, self.length], [0, 0], 'k--', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('x/L')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('First Three Modes')
        ax2.legend()
        
        plt.tight_layout()
        
    def plotConnectedStringModes(self, numModes):
        """
        Plot the modes for the connected-strings problem.
        """
        fig, axes = plt.subplots(numModes, 1, figsize=(8, 2 * numModes))
        fig.suptitle('Standing Wave Modes for Connected Strings')

        for i, ax in enumerate(axes):
            mode = 2 * (i + 1)  # Use even numbers: 2, 4, 6, ...
            y = self.computeConnectedStringMode(mode)

            ax.plot(self.x, y, 'b-')
            ax.plot([0, self.length], [0, 0], 'k--', alpha=0.3)
            ax.set_ylim(-1.2, 1.2)
            ax.set_xticks(np.arange(0, self.length+0.1, 0.1))

            ax.plot(0, 0, 'ro', label='Fixed end')
            ax.plot(self.length/2, y[self.numPoints//2], 'bo', label='Node at center')
            ax.plot(self.length, 0, 'ro', label='Fixed end')

            if i == 0:
                ax.legend(loc='center right')

            ax.set_title(f'Mode {mode}: Different Speeds on A and B')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
    
    def computeCn(self, modeNumber):
        """
        Compute nth Fourier coefficient using Simpson's Rule.
        """
        def integrandN(x):
            """"""
            return self.initialShape(x)*np.sin(modeNumber*np.pi*x/self.length)
        
        # The nth Fourier coefficient contains the integral of the product of the initial shape
        # and the nth mode function. We use Simpson's Rule to approximate this integral.
        simpson = SimpsonsRule(integrandN, 0, self.length, 100)
        
        # Solve the integral using the SimpsonsRule.solve() method
        return (2 / self.length) * simpson.solve()
    
    def animateString(self, totalTime=1.0, frames=60, numModes=10):
        """
        Animate string motion for fixed-initial problem using plt.pause().
        """
        if self.waveProblem != 'fixed-initial':
            print("Animation only available for fixed-initial problem")
            return
            
        plt.figure(figsize=(10, 6))
        for time in np.linspace(0, totalTime, frames):
            plt.clf()  # Clear the current figure
            y = self.computeMotion(time, numModes)  # Compute string motion at time t
            plt.plot(self.x, y, 'b-')  # Plot the string motion with blue line
            plt.plot([0, self.length], [0, 0], 'k--', alpha=0.3)  # Plot the fixed ends
            plt.grid(True, alpha=0.3)  # Add grid lines
            plt.ylim(-1.5, 1.5)  # Set y-axis limits
            plt.xlabel('x/L')  # Label of normalized x-axis
            plt.ylabel(r'$\xi$(x,t)')  # Label of string motion
            plt.title(f'String Motion at t = {time:.2f}')
            plt.pause(0.033)  # Crude animation mechanism
    
    def computeMotion(self, time, numModes=10):
        """
        Compute string motion at time t.
        
        By default, it computes the motion using the first 10 modes.
        Increasing numModes will provide a more accurate representation.
        """
        result = np.zeros_like(self.x)
        
        for n in range(1, numModes+1):
            cn = self.computeCn(n) # Compute Fourier coefficient
            omegaN = n * np.pi * self.waveSpeed / self.length # Angular frequency
            result += cn * np.sin(n * np.pi * self.x / self.length) * np.cos(omegaN * time) # Add mode contribution
        
        return result

# Run the StandingWaveWizard class to solve the problems for Problem Set 3
if __name__ == "__main__":
    # Solve Problem 1 (Fixed-Free)
    print("Solving Problem 1: Fixed-Free String")
    wizard1 = StandingWaveWizard(waveProblem='fixed-free')
    wizard1.plotModes(3)
    
    # Solve Problem 2 (Fixed-Initial)
    print("\nSolving Problem 2: String with Initial Shape")
    wizard2 = StandingWaveWizard(waveProblem='fixed-initial')
    wizard2.plotModes()  # Shows initial shape
    
    # Animate Problem 2
    print("\nAnimating Problem 2 solution...")
    
    T = 0.1  # Total time for animation
    fps = 0.033  # Frame per second
    frames = int(T / fps)  # Number of frames
    wizard2.animateString(totalTime=T, frames=frames, numModes=30)
    
    # Solve Problem 3 (Connected Strings)
    print("Solving Problem 3: Connected Strings")
    wizard3 = StandingWaveWizard(waveProblem='connected-strings', length=2.0)
    wizard3.plotModes(3)
    
    plt.show()
