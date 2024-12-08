import numpy as np
import matplotlib.pyplot as plt

# Morse potential function
def morse_potential(x, D, a, x0):
    """
    Compute the Morse potential V(x).
    
    ## Keyword arguments
    --------------------------
    x : |np.ndarray|
        Position values.
    D : |float|
        Dissociation energy.
    a : |float|
        Potential width parameter.
    x0 : |float|
        Equilibrium bond length.
        
    ## Returns
    ----------------
    |np.ndarray| : Potential values at x.
    """
    return D*(np.exp(-2*a*(x-x0)) - 2*np.exp(-a*(x-x0)))

# Parameters
D = 10 # Dissociation energy
a = 1.0 # Potential width parameter
x0 = 0.0 # Equilibrium bond length

# Range of x for plotting
x = np.linspace(-2, 2, 1000)
V = morse_potential(x, D, a, x0)

# Compute FWHM
z1 = 1 + np.sqrt(2)/2
z2 = 1 - np.sqrt(2)/2
x1 = x0 - np.log(z1)/a
x2 = x0 - np.log(z2)/a
fwhm = x2 - x1

# Plot the Morse potential
plt.figure(figsize=(10, 6))
plt.plot(x, V, label="Morse Potential $V(x)$")
plt.axhline(-D, color="r", linestyle="--", label="$V_{\min} = -D$")
plt.axvline(x0, color="g", linestyle="--", label="$x_0$")
plt.axvline(x1, color="b", linestyle="--", label="$x_1$ (FWHM)")
plt.axvline(x2, color="b", linestyle="--", label="$x_2$ (FWHM)")
plt.title("Morse Potential")
plt.xlabel("$x$")
plt.ylabel("$V(x)$")
plt.legend()
plt.grid(True)
plt.show()

# Part 6: Classical Particle Motion
# Not needed to be implemented in the notebook... but take a look
# Example energy level E < 0
E = -5

# Solve for turning points (x1, x2)
def turning_points(D, a, x0, E):
    """
    Compute the turning points for a classical particle with energy E in the Morse potential.
    """
    z1 = np.log(1 + np.sqrt(1 - E/D))
    z2 = np.log(1 - np.sqrt(1 - E/D))
    return x0 - z1/a, x0 - z2/a

tp1, tp2 = turning_points(D, a, x0, E)

# Plots
plt.figure(figsize=(10, 6))
plt.plot(x, V, label=r"Morse Potential $V(x)$")
plt.axhline(E, color="purple", linestyle="--", label=fr"Energy $E = {E}$")
plt.axvline(tp1, color="orange", linestyle="--", label=r"$x_1$ (Turning Point)")
plt.axvline(tp2, color="orange", linestyle="--", label=r"$x_2$ (Turning Point)")
plt.title("Classical Particle Motion in Morse Potential")
plt.xlabel(r"$x$")
plt.ylabel(r"$V(x)$")
plt.legend()
plt.grid(True)
plt.show()
