# pylint: disable=invalid-name, redefined-outer-name, trailing-whitespace, line-too-long, ambiguous-variable-name

import numpy as np
import matplotlib.pyplot as plt
from random import random, randrange


class MonteCarloGas:
    """
    Monte Carlo simulation of N non-interacting point particles trapped in a 
    one-dimensional infinite well of length L.
    
    ## Keyword arguments
    --------------------
    N : |int|
        Total number of particles
    epsilon1 : |float|
        Ground state energy (per particle)
    kT : |float|
        Temperature in dimensionless energy units
    ns : |int|
        Total number of Monte Carlo sweeps
    """
    
    def __init__(self, N=1000, epsilon1=1.0, kT=10.0, ns=10000):
        self.N = N # Total number of particles
        self.epsilon1 = epsilon1 # Ground state energy
        self.kT = kT # Temperature (kB*T)
        self.ns = ns # Number of Monte Carlo sweeps
        self.beta = 1.0/kT # Inverse temperature
        
        # Initialize arrays
        self.n = np.ones(N, dtype=int) # Quantum numbers (all particles start in ground state)
        self.E = np.zeros(ns) # Total energy at each step
        
    def compute_energy_change(self, n, direction):
        """
        Compute energy change for transition n -> n +- 1.
        
        ## Keyword arguments
        --------------------------
        n : |int|
            Current quantum number
        direction : |int|
            +1 for increase, -1 for decrease
        
        ## Returns
        ----------------
        |float| : Energy change dE
        """
        n_final = n+direction
        E_initial = self.epsilon1*n**2
        E_final = self.epsilon1*n_final**2
        return E_final-E_initial
        
    def compute_total_energy(self):
        """
        Compute total energy for current state.
        
        ## Returns
        ----------------
        |float| : Total energy
        """
        return self.epsilon1*np.sum(self.n**2)
