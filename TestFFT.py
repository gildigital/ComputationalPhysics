"""This class tests np.fft.rfft() using sinusoidal input"""

import numpy as np
import matplotlib.pyplot as plt

class TestFFT:
    """
    ## Parameters:
    --------------
    fs : |int|
        Sampling frequency.
    T : |float|
        Total time.
    t : |np.ndarray|
        Time array.
    f : |float|
        Frequency of the sinusoidal signal.
    x : |np.ndarray|
        Sinusoidal signal.
    X : |np.ndarray|
        Fourier transform of x.
    freqs : |np.ndarray|
        Frequency
    """
    def __init__(self):
        self.fs = 2**15
        self.T = 20
        self.t = np.arange(0, self.T, 1/self.fs)
        self.f = 10
        self.x = np.sin(2*np.pi*self.f*self.t)
        self.X = np.fft.rfft(self.x)
        self.freqs = np.fft.rfftfreq(len(self.x), 1/self.fs)

    def plot(self):
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(self.t, self.x)
        plt.title('Time domain')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.subplot(2, 1, 2)
        plt.plot(self.freqs, np.abs(self.X))
        plt.title('Frequency domain')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.show()
        
    
tester = TestFFT()
TestFFT.plot(tester)