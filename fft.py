import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, ifft


def functionGenerator(t, A, w_0, phi, w_t):
    return A * np.exp(1j*(w_0*t + phi)) + w_t

x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])

def FFT(x): 
    return fft(x)

print(FFT(x))