import numpy as np
import matplotlib.pyplot as plt


def functionGenerator(t, A, w_0, phi, w_t):
    return A * np.exp(1j*(w_0*t + phi)) + w_t

def fft():