import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import fft, ifft


def functionGenerator(SNR):
    A = 1
    t = np.linspace(0,5, num=513)
    wn = np.random.normal(0, 0.5, t.shape)
    phi = math.pi/8
    x = A * np.exp(1j*(w_0*t + phi)) + SNR*wn
    return(x,A)

x = functionGenerator(1)
print(A)
plt.plot(x)
plt.title('x(t)')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()