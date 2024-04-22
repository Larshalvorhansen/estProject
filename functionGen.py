import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import fft, ifft


def functionGenerator(SNR):
    dBinvSNR = math.pow(10,(SNR)/10)
    t = np.linspace(-0.000256, 0.000256, num=513)
    f_0 = 100000
    w_0 = 2*math.pi*f_0
    phi = math.pi/8
    sigma = 1/math.sqrt(2*SNR)
    A = 1
    wn = np.random.normal(0, sigma,t.shape)
    x = A * np.exp(1j*(w_0*t + phi)) + wn
    print(SNR)
    print(dBinvSNR)
    return(x,t)

x, t = functionGenerator(15)
plt.plot(t,x)
plt.title('x(t)')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
