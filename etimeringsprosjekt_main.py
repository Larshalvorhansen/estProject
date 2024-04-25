import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import fft, ifft


A = 1
SNR = np.array([-10,0,10,20,30,40,50,60])
dBinvSNR_list = 10**(SNR/10)
sigma_2 = 1/(2*dBinvSNR_list)
N = 513
n_0 = -256
P = N*(N-1)/2
Q = N*(N-1)*(2*N-1)/(6)
T = 10**-6
simulations = 100
k = 2**18
f_0 = 100000
w_0 = 2*math.pi*f_0
phi = math.pi/8

w_hat_var = []
phi_hat_var = []
w_bias = []
phi_bias = []


CRLB_w = ((12*sigma_2)/((T**2)*N*(N**2-1)))
CRLB_phi = 12*sigma_2*((P**2)/N+Q)/((N**2)*(N**2-1))


def functionGenerator(SNR):
    dBinvSNR = 10**(SNR/10)
    t = np.linspace(-0.000256, 0.000256, num=513)
    f_0 = 100000
    w_0 = 2*math.pi*f_0
    phi = math.pi/8
    sigma = 1/math.sqrt(2*dBinvSNR)
    A = 1
    wn_r = np.random.normal(0, sigma,t.shape)
    wn_i = np.random.normal(0, sigma,t.shape)
    x = A * np.exp(1j*(w_0*t + phi)) + wn_r + 1j*wn_i
    return(x)


def estimator(x,k):
    T = 10**(-6)
    n_0 = -256
    X = np.fft.fft(x,k)
    m = np.argmax(abs(X))
    w_hatt = ((2*np.pi*m)/(k*T))
    phi_hatt = np.angle(np.e**(-1j*w_hatt*n_0*T)*X[m])
    
    return w_hatt,phi_hatt


for snr in SNR:
    w_hat_list = []
    phi_hat_list = []
    w_errors = []
    phi_errors = []
    for i in range(simulations):
        x = functionGenerator(snr)
        w_hat, phi_hat = estimator(x,k)
        
        w_hat_list.append(w_hat)
        w_errors.append(w_hat-w_0)
        
        phi_hat_list.append(phi_hat)
        phi_errors.append(phi)
    
    
    w_hat_var.append(np.var(w_hat_list))
    w_bias.append(np.mean(w_errors))
    phi_hat_var.append(np.var(phi_hat_list))
    phi_bias.append(np.mean(phi_errors))

        
fig1, ax = plt.subplots()
ax.plot(SNR,phi_hat_var, label = 'phi_hat')
ax.plot(SNR,CRLB_phi , label = 'CRLB')
plt.yscale("log")
ax.legend()

fig1, ax = plt.subplots()
ax.plot(SNR,w_bias, label = 'w')
ax.plot(SNR,phi_bias , label = 'phi')
ax.legend()
print(w_bias)

fig2, ax = plt.subplots()
ax.plot(SNR,w_hat_var, label = 'w_hat')
ax.plot(SNR,CRLB_w , label = 'CRLB')
plt.yscale("log")
ax.legend()

        

        