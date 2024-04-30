import numpy as np
import math
import matplotlib.pyplot as plt
import scipy



A = 1
SNR = np.array([-10,0,10,20,30,40,50,60])
dBinvSNR_list = 10**(SNR/10)
sigma_2 = 1/(2*dBinvSNR_list)
N = 513
n_0 = -256
P = N*(N-1)/2
Q = N*(N-1)*(2*N-1)/(6)
T = 10**-6
simulations = 1000
k = 2**10
f_0 = 100000
w_0 = 2*math.pi*f_0
phi = math.pi/8

w_hat_var = []
phi_hat_var = []
w_bias = []
phi_bias = []


CRLB_w = ((12*sigma_2)/((T**2)*N*(N**2-1)))
CRLB_phi = (12*sigma_2*(n_0**2*N+2*n_0*P+Q))/((N**2)*(N**2-1))


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
    return x , t
    
def fouriersum(estimate,x,t):
    return -np.abs(np.sum(x*np.exp(-1j*estimate*t)))
    
    


def estimator(x,t,k):
    T = 10**(-6)
    n_0 = -256
    X = np.fft.fft(x,k)
    m = np.argmax(abs(X))
    w_hatt = ((2*np.pi*m)/(k*T))
    result = scipy.optimize.minimize(fouriersum, w_hatt, args = (x,t), method ='Nelder-Mead')
    w_hatt = result["x"]


    phi_hatt = np.angle(np.exp(-1j*w_hatt*n_0*T)*np.sum(x*np.exp(-1j*w_hatt*(t+0.000256))))
    
    return w_hatt,phi_hatt



for snr in SNR:
    w_hat_list = []
    phi_hat_list = []
    w_errors = []
    phi_errors = []
    for i in range(simulations):
        x,t = functionGenerator(snr)
        w_hat, phi_hat = estimator(x,t,k)
        
        w_hat_list.append(w_hat)
        w_errors.append(w_hat-w_0)
        
        phi_hat_list.append(phi_hat)
        phi_errors.append(phi_hat-phi)
    
    
    w_hat_var.append(np.var(w_hat_list))
    w_bias.append(np.mean(w_errors))
    phi_hat_var.append(np.var(phi_hat_list))
    phi_bias.append(np.mean(phi_errors))

        
fig1, ax1 = plt.subplots()
ax1.plot(SNR,phi_hat_var, label = 'var(phi_hat)')
ax1.plot(SNR,CRLB_phi , label = 'CRLB')
plt.yscale("log")
ax1.legend()
ax1.set(xlabel = 'SNR [dB]', ylabel = 'Variance')

fig2, ax2 = plt.subplots()
ax2.plot(SNR,w_bias, label = 'E(w_hat)')
ax2.plot(SNR,phi_bias , label = 'E(phi_hat)')
ax2.legend()
ax2.set(xlabel = 'SNR [dB]', ylabel = 'bias')

fig3, ax3 = plt.subplots()
ax3.plot(SNR,w_hat_var, label = 'var(w_hat)')
ax3.plot(SNR,CRLB_w , label = 'CRLB')
plt.yscale("log")
ax3.legend()
ax3.set(xlabel = 'SNR [dB]', ylabel = 'Variance')


plt.show()
        