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
simulations = 100
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


    phi_hatt = np.angle(np.exp(-1j*w_hatt*n_0*T)*np.sum(x*np.exp(-1j*w_hatt*(t+0.000256))))
    
    return w_hatt,phi_hatt

w_var_array = np.zeros((6,8))
phi_var_array = np.zeros((6,8))

w_bias_array = np.zeros((6,8))
phi_bias_array = np.zeros((6,8))

for j in range(6):
    k =2**(10+2*j)
    for i in range(len(SNR)):
        w_hat_list = []
        phi_hat_list = []
        w_errors = []
        phi_errors = []
        for s in range(simulations):
            x,t = functionGenerator(SNR[i])
            w_hat, phi_hat = estimator(x,t,k)
            
            w_hat_list.append(w_hat)
            w_errors.append(w_hat-w_0)
            
            phi_hat_list.append(phi_hat)
            phi_errors.append(phi_hat-phi)
        
        w_var_array[j,i] = np.var(w_hat_list)
        phi_var_array[j,i] = np.var(phi_hat_list)

        w_bias_array[j,i] = np.mean(w_errors)
        phi_bias_array[j,i] = np.mean(phi_errors)
        
fig1,axs1 = plt.subplots(3,2)
fig2,axs2 = plt.subplots(3,2)
fig3,axs3 = plt.subplots(3,2)
for i in range(3):
    for j in range(2):
        r = i*2+j
        axs1[i,j].plot(SNR,w_var_array[r,:],label = 'var(w_hat)')
        axs1[i,j].plot(SNR,CRLB_w, label = 'CRLB')
        axs1[i,j].set_title('Results using M = 2^' + str(10+2*r) + ' point FFT')
        axs1[i,j].set(xlabel = 'SNR [dB]', ylabel = 'Variance',yscale = 'log')
        axs1[i,j].legend()

        axs2[i,j].plot(SNR,phi_var_array[r,:],label = 'var(phi_hat)')
        axs2[i,j].plot(SNR,CRLB_phi, label = 'CRLB')
        axs2[i,j].set_title('Results using M = 2^' + str(10+2*r) + ' point FFT')
        axs2[i,j].set(xlabel = 'SNR [dB]', ylabel = 'Variance', yscale = 'log')
        axs2[i,j].legend()

        axs3[i,j].plot(SNR,w_bias_array[r,:],label = 'w_hat bias')
        axs3[i,j].plot(SNR,phi_bias_array[r,:], label = 'phi_hat bias')
        axs3[i,j].set_title('Results using M = 2^' + str(10+2*r) + ' point FFT')
        axs3[i,j].set(xlabel = 'SNR [dB]', ylabel = 'bias')
        axs3[i,j].legend()
        
plt.show()