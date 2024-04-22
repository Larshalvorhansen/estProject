import numpy  as np

def estimator(k,x):
    T = 10**(-6)
    n_0 = -256
    X = np.fft.fft(x,k)
    m = np.argmax(X)
    w_hatt = ((2*np.pi*m)/(k*T))
    phi_hatt = np.angle(np.e**(-1j*w_hatt*n_0*T)*X[m])
    
    return w_hatt,phi_hatt