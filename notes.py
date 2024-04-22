import numpy as np
import matplotlib.pyplot as plt

# Define the function that generates x(t)
def generate_signal(t, A, w_0, phi, w_t):
    return A * np.exp(1j*(w_0*t + phi)) + w_t

# Generate random values for A, w_0, phi
A = np.random.uniform(0, 5)
w_0 = np.random.uniform(0, 2*np.pi)
phi = np.random.uniform(0, 2*np.pi)

# Time vector
t = np.linspace(0, 10, 1000)

# Generate a random noise component w(t)
w_t = np.random.normal(0, 0.5, t.shape)

# Generate the signal
x_t = generate_signal(t, A, w_0, phi, w_t)

# Plot the real part of the signal since it's a complex number
plt.plot(t, x_t.real)
plt.title('Real part of the signal x(t)')
plt.xlabel('Time t')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

