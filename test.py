# import numpy as np
# import math
# import matplotlib.pyplot as plt
# from scipy import fft, ifft

# def functionGenerator():
#     t = np.linspace(0, 0.000513, num=513)
#     return(t)

# print(functionGenerator())


import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()