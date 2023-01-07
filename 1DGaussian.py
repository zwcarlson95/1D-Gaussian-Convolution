import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

#Generates random data points between 0.00000000 and 1.00000000
rng = np.random.default_rng()

#Draws 101 samples from a standard normal distribution
#Find the cumulative sum of all elements in the generated array
x = rng.standard_normal(101).cumsum()

#Apply gaussian convolution to the array with a kernel size of 3
y3 = gaussian_filter1d(x, 3)

#applies another gaussian filter with a kernel size of 6
y6 = gaussian_filter1d(x, 6)

#Plot the data
plt.plot(x, 'k', linewidth=1, label='original data')
plt.plot(y3, 'c--', linewidth=1.5, label='sigma 3 filtered')
plt.plot(y6, 'r--', linewidth=1.5, label='sigma 6 filtered')

plt.legend()
plt.grid()
plt.show()
