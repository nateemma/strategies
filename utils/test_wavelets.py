
# Import libraries
import numpy as np
import pywt
import matplotlib.pyplot as plt

import Wavelets

from sklearn.metrics import mean_squared_error

# Create some random data
# data = np.random.normal (0, 0.1, size=1000)
num_samples = 16
np.random.seed(42) # for reproducibility
X = np.arange(num_samples) # 100 data points
# data = np.sin(X) + np.random.randn(num_samples) * 0.1
data = np.sin(X)

wavelet = Wavelets.make_wavelet(wavelet_type=Wavelets.WaveletType.MODWT)

coeffs = wavelet.get_coeffs(data)
array = wavelet.coeff_to_array(coeffs)

rec_coeffs = wavelet.array_to_coeff(array)
rec_data = wavelet.get_values(rec_coeffs)

print(f'data:{np.shape(data)} rec_data:{np.shape(rec_data)}')

mse = mean_squared_error(data, rec_data)

if mse > 1.0:
    print(f'*** High MSE: {mse}')

# Print the result
print("MSE =", mse)

# Plot the original data and the reconstructed data
plt.plot(data, label='Original', marker="o")
plt.plot(rec_data, label='Reconstructed', linestyle='dashed', marker="o")
plt.legend()
plt.show()
