
# Import libraries
import numpy as np
import pywt
import matplotlib.pyplot as plt

import Wavelets

from sklearn.metrics import mean_squared_error

'''
# test data taken from real run
test_data = [  0.02693603,  0.78708102,  0.29854797,  0.27140725, -0.08078632, -0.08078632,
 -0.88864952, -0.56550424, -0.06764984,  0.10826905, -0.24255491, -0.24255491,
 -0.06792555, -1.78740691, -1.23206066, -1.37893741, -1.82358503, -2.90422802,
 -1.98477433, -0.59285813, -0.87731323, -1.27484578, -1.41717116,  0.01391208,
 -0.29126214,  0.13869626,  0.        , -0.15273535,  0.36287509,  0.02782028,
  0.1391014 ,  0.20775623, -0.58083253, -0.61187596, -0.77875122, -0.77875122,
  0.12501736, -0.3731859 ,  0.26429267,  0.85350497,  1.02312544,  1.02312544,
  0.        ,  0.        ,  0.        ,  0.        , -0.15260821, -0.15260821,
  0.16648169,  0.16648169,  0.16648169, -0.84628191, -0.69473392, -0.69473392,
 -0.47091413, -0.47091413, -0.77562327,  0.08395131, -0.30782146, -0.43374843,
 -0.97411634, -0.79320902, -0.48855388, -0.95065008, -0.29473684, -0.16863406,
  0.14052839, -0.04208164,  0.04208164,  0.57868737,  0.30968468, -0.16891892,
 -0.64552344, -0.98231827, -0.75715087, -1.24894752, -1.15071569, -0.535815,
 -0.36723164, -0.02834467,  0.25430913,  2.23106437,  2.82509938,  1.57357528,
  1.57357528,  1.31840091,  0.62006764, -0.88963025, -0.86980533, -0.58618283,
 -0.58618283, -0.76955366,  0.09803922, -0.09817672, -0.79387187, -0.02807806,
 -0.02807806,  0.40891145, -0.363789 , -0.02807806, -0.02807806,  0.,
  0.3932032 ,  0.3932032 ,  0.61789075,  0.82853532,  1.33408229,  0.983008,
  0.74136243,  0.74136243,  0.51639916,  0.30640669, -0.1940133 ,  0.91781393,
  1.55512358,  1.11080255,  1.0413774 ,  1.0413774 ,  0.6942516 ,  1.01970511,
 -0.36915505,  1.11233178,  1.2367734 ,  1.26425725,  0.20683949, -0.19096985,
  0.60381501, -0.47534972 ]

# Create some random data
num_samples = 128
np.random.seed(42)
f1 = np.random.randn()
np.random.seed(43)
f2 = np.random.randn()
np.random.seed(44)
f3 = np.random.randn(num_samples)

X = np.arange(num_samples)  # 100 data points
gen_data = f1 * np.sin(0.5*X) + f2 * np.cos(0.5*X) + f3 * 0.3

# data = gen_data
data = np.array(test_data)
'''

test_data = np.load('test_data.npy')

data = test_data[0:min(512, len(test_data))]

print(f'pywt version: {pywt.__version__}')

wavelet = Wavelets.make_wavelet(wavelet_type=Wavelets.WaveletType.FFTA)

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
