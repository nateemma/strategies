
# program for testing FFT approximation approaches

# Import libraries
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from regex import F
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import xarray

import Wavelets
import Forecasters

from sklearn.metrics import mean_squared_error

#------------------------------

def smooth(y, window):
    box = np.ones(window) / window
    y_smooth = np.convolve(y, box, mode="same")
    # Hack: constrain to 3 decimal places (should be elsewhere, but convenient here)
    y_smooth = np.round(y_smooth, decimals=3)
    return np.nan_to_num(y_smooth)

#------------------------------

def fft_approximation(x, filter_type=-1, detrend=True, smooth_data=True):

        y = np.array(x).reshape(-1)

        # print(f'y:{np.shape(y)}')

        N = y.size # number of data points
        dt = 1.0 # sampling interval

        #smooth
        if smooth_data:
            y = smooth(y, 6)
        
        # detrend the data
        if detrend:
            '''
            t = np.arange(0, N*dt, dt) # time array
            p = np.polyfit(t, y, 1) # linear fit
            yhat = np.polyval(p, t) # fitted values
            y_detrend = y - yhat # detrended signal
            '''

            f = np.fft.rfft(y)
            f[5:] = 0.0
            yhat = np.fft.irfft(f, N)
            y_detrend =  y - yhat
        else:
            y_detrend =  y

        # apply FFT
        yf = np.fft.fft(y_detrend) # FFT of detrended signal
        yf = np.concatenate((yf[:(N+1)//2], np.zeros(N), yf[(N+1)//2:])) # zero-padding for higher resolution
        # fr = np.fft.fftfreq(2*N, dt) # frequency array
        # yf_abs = np.abs(yf) # magnitude of FFT coefficients

        yf_filt = filter_freqs(yf, filter_type=filter_type)

        # apply IFFT
        y_pred_ifft = np.fft.ifft(yf_filt) # IFFT of predicted values
        y_pred_ifft = np.real(y_pred_ifft) # real part of IFFT values

        # add back trend
        '''
        yhat_pred = np.polyval(p, t_pred) # fitted values for prediction
        y_pred_final = y_pred_ifft + yhat_pred # final predicted values
        '''
        y_pred_final = np.concatenate((y, y_pred_ifft))[-len(y):]
        if detrend:
            y_pred_final = y_pred_final + yhat

        predictions = y_pred_final

        return predictions

def filter_freqs(yf, filter_type=0):

    N = yf.size # number of data points

    if filter_type == 0:
        # simply remove higher frequencies
        yf_filt = yf
        index = max(4, int(len(yf)/2))
        yf_filt[index:] = 0.0

    elif filter_type == 1:
        # bandpass filter

        # select frequencies and coefficients
        # f1, f2 = 5, 20 # frequencies in Hz
        f1, f2 = 0, 0.08 # cutoff frequencies in Hz
        # fr = np.fft.fftfreq(2*N, 1) # frequency array
        fr = np.fft.fftfreq(N, 1) # frequency array
        fr_sort = np.sort(np.abs(fr))
        f1 = 0.0
        f2 = fr_sort[-N//4]
        # print(f'mean:{fr_mean} median:{fr_med}')
        df = 0.08 # bandwidth of bandpass filter
        gpl = np.exp(-((fr-f1)/(2*df))**2) + np.exp(-((fr-f2)/(2*df))**2) # positive frequencies
        gmn = np.exp(-((fr+f1)/(2*df))**2) + np.exp(-((fr+f2)/(2*df))**2) # negative frequencies
        g = gpl + gmn # bandpass filter
        yf_filt = yf * g # filtered FFT coefficients

    elif filter_type == 2:
        # power spectrum filter
        # Compute the power spectrum
        ps = np.abs (yf)**2

        # Define a threshold for filtering
        # threshold = 100
        # threshold = np.mean(ps)
        threshold = np.sort(ps)[-N//2]

        # Apply a mask to the fft coefficients
        mask = ps > threshold
        yf_filt = yf * mask

    elif filter_type == 3:
        # phase filter
        # Compute the phase spectrum
        phase = np.angle (yf)

        # Define a threshold for filtering
        # threshold = 2.0
        # threshold = np.mean(np.abs(phase))

        threshold = np.sort(np.abs(phase))[-N//4]

        # Apply a mask to the phase spectrum
        mask = np.abs (phase) < threshold
        yf_filt = yf * mask

    else:
        # default is no filter at all
        yf_filt = yf

    return yf_filt

#------------------------------


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

# use gen_data for debugging, test_data for the real thing

# data = np.array(gen_data)
data = np.array(test_data)
# data = RobustScaler().fit_transform(data.reshape(-1,1)).reshape(-1)

orig = data


# put the data into a dataframe
dataframe = pd.DataFrame(data, columns=["gain"])

# Plot the original data and the reconstructed data
df = pd.DataFrame(orig, index=np.arange(len(orig)))
# ax = df.plot(label='Original', marker="o", color="black")
plt.plot(dataframe["gain"], label="Original", marker="o", color="black")

detrend_data = True

# dataframe["smooth"] = smooth(data, 6)
# plt.plot(dataframe["smooth"], label="smooth", linestyle='dashed', marker="o")

# dataframe["fft_ff"] = fft_approximation(data, filter_type = -1, detrend=False, smooth_data=False)
# plt.plot(dataframe["fft_ff"], label="fft (-1, False, False)", linestyle='dashed', marker="o")

dataframe["fft_tf"] = fft_approximation(data, filter_type = -1, detrend=True, smooth_data=False)
plt.plot(dataframe["fft_tf"], label="fft (-1, True, False)", linestyle='dashed', marker="x")

# dataframe["fft_ft"] = fft_approximation(data, filter_type = -1, detrend=False, smooth_data=True)
# plt.plot(dataframe["fft_ft"], label="fft (-1, False, True)", linestyle='dashed', marker="^")

# dataframe["fft_tt"] = fft_approximation(data, filter_type = -1, detrend=True, smooth_data=True)
# plt.plot(dataframe["fft_tt"], label="fft (-1, True, True)", linestyle='dashed', marker="x")

# dataframe["fft_0ft"] = fft_approximation(data, filter_type = 0, detrend=False, smooth_data=False)
# plt.plot(dataframe["fft_0ft"], label="fft_0ft", linestyle='dashed', marker="o")

dataframe["fft_0tf"] = fft_approximation(data, filter_type = 0, detrend=True, smooth_data=True)
plt.plot(dataframe["fft_0tf"], label="fft_0tf", linestyle='dashed', marker="^")

# dataframe["fft_1"] = fft_approximation(data, filter_type = 1, detrend=False, smooth_data=False)
# plt.plot(dataframe["fft_1"], label="fft_1", linestyle='dashed', marker="s")

dataframe["fft_2"] = fft_approximation(data, filter_type = 2, detrend=detrend_data, smooth_data=False)
plt.plot(dataframe["fft_2"], label="fft_2", linestyle='dashed', marker="^")

dataframe["fft_3_tf"] = fft_approximation(data, filter_type = 3, detrend=True, smooth_data=False)
plt.plot(dataframe["fft_3_tf"], label="fft_3_tf", linestyle='dashed', marker="v")

# dataframe["fft_3_ff"] = fft_approximation(data, filter_type = 3, detrend=False, smooth_data=False)
# plt.plot(dataframe["fft_3_ff"], label="fft_3_ff", linestyle='dashed', marker="P")


plt.legend()
plt.show()
