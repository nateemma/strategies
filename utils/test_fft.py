
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
        # dt = 1.0 # sampling interval
        dt = 1.0/12.0 # sampling interval

        #smooth
        if smooth_data:
            y = smooth(y, 4)

        # detrend the data
        if detrend:
            t = np.arange(0, len(y))
            poly = np.polyfit(t, y, 1)
            line = np.polyval(poly, y)
            y_detrend = y - line
        else:
            y_detrend =  y

        # apply FFT
        yf = np.fft.fft(y_detrend) # FFT of detrended signal
        # yf = np.concatenate((yf[:(N+1)//2], np.zeros(N), yf[(N+1)//2:])) # zero-padding for higher resolution

        yf_filt = filter_freqs(yf, filter_type=filter_type)

        # apply IFFT
        y_pred_ifft = np.fft.ifft(yf_filt) # IFFT of predicted values
        y_pred_ifft = np.real(y_pred_ifft) # real part of IFFT values

        # add back trend

        y_pred_final = np.concatenate((y, y_pred_ifft))[-len(y):]
        if detrend:
            y_pred_final = y_pred_final + line

        predictions = y_pred_final

        return predictions


def filter_freqs(yf, filter_type=0):

    N = yf.size # number of data points

    if filter_type == 1:
        # simply remove higher frequencies
        yf_filt = yf
        index = max(4, int(N/2))
        yf_filt[index:-index] = 0.0

        # print(f'N:{N} yf_filt2:{yf_filt}')
    elif filter_type == 2:
        # bandpass filter

        # select frequencies and coefficients
        # f1, f2 = 5, 20 # frequencies in Hz
        f1, f2 = 0, 0.08 # cutoff frequencies in Hz
        # fr = np.fft.fftfreq(2*N, 1) # frequency array
        fr = np.fft.fftfreq(N, 1.0/12.0) # frequency array
        fr_sort = np.sort(np.abs(fr))
        f1 = 0.0
        f2 = fr_sort[-N//4]
        # print(f'mean:{fr_mean} median:{fr_med}')
        df = 0.08 # bandwidth of bandpass filter
        gpl = np.exp(-((fr-f1)/(2*df))**2) + np.exp(-((fr-f2)/(2*df))**2) # positive frequencies
        gmn = np.exp(-((fr+f1)/(2*df))**2) + np.exp(-((fr+f2)/(2*df))**2) # negative frequencies
        g = gpl + gmn # bandpass filter
        yf_filt = yf * g # filtered FFT coefficients

    elif filter_type == 3:
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
        # print(f'yf_filt:{yf_filt}')

    elif filter_type == 4:
        # phase filter
        # Compute the phase spectrum
        phase = np.angle (yf)

        # Define a threshold for filtering
        threshold = np.pi / 2.0
        threshold = np.mean(np.abs(phase))

        # # sort phases, set threshold from end
        # threshold = np.sort(np.abs(phase))[-N//8]

        # Apply a mask to the phase spectrum
        mask = np.abs (phase) < threshold
        yf_filt = yf * mask
        # print(f'phase:{phase}')
        # print(f'yf_filt:{yf_filt}')

    else:
        # default is no filter at all
        yf_filt = yf

    return yf_filt


#------------------------------


# test data taken from real run

test_data = np.load('test_data.npy')

data = test_data[0:min(1024, len(test_data))]

orig = data


# put the data into a dataframe
dataframe = pd.DataFrame(data, columns=["gain"])

# Plot the original data and the reconstructed data
df = pd.DataFrame(orig, index=np.arange(len(orig)))
# ax = df.plot(label='Original', marker="o", color="black")
plt.plot(dataframe["gain"], label="Original", marker="o", color="black")

detrend_data = False

dataframe["fft_0"] = fft_approximation(data, filter_type = 0, detrend=False, smooth_data=False)
plt.plot(dataframe["fft_0"], label="fft (0)", linestyle='dashed', marker="^")

dataframe["fft_0"] = fft_approximation(data, filter_type = 0, detrend=True, smooth_data=False)
plt.plot(dataframe["fft_0"], label="fft (0) detrend", linestyle='dashed', marker="v")

dataframe["fft_0"] = fft_approximation(data, filter_type = 0, detrend=False, smooth_data=True)
plt.plot(dataframe["fft_0"], label="fft (0) smooth", linestyle='dashed', marker="x")

dataframe["fft_0"] = fft_approximation(data, filter_type = 0, detrend=True, smooth_data=True)
plt.plot(dataframe["fft_0"], label="fft (0) detrend & smooth", linestyle='dashed', marker="p")

# dataframe["fft_1"] = fft_approximation(data, filter_type = 1, detrend=detrend_data, smooth_data=False)
# plt.plot(dataframe["fft_1"], label="fft (1)", linestyle='dashed', marker="s")

# dataframe["fft_2"] = fft_approximation(data, filter_type = 2, detrend=detrend_data, smooth_data=False)
# plt.plot(dataframe["fft_2"], label="fft (2)", linestyle='dashed', marker="^")

# dataframe["fft_3"] = fft_approximation(data, filter_type = 3, detrend=detrend_data, smooth_data=False)
# plt.plot(dataframe["fft_3"], label="fft (3)", linestyle='dashed', marker="v")

# dataframe["fft_4"] = fft_approximation(data, filter_type = 4, detrend=detrend_data, smooth_data=False)
# plt.plot(dataframe["fft_4"], label="fft (4)", linestyle='dashed', marker="P")


plt.legend()
plt.show()
