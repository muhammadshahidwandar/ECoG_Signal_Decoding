import pywt
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import signal
from scipy.signal import butter, sosfilt


def rolling_window(a, window_size):
    shape = (a.shape[0] - window_size, window_size) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    print(shape, strides)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def make_scalogram(data, freq, window_size):
    N = 50  ## used to save memory
    data = rolling_window(data, window_size)
    n_steps = math.ceil(data.shape[0] / N)
    X = []
    for i in range(n_steps):
        res_ = pywt.cwt(data=data[i * N: (i + 1) * N], wavelet='morl', scales=freq, axis=1)[0]
        # if config["sqr_signal"]:
        res_ = res_.transpose((1, 3, 0, 2))[:, :, :, ::2] ** 2
        # res_=res_.transpose((1,3,0,2))[:,:,:,::2] #shahid modified
        # else:
        #    res_=np.abs(res_.transpose((1,3,0,2))[:,:,:,::10])
        X.append(res_)
    return np.concatenate(X, axis=0)



def bandpass_filter(data, lowcut, highcut, inplace=False, fs=100, order=7):
    nyq = fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, (low, high), btype='band', analog=False, output='sos')
    filtered_signal = np.array([sosfilt(sos, data.signal[:, i]) for i in range(data.signal.shape[1])])
    if inplace:
        data.signal = filtered_signal.T
    return filtered_signal.T


def median_filter(data, f_size):
    lgth, num_signal = data.shape
    f_data = np.zeros([lgth, num_signal])
    for i in range(num_signal):
        f_data[:, i] = signal.medfilt(data[:, i], f_size)
    return f_data


def mean_filter(data, f_size):
    lgth, num_signal = data.shape
    f_data = np.zeros([lgth, num_signal])
    for i in range(num_signal):
        f_data[:, i] = signal.av(data[:, i], f_size)
    return f_data


def Datapreprocess(data):
    # X = data.signal
    y = data.motion
    t = data.time
    # Apply bandpass filtering and test the data
    fs = int(np.floor(np.mean(1. / np.diff(t))))
    lowcut = 0.1
    highcut = 600
    X_fltr = bandpass_filter(data, lowcut, highcut, inplace=False, fs=fs, order=7)
    avg_x = np.average(X_fltr, axis=1)
    X = X_fltr - avg_x[:, None]

    avg_y = np.average(y, axis=0)
    Y = y - avg_y#[:, None]
    # apply media filter
    window = 7
    Y = median_filter(Y, window)
    return X,Y
def ZeroMeanNorm(X):
    X_mx = X.max()
    X_mn = X.min()
    X_nrm = (X-X_mn)/(X_mx-X_mn+1)
    return X_nrm



# 3D plot of signal###########################
def plot3Dsignal(data, Result):
    for el in range(2):#data.signal.shape[1]):
        fig = plt.figure(figsize=(15, 11))
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(projection='3d')

        # Make data.
        Y = np.linspace(10, 150, 10)
        X = data.time[:120000:1000]
        Z = pywt.cwt(Result[:120000:1000, el], wavelet='morl', scales=Y)[0]
        X, Y = np.meshgrid(X, Y)

        #Z = ZeroMeanNorm(Z)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_xlabel('time, sec')
        ax.set_ylabel('frequency, Hz')
        ax.set_zlabel('cwt')
        ax.set_title("Electrode " + str(el))
        # ax.set_zlim(-1.01, 1.01)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=10)

        plt.show()
