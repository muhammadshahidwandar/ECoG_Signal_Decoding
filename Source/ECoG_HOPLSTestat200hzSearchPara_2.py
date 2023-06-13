import numpy as np
from matplotlib import pyplot as plt
import scipy
import ECoG
import pywt
import time
import pandas as pd
import math
import os
import seaborn as sns
import utils
from HoplsUtils import *

from hopls import matricize, qsquared, HOPLS
import torch

from sklearn.cross_decomposition import PLSRegression

import seaborn as sns
sns.set_style('darkgrid')
sns.set(font_scale=1.3)
from sklearn.metrics import mean_squared_error
import tensorflow as tf

def normalize1(data):
    data -= data.mean(dim=0)
    return data


def normalize2(data):
    data -= data.mean(dim=0)
    data /= data.std(dim=0, unbiased=False)
    return data


def normalize3(data):
    return torch.nn.functional.normalize(data, dim=0, p=2)

# N-Pls
def compute_q2_Npls(tdata, tlabel, vdata, vlabel, Rval):
    test = PLSRegression(n_components=Rval)
    test.fit(tdata, tlabel)
    Y_pred = test.predict(vdata)
    Q2 = qsquared(matricize(vlabel), matricize(Y_pred))
    return Q2#,test


# unfold-Pls
def compute_q2_pls(tdata, tlabel, vdata, vlabel, Rval):
    test = PLSRegression(n_components=Rval)
    test.fit(matricize(tdata), matricize(tlabel))
    Y_pred = test.predict(matricize(vdata))
    Q2 = qsquared(matricize(vlabel), matricize(Y_pred))
    return Q2,test

# get the model
def compute_q2_hopls(tdata, tlabel, vdata, vlabel, la, R_max=20):
    Ln = [la] * (len(tdata.shape) - 1)
    if len(tlabel.shape) > 2:
        Km = [la] * (len(tlabel.shape) - 1)
    else:
        Km = None
    test = HOPLS(R_max, Ln, Km)
    test.fit(tdata, tlabel)
    score, r, Q2 = test.predict(vdata, vlabel)
    return r, Q2#, test

if __name__ == "__main__":

    normal_list = [normalize2]#[normalize1, normalize2, normalize3]

    base_dir='../Data/'
    subdir = os.listdir(base_dir)
    fullPath = base_dir+ subdir[0]
    Ecog = fullPath+ "/ECoG.csv"
    Motion = fullPath+ "/Motion.csv"
    x,y = ECoG.read_ECoG_from_csv(Ecog, Motion)
    data = ECoG.ECoG(x, y, downsample = False)

    X = data.signal
    y = data.motion
    # devide y in hand and wrist separatly
    t = data.time
    #Apply bandpass filtering and test the data
    X,y = utils.Datapreprocess(data)
    # data is downsampled at 20Hz
    Train_lst = 600000
    Valid_lst = 900000
    #freq = np.linspace(1, 15, 10)

    #freq = np.linspace(np.log(10),np.log(150),10)


    # generate center frequencies arranged in a logarithmic scale
    f_min = 10
    num_frequencies= 10
    f_max = 150
    #freq = np.logspace(np.log10(f_min), np.log10(f_max), num_frequencies)
    input_scales = np.logspace(np.log10(.1), np.log10(10), 10) # frequencies 16 to 160
    #input_scales = np.logspace(np.log10(1.082), np.log10(16), 10)    #frequencies 10 to 150
    #input_scales = np.logspace(np.log10(1.083), np.log10(16.25), 10)# frequency is 10 to 150
    downsamplng = 5  #2,5,10,.... the original sampling frequency is 1k
    window = 200

        #X = X[::20]

    fnl_Downsample = 20
    ### Apply low pass filtering before downsampling
    Train_data = X[0:Train_lst:downsamplng]
    Valid_Data = X[Train_lst:Valid_lst:downsamplng]
    scalo_V = utils.make_scalogram(Valid_Data,input_scales,window,fnl_Downsample)
    scalo_T = utils.make_scalogram(Train_data,input_scales,window,fnl_Downsample)
    #### Down sampling on scalograms @ 5 10x5=50
    X_V = scalo_V[::10]#25,10,5(scalo_V.reshape(scalo_V.shape[:-2]+(-1,)))#[::20]
    X_T = scalo_T[::10]#(scalo_T.reshape(scalo_T.shape[:-2]+(-1,)))#[::20]
    X_V = X_V#.reshape(X_V.shape[:-2]+(-1,))
    X_T = X_T#.reshape(X_T.shape[:-2]+(-1,))

    for i in range(1):
        #####labels
        Y_t= y[1000:Train_lst:50]#,i]#[::20]
        Y_v = y[(Train_lst+1000):Valid_lst:50]#,i]#[::20]
        #Y_v = np.reshape(Y_v, (Y_v.shape[0], -1)) #, int((Y_v.shape[1])/3)
        Y_v = np.reshape(Y_v, (Y_v.shape[0],-1, int((Y_v.shape[1])/3)))#-1,

        #Y_t = np.reshape(Y_t, (Y_t.shape[0], -1)) #int((Y_t.shape[1])/3),
        Y_t = np.reshape(Y_t, (Y_t.shape[0],-1,  int((Y_t.shape[1])/3)))

        #Y = Y[::20]
    ############################################################# Apply PLS and HOPLS
        #for train_idx, valid_idx in cv.split(X_T, Y_t):
        #     X_train = torch.Tensor(X_T[train_idx])
        #     Y_train = torch.Tensor(Y_t[train_idx])
        #     X_valid = torch.Tensor(X_T[valid_idx])
        #     Y_valid = torch.Tensor(Y_t[valid_idx])

        X_train = torch.Tensor(X_T)
        Y_train = torch.Tensor(Y_t)#[:,0]
        X_valid = torch.Tensor(X_V)
        Y_valid = torch.Tensor(Y_v)


        #for norm in normal_list:
        norm2 = normalize2
        X_train = norm2(X_train)
        X_valid = norm2(X_valid)
        Y_train = norm2(Y_train)
        Y_valid = norm2(Y_valid)
        results =[]
        #results=compute_q2_Npls(X_train, Y_train, X_valid, Y_valid, 5)
        for R in range(1, 20):
            results.append(compute_q2_pls(X_train, Y_train, X_valid, Y_valid, R))
            print('results',results)
        # results = Parallel(n_jobs=2)(
        # delayed(compute_q2_pls)(X_train, Y_train, X_valid, Y_valid, R)
        # for R in range(1, 50))
        old_Q2 = -np.inf
        for i in range(19):
            Q2,model = results[i]
            if Q2 > old_Q2:
                best_r = i + 1
                old_Q2 = Q2
                Model = model

        print('PLS q2 Best score',old_Q2)
        print('PLS Best R',best_r)
        results =[]

        ##########################Data Plot
        fs = 20
        dataLen =  X_valid.shape[0]
        max_time = dataLen/fs
        time_steps = np.linspace(0, max_time, dataLen)
        y_pred = Model.predict(matricize(X_valid))
        plt.figure(figsize=(15, 3))
        plt.subplot(3, 1, 1)

        plt.plot(time_steps, Y_valid[:,0] , color='g', label='Hand_grnd_x')
        plt.plot(time_steps, y_pred[:,0], color='r', label='Hand_prctd_x')

        plt.legend()
        ###############
        plt.subplot(3, 1, 2)

        plt.plot(time_steps, Y_valid[:,1] , color='g', label='Hand_grnd_y')
        plt.plot(time_steps, y_pred[:,1], color='r', label='Hand_prctd_y')
        plt.legend()
        plt.subplot(3, 1, 3)

        plt.plot(time_steps, Y_valid[:,2] , color='g', label='Hand_grnd_z')
        plt.plot(time_steps, y_pred[:,2], color='r', label='Hand_prctd_z')
        plt.xlabel("Time [s]")
        plt.legend()
        plt.show()
        # for lam in range(1, 16):
        #     results.append(compute_q2_hopls(X_train, Y_train, X_valid, Y_valid, lam, R_max=50))
        #
        #
        # # results = Parallel(n_jobs=2)(
        # # delayed(compute_q2_hopls)(X_train, Y_train, X_valid, Y_valid, lam, R_max=50) #X_valid, Y_valid,
        # # for lam in range(1, 16)
        # # )
        # old_Q2 = -np.inf
        # for i in range(15):
        #     r, Q2 = results[i]
        #     if Q2[r-1] > old_Q2:
        #         best_lam = i + 1
        #         best_r = r
        #         old_Q2 = Q2[r-1]
        # print('HOPLS q2 Best score',old_Q2)
        # print('HOPLS Best R',best_r)#best_lam
        # print('HOPLS Best Lamda',best_lam)#best_lam
        # # R_HoPLS = 23
        # # lam  = 10
        # # r, Q2,Model = compute_q2_hopls(X_train, Y_train, X_train, Y_train, lam,R_max=R_HoPLS)
        # # #r, Q2,Model = results
        # # print('HOPLS q2 score',Q2[r-1])
        # # y_pred,R,L = Model.predict(X_valid,Y_valid)
        # # plt.plot(Y_valid[:,0])
        # # plt.plot(y_pred[:,0])
        # # plt.show()
        # # print('Process completed')
        #
        #
        #
        #
        #
        #
        #
        #
        #
        #
