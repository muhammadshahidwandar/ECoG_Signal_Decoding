import numpy as np
from matplotlib import pyplot as plt
import scipy
import ECoG
import pywt
import time
import pandas as pd
import math
import os

#########hoplsfunctions
import warnings
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from scipy.io import loadmat, savemat
from joblib import Parallel, delayed
from hopls import matricize, qsquared, HOPLS
import torch
Feet = 1 # 0 mean left 1 mean right
def compute_q2_pls(tdata, tlabel, vdata, vlabel, Rval):
    test = PLSRegression(n_components=Rval)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test.fit(matricize(tdata), matricize(tlabel))
    Y_pred = test.predict(matricize(vdata))
    Q2 = qsquared(matricize(vlabel), matricize(Y_pred))
    print('pls score',Q2)
    return Q2,Y_pred


def compute_q2_hopls(tdata, tlabel, vdata, vlabel, la, R_max=20):
    Ln = [la] * (len(tdata.shape) - 1)
    if len(tlabel.shape) > 2:
        Km = [la] * (len(tlabel.shape) - 1)
    else:
        Km = None
    test = HOPLS(R_max, Ln, Km)
    test.fit(tdata, tlabel)
    predict, r, Q2 = test.predict(vdata, vlabel)
    score = test.score(vdata, vlabel)
    Q2 = qsquared(matricize(vlabel), matricize(predict))
    print('hopls score',Q2)
    #print('score',score)
    return r, Q2, predict


def do_testing(X, Y, lambda_max=10, R_max=20):
    PATH = "./results/"
    resname = PATH + f"HOPLS_results_complexX.mat"
    # results/HOPLS_results_complexX3_2_ss10.mat
    n_folds = 2
    cv = KFold(n_folds)
    fold = 0
    PLS_r = []
    PLS_q2 = []
    HOPLS_l = []
    HOPLS_r = []
    HOPLS_q2 = []
    NPLS_r = []
    HOPLS_train_q2 = []
    PLS_train_q2 = []
    NPLS_train_q2 = []
    NPLS_q2 = []
    PLS_hyper = np.zeros((n_folds, R_max))
    HOPLS_hyper = np.zeros((n_folds, lambda_max - 1, R_max))
    NPLS_hyper = np.zeros((n_folds, R_max))
    for train_idx, valid_idx in cv.split(X, Y):
        X_train = torch.Tensor(X[train_idx])
        Y_train = torch.Tensor(Y[train_idx])
        X_valid = torch.Tensor(X[valid_idx])
        Y_valid = torch.Tensor(Y[valid_idx])

        results = []
        for R in range(1, R_max + 1):
            results.append(compute_q2_pls(X_train, Y_train, X_valid, Y_valid, R))
        old_Q2 = -np.inf
        PLS_hyper[fold] = results
        for i in range(len(results)):
            Q2 = results[i]
            if Q2 > old_Q2:
                best_r = i + 1
                old_Q2 = Q2
        PLS_q2s = compute_q2_pls(X_train, Y_train, X_train, Y_train, best_r)
        PLS_train_q2.append(PLS_q2s)
        PLS_r.append(best_r)
        PLS_q2.append(old_Q2)

        results = []
        for lam in range(1, lambda_max):
            results.append(
                compute_q2_hopls(X_train, Y_train, X_valid, Y_valid, lam, R_max)
            )
        old_Q2 = -np.inf
        NPLS_hyper[fold] = results[0][1]
        for i in range(1, len(results)):
            r, Q2s,_ = results[i]
            HOPLS_hyper[fold, i - 1] = Q2s
            Q2 = Q2s[r - 1]
            if Q2 > old_Q2:
                best_lam = i + 1
                best_r = r
                old_Q2 = Q2
        _, HOPLS_q2s, _ = compute_q2_hopls(
            X_train, Y_train, X_train, Y_train, best_lam, best_r
        )
        _, NPLS_q2s, _ = compute_q2_hopls(
            X_train, Y_train, X_valid, Y_valid, best_lam, best_r
        )
        HOPLS_train_q2.append(HOPLS_q2s[-1])
        NPLS_train_q2.append(NPLS_q2s[-1])
        HOPLS_l.append(best_lam)
        HOPLS_r.append(best_r)
        HOPLS_q2.append(old_Q2)
        best_npls_r = results[0][0]
        NPLS_r.append(best_npls_r)
        NPLS_q2.append(results[0][1][best_npls_r - 1])
        fold += 1
    results = {
        "PLS_R": PLS_r,
        "PLS_Q2": PLS_q2,
        "PLS_train": PLS_train_q2,
        "PLS_hyp": PLS_hyper,
        "HOPLS_R": HOPLS_r,
        "HOPLS_L": HOPLS_l,
        "HOPLS_Q2": HOPLS_q2,
        "HOPLS_train": HOPLS_train_q2,
        "HOPLS_hyp": HOPLS_hyper,
        "NPLS_R": NPLS_r,
        "NPLS_Q2": NPLS_q2,
        "NPLS_train": NPLS_train_q2,
        "NPLS_hyp": NPLS_hyper,
    }
    savemat(resname, results)


##### train and test
def do_trainTest(X, Y, X_v, Y_v, lambda_max=20, R_max=20):
    PATH = "./results/"
    resname = PATH + f"HOPLS_results_complexX.mat"
    # results/HOPLS_results_complexX3_2_ss10.mat
    # for train_idx, valid_idx in cv.split(X, Y):
    X_train = torch.Tensor(X)
    Y_train = torch.Tensor(Y)
    X_valid = torch.Tensor(X_v)
    Y_valid = torch.Tensor(Y_v)

    results = []
    for R in range(1, R_max + 1):
        results.append(compute_q2_pls(X_train, Y_train, X_valid, Y_valid, R))
    old_Q2 = -np.inf
    for i in range(len(results)):
        Q2 = results[i]
        if Q2 > old_Q2:
            best_r = i + 1
            old_Q2 = Q2
    PLS_q2s = compute_q2_pls(X_train, Y_train, X_train, Y_train, best_r)
    results = []
    for lam in range(1, lambda_max):
        results.append(
            compute_q2_hopls(X_train, Y_train, X_valid, Y_valid, lam, R_max)
        )
    old_Q2 = -np.inf
    for i in range(1, len(results)):
        r, Q2s = results[i]
        Q2 = Q2s[r - 1]
        if Q2 > old_Q2:
            best_lam = i + 1
            best_r = r
            old_Q2 = Q2
    _, HOPLS_q2s, _ = compute_q2_hopls(
        X_train, Y_train, X_train, Y_train, best_lam, best_r
    )
    _, NPLS_q2s, _ = compute_q2_hopls(
        X_train, Y_train, X_valid, Y_valid, best_lam, best_r
    )


def Hopls_BestParamtr(X, Y, X_v, Y_v, lambda_max=10, R_max=30):
    X_train = torch.Tensor(X)
    Y_train = torch.Tensor(Y)
    X_valid = torch.Tensor(X_v)
    Y_valid = torch.Tensor(Y_v)

    results = []
    for lam in range(1, lambda_max):
        results.append(
            compute_q2_hopls(X_train, Y_train, X_valid, Y_valid, lam, R_max)
        )
    old_Q2 = -np.inf
    for i in range(1, len(results)):
        r, Q2s, predctd = results[i]
        Q2 = Q2s[r - 1]
        if Q2 > old_Q2:
            best_lam = i + 1
            best_r = r
            old_Q2 = Q2
    _, HOPLS_q2s, _ = compute_q2_hopls(
        X_train, Y_train, X_train, Y_train, best_lam, best_r
    )
    _, NPLS_q2s, _ = compute_q2_hopls(
        X_train, Y_train, X_valid, Y_valid, best_lam, best_r
    )
    return best_r, best_lam

###################### ECOG File Loader
def ECOG_Spectogrm(FilePath):
    x,y = ECoG.load_ECoG64(FilePath)#ECoG.read_ECoG_from_csv("ECoG.csv", "Motion.csv")
    print('data loaded successfully')
    data = ECoG.ECoG(x, y, downsample = True) #False
    X = data.signal
    y = data.motion
    t = data.time
    X_spec = ECoG.Compute_Spectrogram(X)
    Y = y#.reshape(-1,2,3)
    print('data is ready')
    Y = Y[0:X_spec.shape[0],:]
    return X_spec, Y
