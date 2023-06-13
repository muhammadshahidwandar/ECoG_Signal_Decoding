# ECoG class contains signal and motion data and performs preprocessing procedure

import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import butter, sosfilt
from scipy.interpolate import interp1d
import pywt
import scipy.io
import os

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a
def Compute_Spectrogram(data):
    spectrs = []
    for i in range(10): #data.shape[1]
        frequencies, times, spectrogram = scipy.signal.spectrogram(data[:, i], 100,nperseg=100,noverlap=99,nfft = 128,mode='magnitude')
        #frequencies, times, spectrogram = scipy.signal.spectrogram(data[28000:30000:25, i], 100,nperseg=100,noverlap=99,nfft = 128,mode='magnitude')
        #frequencies, times, spectrogram = scipy.signal.spectrogram(data, 100,nperseg=100,noverlap=75,nfft = 128,mode='magnitude')
        spectrs.append(spectrogram.T)
    spectrs = np.array(spectrs)
    result = np.transpose(spectrs, (1, 0, 2))
    #frequencies, times, spectrogram = scipy.signal.spectrogram(data.signal[25000:65000:100, 0], 100,nperseg=100,noverlap=99,nfft = 128,mode='magnitude')
    return result
def read_ECoG_from_csv(signal_file_path, motion_file_path):
    signal_df = pd.read_csv(signal_file_path)
    motion_df = pd.read_csv(motion_file_path)
    left_shoulder = motion_df.loc[0,motion_df.columns.str.contains('LSH')].values
    right_shoulder = motion_df.loc[0,motion_df.columns.str.contains('RSH')].values
    body_center = (left_shoulder + right_shoulder)/2 #It is a static point in this experiment
    motion_left_hand = motion_df[motion_df.columns[motion_df.columns.str.contains('Motion')].
                                 append((motion_df.columns[motion_df.columns.str.contains('LHND')]))].values #LHND  LWRI
    motion_left_wrist = motion_df[motion_df.columns[motion_df.columns.str.contains('Motion')].
                                 append((motion_df.columns[motion_df.columns.str.contains('LWRI')]))].values #LHND  LWRI
    motion_left_hand[:,1:] -= body_center #centering motion
    #motion_left_wrist[:,1:] -= body_center
    #motion_data = np.hstack((motion_left_hand, motion_left_wrist[:,1:]))
    #motion_data = np.hstack((motion_data, right_hand))
    return signal_df.values, motion_left_hand#motion_data#motion_left_hand,motion_left_wrist
def load_ECoG64(load_dir):
    X = {}
    for i in range(1,65):
        filename = load_dir + "/ECoG_ch" + str(i) +".mat"
        mat = scipy.io.loadmat(filename)
        X[i] = mat["ECoGData_ch" + str(i)]
    motion_file = load_dir + "/Motion.mat"
    motion_mat = scipy.io.loadmat(motion_file)
    left_shoulder = motion_mat["MotionData"][0][0]
    #left_wrist = motion_mat["MotionData"][2][0]
    right_shoulder = motion_mat["MotionData"][3][0]
    body_center = (left_shoulder + right_shoulder)/2
    left_hand = motion_mat["MotionData"][2][0] - body_center
    right_hand = motion_mat["MotionData"][5][0] - body_center
    motion_time = motion_mat['MotionTime']
    time = scipy.io.loadmat(load_dir + "/ECoG_time.mat")["ECoGTime"]
    X = np.vstack([X[k] for k in X.keys()])
    signal_data = np.vstack((time,X))
    motion_data = np.hstack((motion_time.T, left_hand))
    #motion_data = np.hstack((motion_data, right_hand))
    #motion_data = np.dstack((right_shoulder, right_hand)) # [time, axis, marker]
    #Spectrogram_Compute(signal_data)
    return signal_data.T, motion_data
def abs_morlet(M,w = 0.5,s = 0.1):
    return np.abs(signal.morlet(M,w = 0.5,s = 0.1))

class ECoG(object):
    def __init__(self,signal_data,motion_data,downsample = True):
        start = max(signal_data[0,0],motion_data[0,0])
        end = min(signal_data[-1,0],motion_data[-1,0])
        if signal_data.shape[1] == 33:
            self.centers = np.array([[187,168],[190,126],[225,180],[229,141],[264,227],[264,193],[266,152],[282,215],[303,236],
                  [303,198],[306,155],[325,221],[325,178],[323,136],[343,246],[349,205],[347,162],[340,120],[365,230],
                 [365,185],[362,143],[386,212],[385,164],[380,122],[400,233],[405,190],[405,147],[420,214],[424,165],[424,125],
                  [455,165],[460,125]],dtype = 'float')
        else:
            self.centers = np.array([[205,85],[275,85],[170,120],[240,120], [310,120], [135,160],[205,160],[275,160],
                      [100,210],[170,210],[240,210],[310,210],
                      [140,255],[210,255],[280,255],[350,255],
                      [100,295],[170,295],[240,295],[310,295],
                      [140,340],[210,340],[280,340],[340,330],
                      [100,380],[170,380],[240,380],[310,380],
                      [140,420],[210,420],[280,420],[340,420],
                      [100,450],[170,460],[240,460],[310,460],
                      [140,500],[210,500],[280,500],[340,490],
                      [100,530],[170,540],[240,540],[310,540],
                      [140,580],[210,580],[280,580],[340,580],
                      [100,620],[170,620],[240,620],[310,620],
                      [140,660],[210,660],[280,660],[340,660],
                      [100,700],[170,700],[240,700],[310,700],
                      [140,740],[210,750],[280,750],[340,750]], dtype='float')
        #cutting signal and motion, only overlapping time left
        signal_data = signal_data[:,:][(signal_data[:,0]>=start)]
        signal_data = signal_data[:,:][(signal_data[:,0]<=end)]
        motion_data = motion_data[:,:][motion_data[:,0]>= start] 
        motion_data = motion_data[:,:][motion_data[:,0]<= end]
        M = []
        #signal and motion have different time stamps, we ned to synchronise them
        #interpolating motion and calculating arm position in moments of "signal time"
        for i in range(1,motion_data.shape[1]):
            interpol = interp1d(motion_data[:,0],motion_data[:,i],kind="cubic",fill_value="extrapolate")
            x = interpol(signal_data[:,0])
            M.append(x)
        #downsampling in 10 times to get faster calcultions
        self.downsample = downsample
        if downsample:
            self.signal = signal_data[::10,1:]
            self.motion = np.array(M).T[::10,:]
            self.time = signal_data[::10,0]
        else:
            self.signal = signal_data[:,1:]
            self.motion = np.array(M).T[:,:]
            self.time = signal_data[:,0]
    ############
            
    #signal filtering (not sure that it works correctly)
    def bandpass_filter(self, lowcut, highcut,inplace = False, fs = 100, order=7):
        nyq =  fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = signal.butter(order,  (low, high), btype='band',analog=False,output='sos')
        filtered_signal = np.array([sosfilt(sos, self.signal[:,i]) for i in range(self.signal.shape[1])])
        if inplace:
            self.signal = filtered_signal.T
        return filtered_signal.T

    #Generating a scalogram by wavelet transformation 
    def scalo(self, window, freqs,start,end, step = 100,lib='pywt'): #window in sec,freqs in Hz, step in ms
        div = 1
        X = self.signal[start:end,:]
        if self.downsample:
            div = 10
        window_len = int(((window * 1000 // step) + 2) * step//div)
        scalo = np.empty((X.shape[0]-window_len,X.shape[1],freqs.shape[0],(window * 1000 // step) + 2))
        for i in range(X.shape[1]):
            for j in range(window_len,X.shape[0]):
                if lib == 'scipy':
                    scalo[j-window_len,i,:,:] = signal.cwt(data = X[j-window_len:j,i],
                                                       wavelet=abs_morlet,widths = freqs)[:,::step//div] **2
                if lib == 'pywt':
                    #print(type(pywt.cwt(data = X[j-window_len:j,i],wavelet='morl',scales = freqs)[0]))
                    scalo[j-window_len,i,:,:] = pywt.cwt(data = X[j-window_len:j,i],
                                                       wavelet='morl',scales = freqs)[0][:,::step//div] **2
        return scalo, self.motion[start+window_len:end,:], self.time[start+window_len:end]
    
def matrix_gaussian(centers, args): #x is 32x2 
    mu = args[:,:2] #Nx2
    c = np.repeat(centers,args.shape[0],axis=1).reshape(centers.shape[0],centers.shape[1],args.shape[0]).transpose(2,0,1)
    
    mu = np.expand_dims(mu,1)
    x_avg = c - mu
    #print(x_avg.shape)
    Sigmas = np.empty((args.shape[0],2,2))
    Sigmas[:,0,0] = args[:,2]
    Sigmas[:,0,1] = args[:,3]
    Sigmas[:,1,1] = args[:,4]
    Sigmas[:,1,0] = args[:,3]
    A = np.expand_dims(args[:,5],1)
    
    Inv = np.linalg.inv(Sigmas)
    #print(Inv.shape)
    #print((Inv @ x_avg.transpose(0,2,1)).shape)
    val = np.einsum('jii->ji',(x_avg @ Inv @ x_avg.transpose(0,2,1)))
    return  (A * np.exp( -0.5 * val)).T
def matrix_find_best_A(args,signal,centers):
    #def func(A):
    pred = matrix_gaussian(centers,args)#np.array([matrix_gaussian(centers[i],args) for i in range(32)])
    # return np.sqrt(np.sum(((signal - A * pred)) ** 2))
    #a_best = scipy.optimize.minimize(func,0.1)
    return np.sum(signal,axis=1)/np.sum(pred,axis=0)
def matrix_normal_features(signal,centers): #signal is Nx32, centers 32x2
    c = centers.copy()
    M = (signal @ centers)/np.expand_dims(signal.sum(axis=1),1)
    w_sum = np.sum(signal,axis=1)
    fact = np.expand_dims(np.expand_dims(w_sum - 1 * np.sum(signal ** 2,axis=1)/w_sum, 1),2)

    S = np.array([np.diag(signal[i]) for i in range(signal.shape[0])])/fact

    centers = np.repeat(centers,M.shape[0],axis=1).reshape(centers.shape[0],centers.shape[1],M.shape[0]).transpose(2,0,1)
    centers -= np.expand_dims(M,1)
    Sigma = centers.transpose(0,2,1) @ S @ centers
    
    Sigma = np.expand_dims(Sigma,3)
    #print((w_sum - 1 * np.sum(signal ** 2,axis=1)/w_sum))
    #print(np.expand_dims(M,1).shape, Sigma.shape)
    args = np.hstack([M,Sigma[:,0,0],Sigma[:,0,1],Sigma[:,1,1],np.ones((signal.shape[0],1))])
    A = matrix_find_best_A(args,signal,c)
    args[:,-1] = A
    return args

def spectral_lm(scalo,centers,config):
    # the shape of sclogram is N_samples * N_electrodes * N_freqs * N_timestamps
    res = np.zeros((scalo.shape[0],6,scalo.shape[2],scalo.shape[3]))
    for j in range(scalo.shape[2]):
        for k in range(scalo.shape[3]):
            #if config["spectral_lm_kind"] == "Normal":
           res[:,:,j,k] = matrix_normal_features(scalo[:,:,j,k],centers) 
            #if config["spectral_lm_kind"] == "rbf":
             #   local_centers = np.array([[160,200],[165,240],[175,290],[220,305],[150,330],[210,370],[120,360], [165,400], [150,440], [180,350] ])
              #  weight_matrix = RBF(length_scale=100.)(centers,local_centers)
               # res[:,:,j,k] = np.dot(scalo[:,:,j,k],weight_matrix)              
    return res
