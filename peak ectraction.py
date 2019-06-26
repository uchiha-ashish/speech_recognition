# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 14:29:54 2019

@author: tatras
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
path = "/home/tatras/Downloads/"
import librosa
import numpy as np
import python_speech_features
def data(file):
   signal, sample_rate = librosa.load(path+file+".wav", res_type='kaiser_fast') 
   mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=12).T,axis=0) 
   pre_emphasis = 0.97
   emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
   frame_size = 0.025
   frame_stride = 0.01
   frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  
   signal_length = len(emphasized_signal)
   frame_length = int(round(frame_length))
   frame_step = int(round(frame_step))
   num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  
   pad_signal_length = num_frames * frame_step + frame_length
   z = np.zeros((pad_signal_length - signal_length))
   pad_signal = np.append(emphasized_signal, z) 
   indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
   frames = pad_signal[indices.astype(np.int32, copy=False)]
   frames *= np.hamming(frame_length) 
   m = (frames.max(axis=1))
   silent_frames = np.array(np.where(frames.max(axis=1) < (0.016)))
   frames_final = np.delete(frames, silent_frames[:, :], axis=0)
   signal_final = python_speech_features.sigproc.deframesig(frames_final, len(emphasized_signal), frame_length, frame_step)   
   return sample_rate, signal_final
   
   
sr1, signal1 = data("vocalc#")
sr2, signal2 = data("harmc#")
from dtw import dtw
from numpy.linalg import norm
mfcc1 = np.mean(librosa.feature.mfcc(y=signal1, sr=sr1, n_mfcc=14).T,axis=0) 
mfcc2 = np.mean(librosa.feature.mfcc(y=signal2, sr=sr2, n_mfcc=14).T,axis=0) 
dist, cost, acc_cost, path = dtw(mfcc1, mfcc2, dist = lambda mfcc1, mfcc2: abs(mfcc1 - mfcc2))
dist
plt.imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.xlim((-0.5, cost.shape[0]-0.5))
plt.ylim((-0.5, cost.shape[1]-0.5))



import scipy
import librosa
import numpy as np
import python
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.signal import spectrogram, find_peaks
path = "/home/tatras/Downloads/"
signal1, sr1 = librosa.load(path+"c.wav", res_type='kaiser_fast') 
signal2, sr2 = librosa.load(path+"c#.wav", res_type='kaiser_fast') 

f2, t2, s2 = spectrogram(signal1, fs=sr1)
sxx1 = np.squeeze(sxx1)
plt.plot(sxx1)
signal1 = normalize(signal1.reshape(1,-1))
signal2 = normalize(signal2.reshape(1,-1))


s1, f1, t1, im = plt.specgram(signal1, Fs=sr1)



indexes1, _ = scipy.signal.find_peaks(signal1)
indexes2, _ = scipy.signal.find_peaks(signal2)
plt.plot(signal1.T)

sample = np.sin(2*np.pi*(2**np.linspace(2,10,1000))*np.arange(1000)/48000) + np.random.normal(0, 1, 1000) * 0.15


peaks1, _ = find_peaks(signal1)
peaks2, _ = find_peaks(signal2)
plt.subplot(2, 2, 1)
plt.plot(peaks1, signal1[peaks1], "xr"); plt.plot(signal1); plt.legend(['first'])
plt.subplot(2, 2, 2)
plt.plot(peaks2, signal2[peaks2], "ob"); plt.plot(signal2); plt.legend(['prominence'])

from tqdm import tqdm

from scipy.fftpack import rfft
freq1 = rfft(signal1)
freq2 = rfft(signal2)


peaks1, _ = find_peaks(freq1)
peaks2, _ = find_peaks(freq2)
fig = plt.figure()
ax1 = plt.subplot((2, 2, 1), figsize(6,3))
ax1.plot(peaks1, freq1[peaks1], "xr")
x1.plot(freq1)
ax1.legend(['sig1'])

ax1.margins=(0.005)

plt.subplot(2, 2, 2)
plt.plot(peaks2, freq2[peaks2], "ob"); plt.plot(freq2); plt.legend(['sig2'])



