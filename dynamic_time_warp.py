#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:30:33 2019

@author: uchiha_ashish
"""

import numpy as np
import scipy.io.wavfile
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.fftpack import dct
sr1, y1 = scipy.io.wavfile.read('/Users/uchiha_ashish/Downloads/AUD-20190525-WA0005.wav')
sr2, y2 = scipy.io.wavfile.read('/Users/uchiha_ashish/Downloads/AUD-20190525-WA0006.wav') 
from python_speech_features import mfcc 
mfcc1 = mfcc(y1, samplerate = sr1)
mfcc2 = mfcc(y2, samplerate = sr2)
from dtw import accelerated_dtw
from dtw import dtw
from sklearn.metrics.pairwise import euclidean_distances
from numpy.linalg import norm
dist_val = euclidean_distances(mfcc1, mfcc2)
dist, cost, acc_cost, path = accelerated_dtw(mfcc1, mfcc2, dist=lambda x, y: norm(x - y, ord=1))
print(dist)
plt.imshow(acc_cost.T, origin='lower', cmap='gray', interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.show()

path2, sim = dtw_path(mfcc1, mfcc2)

np.random.seed(0)
n_ts, sz, d = 2, 2226, 1
matrix_path = np.zeros((2060, 2226), dtype=np.int)
for i, j in path2:
    matrix_path[i, j] = 1
"""
plt.figure()
plt.subplot2grid((1, 3), (0, 0), colspan=2)
plt.plot(np.arange(2226), mfcc2)
plt.plot(np.arange(2060), mfcc1)
plt.subplot(1, 3, 3)
plt.imshow(matrix_path, cmap="gray")

plt.tight_layout()
plt.show()
"""
from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np
# Load the data and calculate the time of each sample
times1 = np.arange(len(y1))/float(sr1)
# Make the plot
# You can tweak the figsize (width, height) in inches
plt.figure(figsize=(30, 4))
plt.fill_between(times1, y1[:,0], y1[:,1], color='b') 
plt.xlim(times1[0], times1[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
# You can set the format by changing the extension
# like .pdf, .svg, .eps
plt.show()


from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np
# Load the data and calculate the time of each sample
times = np.arange(len(y2))/float(sr2)
# Make the plot
# You can tweak the figsize (width, height) in inches
plt.figure(figsize=(30, 4))
plt.fill_between(times, y2[:,0], y2[:,1], color='r') 
plt.xlim(times[0], times[-1])
plt.xlabel('time (s)')
plt.ylabel('amplitude')
# You can set the format by changing the extension
# like .pdf, .svg, .eps
plt.show()

fig = plt.figure(figsize=(30, 4))
fig.add_subplot(224)
plt.plot(times1[:], y1[:], color='b')
fig.add_subplot(224)
plt.plot(times[:], y2[:], color='r')
plt.xlabel('time')
plt.ylabel('x')
plt.show()

from scipy import signal
import matplotlib.pyplot as plt
corr = signal.correlate(y1, y2, mode='same')
clock = np.arange(0, 12, 1)
fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, sharex=True)
ax_orig.plot(y1)
ax_orig.plot(clock, y1[clock], 'b')
ax_orig.set_title('005')
ax_noise.plot(y2, 'r')
ax_noise.set_title('006')
ax_corr.plot(corr)
ax_corr.plot(clock, corr[clock], 'y')
ax_corr.axhline(0.5, ls=':')
ax_corr.set_title('Cross-correlated')
fig.show()

