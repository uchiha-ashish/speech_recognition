import pandas as pd
from scipy.io import wavfile
path = "/home/tatras/Downloads/"
sr1, data1 = wavfile.read(path+"c.wav")
sr2, data2 = wavfile.read(path+"c#.wav")
x = (data1[:,0]+data1[:,1])/2
y = (data2[:,0]+data2[:,1])/2
from dtw import dtw
from numpy.linalg import norm
dist, cost, acc_cost, path = dtw(x, y, dist=lambda x, y: norm(x - y, ord=1))
