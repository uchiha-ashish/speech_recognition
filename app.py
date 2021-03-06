# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:56:20 2019

@author: tatras
"""
import pandas as pd
file = pd.DataFrame({"name":["sam", "tom"], "occ":["stud", "stud"], "roll":[1, 2]})
filename = '/home/tatras/Desktop/appdata.csv'
file.to_csv(filename)

import pandas as pd
import csv
from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
import json
from werkzeug import secure_filename
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'SQlite:////home/tatras/Desktop/flask/filestorage.db'

db = SQLAlchemy(app)


class file_contents(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    data = db.Column(db.LargeBinary)


@app.route('/data', methods=['GET'])
def file():
    filename = '/home/tatras/Desktop/appdata.csv'
    data = pd.read_csv(filename, header=0, index_col=0)
    export = data.to_json(orient='records')              
    return export

@app.route('/addentry', methods=['POST'])
def add_data():
    filename = '/home/tatras/Downloads/appdata.csv'    
    data = pd.read_csv(filename, header=0, index_col=0)    
    export = data.to_json(orient='records')    
    #export = eval(export)       
    request_data = request.get_json()    
    request_data = str(request_data)    
    request_data = repr(request_data)    
    parser = json.loads(request_data)    
    parser = eval(parser)    
    data = data.append(parser, ignore_index=True)
    data.to_csv(filename)
 #evan and repr remaining    
    #df = open(filename, 'w')
    #csvwriter = csv.writer(df)
    #count = 0    
    #for i in range(len(parser)):    
     #   for p in parser:       
        #    if count == 0:
                #header = parser[i].keys()
                #csvwriter.writerow(header)                
    #csvwriter.writerow(parser.values())
            #if count == 1:
             #   csvwriter.writerow(parser[i].values())
       # count += 1
    #df.close()
    #data1 = pd.read_csv(filename, header=0, index_col=0)    
    export = data.to_json(orient='records')            
    return export

@app.route('/deleteentry', methods=['DELETE'])
def del_data():
    filename = '/home/tatras/Desktop/appdata.csv'    
    data = pd.read_csv(filename, header=0, index_col=0)    
    export = data.to_json(orient='records') 
    request_data = request.get_json()     
    data = data.drop(request_data)
    data.to_csv(filename)
    export = data.to_json(orient='records')            
    return export

@app.route('/upload',methods = ['POST'])
def upload_file():
    file = request.files['input']
    return file.filename

@app.route('/streamaudio', methods=['POST'])
def get_freq():
    import pyaudio
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import style
    import numpy as np
    import time
    style.use('fivethirtyeight')
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    NOTE_MIN = 40       # C4
    NOTE_MAX = 64       # A4
    FSAMP = 44100       # Sampling frequency in Hz
    FRAME_SIZE = 512   # How many samples per frame?
    FRAMES_PER_FFT = 8 # FFT takes average across how many frames?

######################################################################
# Derived quantities from constants above. Note that as
# SAMPLES_PER_FFT goes up, the frequency step size decreases (so
# resolution increases); however, it will incur more delay to process
# new sounds.

    SAMPLES_PER_FFT = FRAME_SIZE*FRAMES_PER_FFT
    FREQ_STEP = float(FSAMP)/SAMPLES_PER_FFT\
    
    ######################################################################
    # For printing out notes

    NOTE_NAMES = 'C C# D D# E F F# G G# A A# B'.split()

######################################################################
# These three functions are based upon this very useful webpage:
# https://newt.phys.unsw.edu.au/jw/notes.html

    def freq_to_number(f): return 69 + 12*np.log2(f/440.0)
    def number_to_freq(n): return 440 * 2.0**((n-69)/12.0)
    def note_name(n): return NOTE_NAMES[n % 12] + str(n/12 - 1)

######################################################################
# Ok, ready to go now.

# Get min/max index within FFT of notes we care about.
# See docs for numpy.rfftfreq()
    def note_to_fftbin(n): return number_to_freq(n)/FREQ_STEP
    imin = max(0, int(np.floor(note_to_fftbin(NOTE_MIN-1))))
    imax = min(SAMPLES_PER_FFT, int(np.ceil(note_to_fftbin(NOTE_MAX+1))))
    print(imin,imax)
    # Allocate space to run an FFT. 
    buf = np.zeros(SAMPLES_PER_FFT, dtype=np.float32)
    num_frames = 0
    print(buf.shape)
# Initialize audio
    stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=FSAMP,
                                    input=True,
                                    frames_per_buffer=FRAME_SIZE)
#     t1=time.time()
    stream.start_stream()

# Create Hanning window function
    window = 0.5 * (1 - np.cos(np.linspace(0, 2*np.pi, SAMPLES_PER_FFT, False)))

# Print initial text
    print('sampling at', FSAMP, 'Hz with max resolution of', FREQ_STEP, 'Hz')
# print
    frequencies=[] # min freq is set to 20hz 
# As long as we are getting data:
    current_note='a'
    count=0
    start_time=time.time()
    while stream.is_active():
        count+=1
        # Shift the buffer down and new data in
        buf[:-FRAME_SIZE] = buf[FRAME_SIZE:]
        buf[-FRAME_SIZE:] = np.fromstring(stream.read(FRAME_SIZE), np.int16)
        #     print(np.fromstring(stream.read(FRAME_SIZE), np.int16).shape)
        # Run the FFT on the windowed buffer
        fft = np.fft.rfft(buf * window)
        
        # Get frequency of maximum response in range
        freq = (np.abs(fft[imin:imax]).argmax() + imin) * FREQ_STEP

    # Get note number and nearest note
        n = freq_to_number(freq)
        n0 = int(round(n))

    # Console output once we have a full buffer
        num_frames += 1
        frequencies.append([freq,round((time.time()-start_time)*2)/2])
        str_to_write="{},{}\n".format(int(freq),int(time.time()-start_time))
        file=open('/home/tatras/Desktop/flask/freq_time.txt','a')
        file.write(str_to_write)
        file.close()
    return freq
    
    



if __name__ == '__main__':
   app.run(host='0.0.0.0',debug = True, port=8017)
   
   