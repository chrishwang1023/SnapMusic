from scipy.io import wavfile
import numpy as np
import pylab as pl
import contextlib
import wave
import pyaudio

song = 'sample.wav'

(rate, data) = wavfile.read(song)
amp0 = data[:, 0]
amp1 = data[:, 1]
amp_per_sec = 0
amplitude = []

for i in range(len(amp0)):
    if (i == 0 or i % 500 != 0):
        amp_per_sec += ((amp0[i]) ** 2)
        amp_per_sec += ((amp1[i]) ** 2)
    if (i % 500 == 0):
        amp_per_sec = (amp_per_sec / 1000) ** (1/2)
        amplitude += [amp_per_sec]
        amp_per_sec = 0
    if (i == len(amp0)):
        amp_per_sec = (amp_per_sec / (i % 1000)) ** (1/2)
        amplitude += [amp_per_sec]

#get duration of wave file, length of song 
fname = 'sample.wav'
with contextlib.closing(wave.open(fname,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)

# play the song
CHUNK = 1024  
framecount = 0
LATENCY = 0.023219954648526078
SR = 0
playtime = 0

def play(song):
    global framecount, CHUNK, LATENCY, SR, playtime

    f = wave.open(song,"rb")  

    p = pyaudio.PyAudio()  
    SR = f.getframerate()

    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                    channels = f.getnchannels(),  
                    rate = SR,  
                    output = True)  

    data = f.readframes(CHUNK)  
     
    while data:  
        stream.write(data)
        framecount += CHUNK
        playtime = (framecount / SR - LATENCY)
        data = f.readframes(CHUNK)

    #stop
    SR = 0
    playtime = 0
    framecount = 0
    stream.stop_stream()  
    stream.close()  
    #close
    p.terminate()

