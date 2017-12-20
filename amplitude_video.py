import cv2
import sys
import logging as log
import datetime as dt
import time
from time import sleep
import numpy as np
from scipy.io import wavfile
import pylab as pl
import contextlib
import wave
import time
import pyaudio
import copy
from threading import Thread
import musicanalysis
from musicanalysis import play
from tkinter import * 
import random
import math



cascPath = "haarcascade_frontalface_default.xml"  # for face detection
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

ret = video_capture.set(3, 320)

if ret != 1:
    print ("fail")

ret = video_capture.set(4, 240)

if ret != 1:
    print ("fail")

# amplitude
song = 'sample.wav'
(rate, data) = wavfile.read(song)
amp0 = data[:, 0]
amp1 = data[:, 1]
amp_per_sec = 0
amplitude = []

for i in range(len(amp0)):
    amp_per_sec += ((amp0[i]) ** 2)
    amp_per_sec += ((amp1[i]) ** 2)
    if (i != 0 and i % 500 == 0):
        amp_per_sec = (amp_per_sec / 1000) ** (1/2)
        amplitude.extend([amp_per_sec])
        amp_per_sec = 0
    if (i == len(amp0)):
        amp_per_sec = (amp_per_sec / (i % 1000)) ** (1/2)
        amplitude.extend([amp_per_sec])

# length (time) of wav file
fname = song
with contextlib.closing(wave.open(fname,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)

ANALYSISRATE = 88

#raw data from taps.py
Final_Times_Array = [0.8732759475708008, 1.9280156294504804, 2.88275531133016, 3.88749499320984]

ind = 0

count = 0
success = True

while success:

    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40,40)
    )

    if cv2.waitKey(1) & 0xFF == ord('p'):
        ind = 0
        playthread = Thread(target = play, args = [song], daemon = True)
        playthread.start()
        
    # Draw a rectangle around the faces

    color_range = (max(amplitude) - min(amplitude)) // 255

    img_color = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for (x, y, w, h) in faces:
        if musicanalysis.SR <= 0:
            cv2.circle(frame, (x+w//2, y+h//2), int((4/6)*w), (150, 50, 50), 5)
        else:
            index = round(musicanalysis.playtime * 88)
            # print(musicanalysis.playtime)
            current_amp = amplitude[index] - min(amplitude)

            cv2.circle(frame, (x + w//2 + int(1.5*(amplitude[index] // 100)), y + h//2 + int(1.5*(amplitude[index] // 100))), int(amplitude[index] // 100), (100, 100, amplitude[index] / color_range), 2 + int(amplitude[index] // 500))
            cv2.circle(frame, (x + w//2 + int(1.5*(amplitude[index] // 100)), y + h//2 - int(1.5*(amplitude[index] // 100))), int(amplitude[index] // 100), (100, amplitude[index] / color_range, 100), 2 + int(amplitude[index] // 500))
            cv2.circle(frame, (x + w//2 - int(1.5*(amplitude[index] // 100)), y + h//2 + int(1.5*(amplitude[index] // 100))), int(amplitude[index] // 100), (amplitude[index] / color_range, 100, 100), 2 + int(amplitude[index] // 500))
            cv2.circle(frame, (x + w//2 - int(1.5*(amplitude[index] // 100)), y + h//2 - int(1.5*(amplitude[index] // 100))), int(amplitude[index] // 100), (amplitude[index] / color_range, amplitude[index] / color_range, amplitude[index] / color_range), 2 + int(amplitude[index] // 500))
            rectX = (x + w//2 - int((4/6)*w))
            rectY = (y + h//2 - int((4/6)*w))

            print(ind)
            if ind == len(Final_Times_Array):
                ind = 0
            if musicanalysis.playtime <= Final_Times_Array[ind]:
                cv2.rectangle(frame, (0,0), (int(70*(Final_Times_Array[ind] - musicanalysis.playtime)), int(70*(Final_Times_Array[ind] - musicanalysis.playtime))), (255, 255, 255), 10)
            else: 
                if ind == len(Final_Times_Array):
                    ind = 0 
                else:
                    ind += 1
        
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()









