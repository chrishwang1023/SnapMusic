import time
import random
import wave
import pyaudio
import cv2
from tkinter import *
from threading import Thread
from tapping_play import tap_play

class Attributes():
    def __init__(self):
        self.song = 'sample.wav'
        self.Times = []
        self.Start_Interval = 0
        self.avg = 0
        self.Fin_Times = []
        self.Final_Times_Array = []
        self.last_count = 0
attributes = Attributes()


class tapping():
    def __init__(self):
        self.j=0
        self.mgui=Tk()
        self.playthread = Thread(target = tap_play, args = [attributes.song], daemon = True)
        self.st = Button(self.mgui, text="Beat", command = self.PrintNumber)
        self.st.pack()
        self.label = Label(self.mgui, text=str(self.j))
        self.label.pack()  
        self.mgui.mainloop()

    def PrintNumber(self):
        # global Times
        self.j+=1
        if self.j == 1:
            self.playthread.start()
        self.label.config(text=str(self.j))
        t = time.time()
        attributes.Times.append(t)
        return

def BPM():
    if __name__ == "__main__":
        tapping()
        size = len(attributes.Times)
        if size > 1:
            for i in range(size-1):
                if i == 0:
                    attributes.Start_Interval = attributes.Times[i + 1] - attributes.Times[i]
                    attributes.Fin_Times.append(attributes.Start_Interval)
                else:
                    attributes.avg += attributes.Times[i + 1] - attributes.Times[i]
                    if i != 1 and i % 5 == 0:
                        attributes.avg = attributes.avg / 5
                        attributes.Fin_Times.append(attributes.avg)
                        attributes.avg = 0
                    elif i == size-2:
                        attributes.last_count = (i + 5) % 5
                        if attributes.last_count == 0:
                            attributes.avg = attributes.avg / 5
                            attributes.Fin_Times.append(attributes.avg)
                        else: 
                            attributes.avg = attributes.avg / (attributes.last_count)
                            attributes.Fin_Times.append(attributes.avg)
        for i in range(len(attributes.Fin_Times)):
            if i == 0:
                attributes.Final_Times_Array.append(attributes.Fin_Times[i])
            else: 
                for i in range(size-2):
                    attributes.Final_Times_Array.append(attributes.Final_Times_Array[i] + attributes.Fin_Times[size//5 * (i//5 + 1)])
        print(attributes.Fin_Times)

BPM()
print(attributes.Final_Times_Array)

