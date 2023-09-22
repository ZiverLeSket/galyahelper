from pyaudio import (
    PyAudio,
    paFloat32
)
import numpy as np
from libfifobuffer import ByteFIFO
from myspectrogram import log_mel_spectrogram
import cv2
import subprocess as subp

template = cv2.imread('galka.png', cv2.IMREAD_GRAYSCALE).astype('float32')
template = template/255

threshold = 0.8
color = 255

sample_rate = 16000
samples_per_chunk = 1600

audio = PyAudio()
stream = audio.open(input=True, format=paFloat32, channels=1,
                     rate=sample_rate, frames_per_buffer=samples_per_chunk)

r = ByteFIFO()

for i in range(5):
    r.put(stream.read(samples_per_chunk))

while True:
    r.put(stream.read(samples_per_chunk))
    r.discard(samples_per_chunk*4)
    aboba = log_mel_spectrogram(np.array(np.frombuffer(r.getbuffer(), dtype=np.float32))).numpy()
    res = cv2.matchTemplate(aboba, template, cv2.TM_CCOEFF_NORMED)
    loc = np.count_nonzero(res >= threshold)
    
    if loc == 0:
        continue
    
    print("Sho")