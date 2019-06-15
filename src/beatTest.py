from matplotlib import pyplot as plt
import numpy as np
import librosa
from librosa import display
import cv2
from utils.pathbuilder import PathBuilder
import time





# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

pv = PianoVision(str(PathBuilder().miscdir("Solfeggietto.wav")),str(PathBuilder().miscdir("Solfeggietto.mp4")), 60 )
print('vai dar bom')

try:
    pv.TranscribeNotes()
except KeyboardInterrupt:
    print("nope")
finally:
    pv.end()

exit(0)

