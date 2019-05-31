from matplotlib import pyplot as plt
import numpy as np
import librosa
from librosa import display
import cv2
from utils.pathbuilder import PathBuilder
import time

class FramePicker:
    def __init__(self, filename_wav, filename_mp4, duration):
        self.data, self.sample_rate = librosa.load(filename_wav)
        self.cap = cv2.VideoCapture(filename_mp4)
        self.D = np.abs(librosa.stft(self.data))
        #Parameters for analysis
        #aggregate = np.median
        aggregate = 0
        if aggregate!=0:
            self.onset_envelope = librosa.onset.onset_strength(self.data, sr=self.sample_rate,aggregate=aggregate)
        else:
            self.onset_envelope = librosa.onset.onset_strength(self.data, sr= self.sample_rate)
        #Pre/Post max = number of comparison samples before/after the analysis point to compare
        #Pre/Post avg = number of comparison samples before/after the analysis point to take average from
        #Delta = threshold -> lower numbers means more sensitive to noise
        #Wait = number of samples to wait until next analysis point
        self.peaks = librosa.util.peak_pick(self.onset_envelope,
                                            pre_max=3, post_max=3, pre_avg=8,
                                            post_avg=8, delta= 0.1, wait=0)
        #Arrange points into time slots
        self.times = librosa.frames_to_time(np.arange(len(self.onset_envelope)),
                                            sr = self.sample_rate, hop_length = 512)



    def TranscribeNotes(self):
        frames_shown = 0
        frame_rate = 30
        peak_times = (self.times[self.peaks]*frame_rate)
        print(peak_times)
        index = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if(ret):
                if (frames_shown == int(peak_times[index])):
                    cv2.imshow('Frame', frame)
                    #TODO - Note transcription
                    cv2.waitKey(0)
                    index += 1

                frames_shown += 1

    def end(self):
        self.cap.release()
        cv2.destroyAllWindows()



# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

frame_picker = FramePicker(str(PathBuilder().miscdir("Insane.wav")),str(PathBuilder().miscdir("Insane.mp4")), 60 )
print('vai dar bom')

try:
    frame_picker.run()
except KeyboardInterrupt:
    print("nope")
finally:
    frame_picker.end()

exit(0)

