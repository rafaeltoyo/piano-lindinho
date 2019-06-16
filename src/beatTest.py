from matplotlib import pyplot as plt
import numpy as np
import librosa
from librosa import display
import cv2

from pianovision.Keyboard import Keyboard
from utils.pathbuilder import PathBuilder
import time


class PianoVision:
    def __init__(self, filename_wav, filename_mp4, duration):
        self.data, self.sample_rate = librosa.load(filename_wav)
        self.cap = cv2.VideoCapture(filename_mp4)
        self.D = np.abs(librosa.stft(self.data))
        self.last_frame = None
        self.current_frame = None

        # Parameters for analysis
        # aggregate = np.median
        aggregate = 0
        if aggregate != 0:
            self.onset_envelope = librosa.onset.onset_strength(self.data, sr=self.sample_rate, aggregate=aggregate)
        else:
            self.onset_envelope = librosa.onset.onset_strength(self.data, sr=self.sample_rate)
        # Pre/Post max = number of comparison samples before/after the analysis point to compare
        # Pre/Post avg = number of comparison samples before/after the analysis point to take average from
        # Delta = threshold -> lower numbers means more sensitive to noise
        # Wait = number of samples to wait until next analysis point
        self.peaks = librosa.util.peak_pick(self.onset_envelope,
                                            pre_max=3, post_max=3, pre_avg=10,
                                            post_avg=10, delta=0.1, wait=0)
        # Arrange points into time slots
        self.times = librosa.frames_to_time(np.arange(len(self.onset_envelope)),
                                            sr=self.sample_rate, hop_length=512)
        D = np.abs(librosa.stft(self.data))
        plt.figure(figsize=(13, 7))
        ax = plt.subplot(2, 1, 1)
        librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                                 y_axis='log', x_axis='time')
        plt.subplot(2, 1, 2, sharex=ax)
        plt.plot(self.times, self.onset_envelope, alpha=0.8, label='Onset strength')
        plt.vlines(self.times[self.peaks], 0,
                   self.onset_envelope.max(), color='r', alpha=0.8,
                   label='Selected peaks', linestyle='--')
        plt.legend(frameon=True, framealpha=0.8)
        plt.axis('tight')
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.show()


def first(self):
    image = self.frame
    print(image.shape)
    print((cv2.imread('piano1.png')).shape)
    cv2.imshow('yep', image)
    cv2.waitKey(0)
    (h, w, c) = image.shape

    # Marcar corte
    hi, hf = 548, 804
    hi, hf = 300, 400
    cropped_preview = image.copy()
    cv2.line(cropped_preview, (0, hi), (w - 1, hi), (0, 0, 255), 3)
    cv2.line(cropped_preview, (0, hf), (w - 1, hf), (0, 0, 255), 3)

    cropped = image[hi:hf, :, :]

    test = (255 - cropped).prod(axis=2).astype('float32')
    test = ((test * 255) / test.max()).astype('uint8')

    ret, thresh = cv2.threshold(test, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_size = int(cropped.shape[1] * 0.005)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

    maxsize = int(np.sqrt(cropped.shape[0] ** 2 + cropped.shape[1] ** 2))

    hough = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    h_lines = cv2.HoughLines(edges, 1, np.pi / 4, int(cropped.shape[1] * 0.1))
    if h_lines is not None:
        for line in h_lines:
            rho, theta = line[0]

            if not (np.pi * 3 / 4 >= (theta % np.pi) >= np.pi * 1 / 4):
                continue

            a = np.cos(theta)
            b = np.sin(theta)

            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + maxsize * (-b))
            y1 = int(y0 + maxsize * (a))
            x2 = int(x0 - maxsize * (-b))
            y2 = int(y0 - maxsize * (a))

            keyHeight = y2
            cv2.line(hough, (x1, y1), (x2, y2), (0, 0, 255), 3)

    v_lines = cv2.HoughLines(edges, 1, np.pi / 32, int(cropped.shape[0] * 0.25))
    x_coordinates = []
    i = 0
    if v_lines is not None:
        for line in v_lines:
            rho, theta = line[0]
            if (np.pi * 3 / 4 >= (theta % np.pi) >= np.pi * 1 / 4):
                continue
            if (i % 2 == 0):
                color = (0, 0, 255)
            if (i % 2 == 1):
                color = (0, 255, 0)
            i = i + 1

            a = np.cos(theta)
            b = np.sin(theta)

            x0 = a * rho
            y0 = b * rho

            x1 = int(x0 + maxsize * (-b))
            y1 = int(0)
            x2 = int(x0 - maxsize * (-b))
            y2 = int(keyHeight)

            x_coordinates.append(x1)
            cv2.line(hough, (x1, y1), (x2, y2), (255, 0, 0), 5)

    mask = np.zeros(cropped.shape)

    keys = []
    x_coordinates.sort()
    key_shift = 9
    # TODO - Calculating KeyShift
    keyIndex = key_shift
    white_key_x = []
    for i in range(len(x_coordinates)):
        if (
                keyIndex == 0 or keyIndex == 2 or keyIndex == 4 or keyIndex == 6 or keyIndex == 8 or keyIndex == 3 or keyIndex == 9):
            if (i < len(x_coordinates) - 1):
                midway = int(x_coordinates[i] + (x_coordinates[i + 1] - x_coordinates[i]) / 2)
                white_key_x.append(midway)
                hough = cv2.line(hough, (midway, 0), (midway, cropped.shape[0]), (255, 255, 255), 4)
        keyIndex = (keyIndex + 1) % 10

    lastX = 0
    index = 0
    colors = [(255, 255, 0), (0, 255, 255)]
    color = colors[0]
    for x in white_key_x:
        mask = cv2.rectangle(mask, (lastX, 0), (x, cropped.shape[0]), color, cv2.FILLED)
        lastX = x
        index = (index + 1) % 2
        color = colors[index]

    keyIndex = key_shift
    for i in range(len(x_coordinates) - 1):
        if (keyIndex == 0 or keyIndex == 2 or keyIndex == 4 or keyIndex == 6 or keyIndex == 8):
            if (i < len(x_coordinates) - 1):
                mask = cv2.rectangle(mask, (x_coordinates[i], 0), (x_coordinates[i + 1], keyHeight), (0, 0, 0),
                                     cv2.FILLED)
        keyIndex = (keyIndex + 1) % 10

    cv2.imshow('First frame', mask)
    cv2.waitKey(0)


def TranscribeNotes(self):
    frames_shown = 0
    frame_rate = 30
    peak_times = (self.times[self.peaks] * frame_rate)
    print(peak_times)
    index = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while self.cap.isOpened():
        ret, frame = self.cap.read()
        if (ret):
            self.frame = frame.copy()

            if self.last_frame is None:
                self.first()

            if (frames_shown == int(peak_times[index])):
                # TODO - Note transcription
                index += 1
            cv2.putText(frame, str(index), (0, 200), font, 4, (255, 255, 255), 5, cv2.LINE_AA)
            cv2.imshow('Frame', frame)

            frames_shown += 1
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

            self.last_frame = self.frame
        else:
            break


def end(self):
    self.cap.release()
    cv2.destroyAllWindows()

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

# pv = PianoVision(str(PathBuilder().miscdir("Solfeggietto.wav")),str(PathBuilder().miscdir("Solfeggietto.mp4")), 60 )
# print('vai dar bom')
#
# try:
#     pv.TranscribeNotes()
# except KeyboardInterrupt:
#     print("nope")
# finally:
#     pv.end()
#
# exit(0)
#
image = cv2.imread('piano1.png')
(h, w, c) = image.shape

# Marcar corte
hi, hf = 548, 804
hi, hf = 535, 760
cropped_preview = image.copy()
cv2.line(cropped_preview, (0, hi), (w - 1, hi), (0, 0, 255), 3)
cv2.line(cropped_preview, (0, hf), (w - 1, hf), (0, 0, 255), 3)

cropped = image[hi:hf, :, :]

test = (255 - cropped).prod(axis=2).astype('float32')
test = ((test * 255) / test.max()).astype('uint8')

ret, thresh = cv2.threshold(test, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel_size = int(cropped.shape[1] * 0.005)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

maxsize = int(np.sqrt(cropped.shape[0] ** 2 + cropped.shape[1] ** 2))

hough = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

h_lines = cv2.HoughLines(edges, 1, np.pi / 4, int(cropped.shape[1] * 0.1))
if h_lines is not None:
    for line in h_lines:
        rho, theta = line[0]

        if not (np.pi * 3 / 4 >= (theta % np.pi) >= np.pi * 1 / 4):
            continue

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + maxsize * (-b))
        y1 = int(y0 + maxsize * (a))
        x2 = int(x0 - maxsize * (-b))
        y2 = int(y0 - maxsize * (a))

        keyHeight = y2
        cv2.line(hough, (x1, y1), (x2, y2), (0, 0, 255), 3)

v_lines = cv2.HoughLines(edges, 1, np.pi / 32, int(cropped.shape[0] * 0.25))
x_coordinates = []
i = 0
if v_lines is not None:
    for line in v_lines:
        rho, theta = line[0]
        if (np.pi * 3 / 4 >= (theta % np.pi) >= np.pi * 1 / 4):
            continue
        if (i % 2 == 0):
            color = (0, 0, 255)
        if (i % 2 == 1):
            color = (0, 255, 0)
        i = i + 1

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + maxsize * (-b))
        y1 = int(0)
        x2 = int(x0 - maxsize * (-b))
        y2 = int(keyHeight)

        x_coordinates.append(x1)
        cv2.line(hough, (x1, y1), (x2, y2), (255, 0, 0), 5)

mask = np.zeros(cropped.shape)

keys = []
x_coordinates.sort()
key_shift = 9
# TODO - Calculating KeyShift
keyIndex = key_shift
white_key_x = []
Keyboard(cropped, x_coordinates, key_shift, keyHeight)