import cv2

from pianovision.MotionDetector import MotionDetector
from utils.resourceloader import ResourceLoader

from pianovision.keyboard.base import Keyboard, KeyValue
from pianovision.keyboard.masking import KeyboardMasking
from pianovision.keyboard.mapping import KeyboardMapping

from pianovision.keyboard.detector import KeyboardDetector

from pianovision.sound import SoundAnalyser

from utils.videohandler import VideoHandler
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
__resource = ResourceLoader("alice")

# Creating the keyboard representation
__keyboard = Keyboard(__resource)
__keyboard.loadImage()

analyser = SoundAnalyser(__resource.audioname)
peak_frames = analyser.peak_frames(30)

#print(peak_frames)

# Crop keyboard
KeyboardDetector(__keyboard)

# Create mask for black keys
kbMasking = KeyboardMasking(__keyboard)

# TODO Estimate white keys

print(__keyboard.mask.top_x_array)
print(__keyboard.mask.bottom_x_array)
print(__keyboard.mask.black_x_array)

# TODO Keys mapping

kbMapping = KeyboardMapping(__keyboard)
# Estimate first key
print(KeyValue.to_string(__keyboard.mask.bkeys[0].id))

# FIXME Debug the result


cap = cv2.VideoCapture('alice.mp4')
ret, current_frame = cap.read()
current_frame = current_frame[350:470 ,0:current_frame.shape[1]]
previous_frame = current_frame

previous_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
previous_frame_hsv =  cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

# min_white_key_val = int(input('type in what the minimum white key value is: '))
# max_white_key_sat = int(input('type in what the maximum white key saturation is: '))
min_white_key_val = 100
max_white_key_sat = 7

red_difference = 40
red_threshold = 160

previous_hand_mask = np.zeros(previous_frame_gray.shape)

altura, largura = previous_frame_gray.shape

mask = __keyboard.mask.createMask(visual=False)

mask_copy = mask.copy()
number_mask = 15

framenumber = 0
cur_beat = 1

max_mask = np.max(mask)
key_motion_values = np.zeros(((max_mask >> 4)*12) + (max_mask&15))

white_key_mask = np.zeros(__keyboard.mask.thresh.shape)
white_key_mask[__keyboard.mask.thresh == 0] = 1

motion_detector = MotionDetector(40, 2, white_key_mask, mask)

detection_delay = 1

while(cap.isOpened()):
    ret, current_frame = cap.read()
    if(ret):
        current_frame = current_frame[350:470,:]
        if(framenumber == int(peak_frames[cur_beat])):
            cv2.imshow('video', current_frame)
            cv2.waitKey(0)
            cur_beat +=1
        if(framenumber+detection_delay == int(peak_frames[cur_beat])):
            motion_detector.detect_key_stroke(current_frame, previous_frame, True)
        framenumber+=1
        previous_frame = current_frame

cap.release()
cv2.destroyAllWindows()
