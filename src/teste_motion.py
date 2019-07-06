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

__resource = ResourceLoader("alice")

# Creating the keyboard representation
__keyboard = Keyboard(__resource)
__keyboard.loadImage()

# analyser = SoundAnalyser(__resource.audioname)
# peak_frames = analyser.peak_frames(30)

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

cv2.imshow("teste", __keyboard.image)
cv2.waitKey()
cv2.imshow("teste", __keyboard.cropped)
cv2.waitKey()
cv2.imshow("teste", __keyboard.mask.thresh)
cv2.waitKey()
cv2.imshow("teste", __keyboard.mask.createMask(visual=False))
# cv2.waitKey()


cap = cv2.VideoCapture('alice.mp4')
ret, current_frame = cap.read()
current_frame = current_frame[350:470 ,0:current_frame.shape[1]]
previous_frame = current_frame

previous_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
previous_frame_hsv =  cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

cv2.imshow('frame_test', current_frame)
cv2.waitKey(0)

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
cur_beat = 0

max_mask = np.max(mask)
print(np.sum(previous_frame_gray[mask == 124]))
key_motion_values = np.zeros(((max_mask >> 4)*12) + (max_mask&15))

white_key_mask = np.zeros(__keyboard.mask.thresh.shape)
white_key_mask[__keyboard.mask.thresh == 0] = 1

motion_detector = MotionDetector(40, 2, white_key_mask, current_frame, mask)



while(cap.isOpened()):
    #
    # current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    #
    # current_frame_gray = cv2.GaussianBlur(current_frame_gray, (11,11), 3)
    #
    # current_frame_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    #
    # current_frame_hsv = cv2.GaussianBlur(current_frame_hsv, (11,11),3)
    #
    # current_frame_sat = current_frame_hsv[:,:,1]
    #
    # frame_diff_l = cv2.absdiff(current_frame_gray, previous_frame_gray)
    #
    # frame_diff_s = cv2.absdiff(current_frame_sat, previous_frame_hsv[:,:,1])
    #
    #
    # hand_mask = np.zeros(frame_diff_l.shape)
    # red_diff_frame = current_frame[:,:,2]*2  - current_frame[:,:,1] - current_frame[:,:,0]
    # red_diff_frame[current_frame[:,:,2] < red_threshold] = 0
    # red_diff_frame[red_diff_frame > 200] = 0
    # hand_mask[red_diff_frame > red_difference] = 1
    #
    # cv2.imshow('teste_red', motion_detector.detect_hand(current_frame))
    # cv2.waitKey(0)
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    # im = np.zeros((50, 50), dtype=np.uint8)
    # im[50:, 50:] = 255
    #
    # mean, std_dev = cv2.meanStdDev(frame_diff_l)
    #
    # hand_mask = cv2.dilate(hand_mask, kernel, iterations=2)
    # #frame_diff_l[frame_diff_l < mean + std_dev] = 0
    #
    #
    #
    # frame_diff_l[previous_hand_mask == 1] = 0
    # frame_diff_l[white_key_mask == 1] = frame_diff_l[white_key_mask == 1]*20
    # frame_diff_l[hand_mask == 1] = 0
    #
    # frame_diff_l = cv2.blur(frame_diff_l, (13,13))
    # frame_diff_l = cv2.normalize(frame_diff_l, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #
    #
    #
    # cur_beat += 1
    #
    # for key_number in range(len(key_motion_values)):
    #     key_motion_values[key_number] = np.sum(frame_diff_l[mask == key_number])
    #
    # pressed = key_motion_values.argmax()
    #
    # mask_copy[mask == pressed] = 1000
    # cv2.imshow('frame', mask_copy)
    # cv2.waitKey(0)
    # mask_copy[mask == pressed] = mask[mask == pressed]
    #
    # cv2.imshow('frame diff saturation',frame_diff_l)
    #
    # cv2.waitKey(0)
    # previous_frame_gray = current_frame_gray.copy()
    # previous_frame_hsv = current_frame_hsv.copy()
    # previous_hand_mask = hand_mask
    # ret, current_frame = cap.read()
    # framenumber +=1
    # sleep(0.05)
    # current_frame = current_frame[350:470,:]
    # teste = cv2.absdiff(current_frame, previous_frame)
    # cv2.imshow('teste', hand_mask)
    # cv2.waitKey(0)
    ret, current_frame = cap.read()
    current_frame = current_frame[350:470, :]
    motion_detector.detect_key_stroke(current_frame)
    cv2.imshow('teste', current_frame )
    cv2.waitKey(0)


cap.release()
cv2.destroyAllWindows()
