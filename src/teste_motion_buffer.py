import cv2

from pianovision.MotionDetector_Test import MotionDetector
from utils.resourceloader import ResourceLoader

from pianovision.keyboard.base import Keyboard, KeyValue
from pianovision.keyboard.masking import KeyboardMasking
from pianovision.keyboard.mapping import KeyboardMapping

from pianovision.keyboard.detector import KeyboardDetector

from pianovision.sound import SoundAnalyser

import numpy as np

def get_sequential_motion_frame(buffer):
    diff_frame = np.zeros(buffer[0].shape)
    for cur_frame, next_frame in zip(buffer, buffer[1:]):
        diff_frame += cv2.absdiff(cur_frame, next_frame)
    return(diff_frame)

def iterate_buffer(buffer,new_frame):
    buffer.append(new_frame)
    buffer.pop(0)



cap = cv2.VideoCapture('alice.mp4')
ret, current_frame = cap.read()
current_frame = cv2.cvtColor(current_frame[350:470, :], cv2.COLOR_BGR2GRAY)
buffer_sizes = [2,3,4]
buffer_list = []
for buffer_size in buffer_sizes:
    buffer = []
    for i in range(buffer_size):
        if(ret):
            buffer.append(current_frame)
        ret, current_frame = cap.read()
        current_frame = cv2.cvtColor(current_frame[350:470, :], cv2.COLOR_BGR2GRAY)
    buffer_list.append(buffer)

print(buffer_list, len(buffer_list))

while(cap.isOpened()):
    if(ret):

        for buffer in buffer_list:
            iterate_buffer(buffer, current_frame)
            motion_frame = get_sequential_motion_frame(buffer)
            motion_frame = cv2.normalize(motion_frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.imshow('motion from ' + str(len(buffer)) + ' images', motion_frame)
        cv2.waitKey(0)
    ret, current_frame = cap.read()
    current_frame = cv2.cvtColor(current_frame[350:470, :], cv2.COLOR_BGR2GRAY)