import cv2
import numpy as np
from pianovision.keyboard.data import Keyboard, KeyValue

class MotionDetector:
    """ Senses the motion in the video comparing two at a time images, the current with the previous

    Args:
        red_hand_threshold: Threshold for hand detection
        hand_mask_dilation_size: Dilation kernel size for hand mask
        key_mask: keyboard key mask
        initial_buffer: Initial Buffer for motion detection
    """
    def __init__(self, red_hand_threshold, hand_mask_dilation_size,threshold_image, key_map, keyboard:Keyboard):
        """"Input type: first_image = 3 channel image of the current cropped keyboard frame"""
        self.keyboard = keyboard #
        self.key_mappings = keyboard.mask.bkeys
        self.key_mappings.extend(keyboard.mask.wkeys)
        for key in self.key_mappings:
            print(key.encode(key.id, key.octave), key.x1i )
        self.red_hand_threshold = red_hand_threshold
        self.hand_mask_dilation = hand_mask_dilation_size
        self.threshold_image = threshold_image
        self.key_map = key_map


        self.previous_hand_mask = np.zeros(key_map.shape)

        key_map_max = np.max(key_map)
        self.key_motion_values = np.zeros(((key_map_max>> 4) * 12) + (key_map_max & 15))

        self.current_hand_mask = np.zeros(key_map.shape)
        self.previous_hand_mask = np.zeros(key_map.shape)
        self.map_copy = key_map.copy()

    def detect_hand(self, rgb_frame):
        #blurred = cv2.GaussianBlur(rgb_frame, (11,11), 5)
        distance = (rgb_frame[:,:,2]*2  - rgb_frame[:,:,1] - rgb_frame[:,:,0])

        distance[rgb_frame[:,:,2] < 160] = 0
        distance[distance > 200] = 0
        hand_mask = np.zeros(distance.shape)
        hand_mask[distance > self.red_hand_threshold/2] = 1

        return(hand_mask)


    def detect_white_key_motion(self, current_frame, previous_frame, current_hand_mask, previous_hand_mask,
                                is_synthesia):
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)


        if not is_synthesia:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            im = np.zeros((50, 50), dtype=np.uint8)
            im[50:, 50:] = 255
            current_hand_mask = cv2.dilate(current_hand_mask, kernel, iterations=2)
            previous_hand_mask = cv2.dilate(previous_hand_mask, kernel, iterations=2)
            hand_region = cv2.blur(current_hand_mask, (1, current_hand_mask.shape[1]))
            current_hand_mask = 1 - current_hand_mask
            previous_hand_mask = 1 - previous_hand_mask
            current_hand_mask = cv2.GaussianBlur(current_hand_mask, (19,19), sigmaY=5, sigmaX=1)
            previous_hand_mask = cv2.GaussianBlur(previous_hand_mask, (19,19), sigmaY=5, sigmaX=1)

        frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)

        if not is_synthesia:
            frame_diff = frame_diff*current_hand_mask
            frame_diff = frame_diff*previous_hand_mask
            frame_diff[hand_region == 0] = 0
            cv2.imshow('hand', current_hand_mask)

        frame_diff[self.threshold_image > 0] = frame_diff[self.threshold_image > 0] * 100
        frame_diff = cv2.blur(frame_diff, (3,frame_diff.shape[0]))
        frame_diff[self.threshold_image == 0] = 0

        return(frame_diff)

    def detect_black_key_motion(self, current_frame,previous_frame, current_hand_mask, previous_hand_mask,
                                is_synthesia):

        frame_diff = np.zeros(current_frame[:,:, 0].shape)
        return(frame_diff)


    def detect_key_stroke(self, current_frame, previous_frame, is_synthesia):
        """"Detects the motion and the hand mask for the current frame"""
        if not is_synthesia :
            self.current_hand_mask = self.detect_hand(current_frame)
            self.previous_hand_mask= self.detect_hand(previous_frame)
        else:
            self.current_hand_mask = np.zeros(current_frame.shape)
            self.previous_hand_mask = np.zeros(current_frame.shape)



        white_key_diff = self.detect_white_key_motion(current_frame, previous_frame,self.current_hand_mask,
                                                      self.previous_hand_mask, is_synthesia)
        black_key_diff = self.detect_black_key_motion(current_frame, previous_frame,self.current_hand_mask,
                                                      self.previous_hand_mask, is_synthesia)

        """"Sums up the values fo the two images"""
        total_diff = white_key_diff + black_key_diff
        """"Removes noisy values from image"""
        for kernel_size in [11]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 3))
            im = np.zeros((50, 50), dtype=np.uint8)
            im[50:, 50:] = 255
            total_diff = cv2.morphologyEx(total_diff, cv2.MORPH_OPEN, kernel, iterations=1)

        """"Removes noisy values from image"""
        for kernel_size in [5, 3, 5, 3]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, kernel_size))
            im = np.zeros((50, 50), dtype=np.uint8)
            im[50:, 50:] = 255
            total_diff = cv2.morphologyEx(total_diff, cv2.MORPH_OPEN, kernel, iterations=2)

        total_diff = cv2.normalize(total_diff, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        """"Collects the motion values from the motion image using the key map"""
        for key_number in range(len(self.key_motion_values)):
            self.key_motion_values[key_number] = np.sum(total_diff[self.key_map == key_number])

        pressed_keys = self.gets_pressed_keys(self.key_motion_values, 4)

        for pressed_key in pressed_keys:
            self.map_copy[self.key_map == pressed_key] = 255
            for key in self.key_mappings:
                if(key.encode(key.id, key.octave) == pressed_key):
                    cv2.putText(self.map_copy, KeyValue.to_string(key.id) + str(key.octave), (key.x1i+5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (0, 0, 0), lineType=cv2.LINE_AA)


        cv2.imshow('pressed_key ',self.map_copy)
        cv2.imshow('difference seen ', np.uint8(total_diff * 255))
        self.map_copy = self.key_map.copy()


    def get_motion_frame(self,  current_frame, previous_frame):
        """"Detects the motion and the hand mask for the current frame"""
        self.current_hand_mask = self.detect_hand(current_frame)
        self.previous_hand_mask= self.detect_hand(previous_frame)



        white_key_diff = self.detect_white_key_motion(current_frame, previous_frame,self.current_hand_mask, self.previous_hand_mask)
        black_key_diff = self.detect_black_key_motion(current_frame, previous_frame,self.current_hand_mask, self.previous_hand_mask)

        """"Sums up the values fo the two images"""
        total_diff = white_key_diff + black_key_diff
        """"Removes noisy values from image"""
        # for kernel_size in [5]:
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, 3))
        #     im = np.zeros((50, 50), dtype=np.uint8)
        #     im[50:, 50:] = 255
        #     total_diff = cv2.morphologyEx(total_diff, cv2.MORPH_OPEN, kernel, iterations=2)

        """"Removes noisy values from image"""
        for kernel_size in [15, 9, 5, 3]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, kernel_size))
            im = np.zeros((50, 50), dtype=np.uint8)
            im[50:, 50:] = 255
            total_diff = cv2.morphologyEx(total_diff, cv2.MORPH_OPEN, kernel, iterations=3)

        total_diff = cv2.normalize(total_diff, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return(total_diff)

    def gets_pressed_keys(self, motion_values, initial_robustness):

        """Gets statistical data from the array"""
        mean = np.mean(motion_values)
        standard_dev = np.std(motion_values)

        if standard_dev.min() <= 0:
            return []

        """"Computes Z scores for each array element"""
        z_scores = (motion_values - mean)/standard_dev
        pressed_keys = []
        robustness = initial_robustness

        while(len(pressed_keys) == 0):
            pressed_keys = []
            for key_value in range(len(z_scores)):
                if z_scores[key_value] > robustness:
                    pressed_keys.append(key_value)
            robustness -=0.2

        """"Returns the values seen that have higher than X standard deviations"""
        return(pressed_keys)