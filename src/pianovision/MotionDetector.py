import cv2
import numpy as np

class MotionDetector:
    """ Senses the motion in the video comparing two at a time images, the current with the previous

    Args:
        red_hand_threshold: Threshold for hand detection
        hand_mask_dilation_size: Dilation kernel size for hand mask
        key_mask: keyboard key mask
        initial_buffer: Initial Buffer for motion detection
    """
    def __init__(self, red_hand_threshold, hand_mask_dilation_size,threshold_image, first_image, key_map):
        """"Input type: first_image = 3 channel image of the current cropped keyboard frame"""

        self.red_hand_threshold = red_hand_threshold
        self.hand_mask_dilation = hand_mask_dilation_size
        self.threshold_image = threshold_image
        self.key_map = key_map

        self.previous_frame = first_image


        self.previous_frame_gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        self.previous_hand_mask = np.zeros(first_image[:, :, 0].shape)

        key_map_max = np.max(key_map)
        self.key_motion_values = np.zeros(((key_map_max>> 4) * 12) + (key_map_max & 15))

        self.current_hand_mask = np.zeros(first_image[:,:,0].shape)
        self.previous_hand_mask = np.zeros(first_image[:,:,0].shape)
        self.map_copy = key_map.copy()

    def detect_hand(self, rgb_frame):
        blurred = cv2.GaussianBlur(rgb_frame, (11,11), 5)
        distance = (rgb_frame[:,:,2]*2  - rgb_frame[:,:,1] - rgb_frame[:,:,0])

        distance[rgb_frame[:,:,2] < 160] = 0
        distance[distance > 200] = 0
        hand_mask = np.zeros(distance.shape)
        hand_mask[distance > self.red_hand_threshold] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        im = np.zeros((50, 50), dtype=np.uint8)
        im[50:, 50:] = 255

        hand_mask = cv2.dilate(hand_mask, kernel, iterations=2)

        return(hand_mask)


    def detect_white_key_motion(self, current_frame):
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        frame_diff = cv2.absdiff(current_frame_gray, cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY))
        frame_diff[self.threshold_image > 0 ] = frame_diff[self.threshold_image > 0]*20
        frame_diff[self.current_hand_mask == 1] = 0


        frame_diff = cv2.blur(frame_diff, (13,13))
        frame_diff = cv2.normalize(frame_diff, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return(frame_diff)

    def detect_black_key_motion(self, current_frame):

        frame_diff = np.zeros(current_frame[:,:, 0].shape)
        return(frame_diff)

    def detect_key_stroke(self, current_frame):
        self.current_hand_mask = self.detect_hand(current_frame)
        white_key_diff = self.detect_white_key_motion(current_frame)
        black_key_diff = self.detect_black_key_motion(current_frame)

        total_diff = white_key_diff + black_key_diff
        for key_number in range(len(self.key_motion_values)):
            self.key_motion_values[key_number] = np.sum(total_diff[self.key_map == key_number])

        pressed = self.key_motion_values.argmax()
        self.map_copy[self.key_map == pressed] = 1000
        cv2.imshow('pressed_key ',self.map_copy )
        self.map_copy[self.key_map == pressed] = self.key_map[self.key_map == pressed]

        self.previous_hand_mask = self.current_hand_mask
        self.previous_frame = current_frame



