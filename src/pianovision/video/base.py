#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from .motion import MotionDetector

from ..keyboard import KeyboardHandler
from ..sound import SoundAnalyser

from utils.videohandler import VideoHandler
from utils.resourceloader import ResourceLoader


class MotionHandler:

    def __init__(self, resource: ResourceLoader, debug=False):
        self.debug = debug

        # Keyboard handler
        self.kbHandler = KeyboardHandler(resource)
        if self.debug:
            self.kbHandler.print()

        # Process the audio
        self.beat_frames = SoundAnalyser(resource.audioname).peak_frames(30)
        self.current_beat = 0
        self.current_frame_iter = 1
        self.beat_detection_delay = 1

        #
        self.previous_frame = None
        self.current_frame = None

        # Create a key mapping
        mask = self.kbHandler.keyboard.mask.createMask(visual=False)

        # Create a white keys mask
        thresh = self.kbHandler.keyboard.mask.thresh.copy()
        white_key_mask = np.zeros(thresh.shape)
        white_key_mask[thresh == 0] = 1

        # Motion detector
        self.motion_detector = MotionDetector(40, 2, white_key_mask, mask)

        # Create a video handler
        player = VideoHandler(resource.videoname, self.process_frame)
        player.run()

    def process_frame(self, frame: np.ndarray) -> bool:

        # Save the first frame and go to the next frame
        if self.previous_frame is None:
            self.previous_frame = self.current_frame = self.kbHandler.kbDetector.crop(frame)
            return True

        # Update current frame
        self.current_frame = self.kbHandler.kbDetector.crop(frame)

        if self.current_frame_iter == int(self.beat_frames[self.current_beat]):

            # Next beat
            self.current_beat += 1

        if (self.current_frame_iter + self.beat_detection_delay) == int(self.beat_frames[self.current_beat]):

            self.motion_detector.detect_key_stroke(self.current_frame, self.previous_frame)
            cv2.waitKey(0 if self.debug else 5)

        # Show current frame
        cv2.imshow("video", self.current_frame)

        # Next frame ...
        self.current_frame_iter += 1
        self.previous_frame = self.current_frame

        # Continue after 5 seconds or quit
        return not (cv2.waitKey(5) & 0xFF == ord('q'))
