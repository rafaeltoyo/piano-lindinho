#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

from utils.resourceloader import ResourceLoader

from .data import Keyboard, KeyValue
from .masking import KeyboardMasking
from .mapping import KeyboardMapping

from pianovision.keyboard.detector import KeyboardDetector


class KeyboardHandler:

    def __init__(self, resource: ResourceLoader):
        """
        Main class of keyboard module
        :param resource: video
        """

        self.__resource = resource

        # Creating the keyboard representation
        self.keyboard = Keyboard(resource)
        self.keyboard.loadImage()

        # Crop keyboard
        self.kbDetector = KeyboardDetector(self.keyboard)

        # Create mask for black keys
        self.kbMasking = KeyboardMasking(self.keyboard)

        # Create the mapping of keys
        self.kbMapping = KeyboardMapping(self.keyboard)

    def print(self):

        print(self.keyboard.mask.top_x_array)
        print(self.keyboard.mask.bottom_x_array)
        print(self.keyboard.mask.black_x_array)

        # First key
        print(KeyValue.to_string(self.keyboard.mask.bkeys[0].id))

        cv2.imshow("Keyboard - Original", self.keyboard.image)
        cv2.imshow("Keyboard - Cropped", self.keyboard.cropped)
        cv2.imshow("Keyboard - Binarized", self.keyboard.mask.thresh)
        cv2.imshow("Keyboard - Mask/Mapping", self.keyboard.mask.createMask(visual=False))
        cv2.waitKey()
