#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

from utils.resourceloader import ResourceLoader

from .data import Keyboard
from .masking import KeyboardMasking

from .func.detector import KeyboardDetector
from .func.firstkey import FirstBlackKeyRecognition


class KeyboardHandler:

    def __init__(self, resource: ResourceLoader):
        """
        Main class of keyboard module
        :param resource: video
        """

        self.__resource = resource
        self.__keyboard = Keyboard(resource)

        # Load image
        self.__keyboard.loadImage()

        # Crop keyboard
        self.__keyboard.cropped = KeyboardDetector(self.__keyboard.image).cropped

        # Create mask
        KeyboardMasking(self.__keyboard)

        cv2.imshow("teste", self.__keyboard.image)
        cv2.waitKey()
        cv2.imshow("teste", self.__keyboard.cropped)
        cv2.waitKey()
        cv2.imshow("teste", self.__keyboard.mask.thresh)
        cv2.waitKey()
        cv2.imshow("teste", self.__keyboard.mask.mask)
        cv2.waitKey()

        first_black_keys = FirstBlackKeyRecognition(self.__keyboard.mask.mask[int(self.__keyboard.mask.vlimit/2), :])

        print(first_black_keys.first_key_label)
