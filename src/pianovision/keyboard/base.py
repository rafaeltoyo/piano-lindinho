#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

from utils.resourceloader import ResourceLoader

from .data import Keyboard, KeyValue
from .masking import KeyboardMasking
from .mapping import KeyboardMapping

from pianovision.keyboard.detector import KeyboardDetector
from .func.firstkey import FirstBlackKeyRecognition


class KeyboardHandler:

    def __init__(self, resource: ResourceLoader):
        """
        Main class of keyboard module
        :param resource: video
        """

        self.__resource = resource

        # Creating the keyboard representation
        self.__keyboard = Keyboard(resource)
        self.__keyboard.loadImage()

        # Crop keyboard
        KeyboardDetector(self.__keyboard)

        # Create mask for black keys
        kbMasking = KeyboardMasking(self.__keyboard)
        mask = self.__keyboard.mask.createMask()

        # TODO Estimate white keys

        print(self.__keyboard.mask.top_x_array)
        print(self.__keyboard.mask.bottom_x_array)
        print(self.__keyboard.mask.black_x_array)

        # TODO Keys mapping
        kbMapping = KeyboardMapping(self.__keyboard)
        # Estimate first key
        print(KeyValue.to_string(kbMapping.estimate_first_black_key()))

        # FIXME Debug the result

        cv2.imshow("teste", self.__keyboard.image)
        cv2.waitKey()
        cv2.imshow("teste", self.__keyboard.cropped)
        cv2.waitKey()
        cv2.imshow("teste", self.__keyboard.mask.thresh)
        cv2.waitKey()
        cv2.imshow("teste", mask)
        cv2.waitKey()
