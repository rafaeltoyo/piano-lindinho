#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np


class KeyboardDetector:

    def __init__(self, image: np.ndarray):
        """
        Keyboard Extractor
        :param image: Original image
        """

        # TODO implements

        self.cropped = image[550:804, :]
        # self.cropped = image[524:726, :]
