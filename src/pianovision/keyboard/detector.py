#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

from .data import Keyboard
from utils.pathbuilder import PathBuilder


class KeyboardDetector:

    def __init__(self, keyboard: Keyboard):
        """
        Keyboard Extractor
        :param image: Original image
        """
        self.kb = keyboard

        with open(str(PathBuilder().miscdir("crops.json"))) as f:
            self.__crops = json.load(f)

        image = self.kb.image
        crops = self.__crops[self.kb.resource.name]

        yi = crops["yi"] if "yi" in crops.keys() else 0
        yf = crops["yf"] if "yf" in crops.keys() else image.shape[0]

        xi = crops["xi"] if "xi" in crops.keys() else 0
        xf = crops["xf"] if "xf" in crops.keys() else image.shape[1]

        self.kb.cropped = image[yi:yf, xi:xf]
