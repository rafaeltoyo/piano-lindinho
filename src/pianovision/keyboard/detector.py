#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np

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

        self.yi = crops["yi"] if "yi" in crops.keys() else 0
        self.yf = crops["yf"] if "yf" in crops.keys() else image.shape[0]

        self.xi = crops["xi"] if "xi" in crops.keys() else 0
        self.xf = crops["xf"] if "xf" in crops.keys() else image.shape[1]

        self.kb.cropped = self.crop(image)

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.yi:self.yf, self.xi:self.xf]
