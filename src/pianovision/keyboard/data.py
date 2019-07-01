#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum
from typing import List

import cv2
import numpy as np

from utils.resourceloader import ResourceLoader
from utils.bgextractor import BackgroundExtractor


class KeyValue(Enum):

    KEY_C = 1
    KEY_D = 2
    KEY_E = 3
    KEY_F = 4
    KEY_G = 5
    KEY_A = 6
    KEY_B = 7
    KEY_Cs = KEY_Db = 8
    KEY_Ds = KEY_Eb = 9
    KEY_Fs = KEY_Gb = 10
    KEY_Gs = KEY_Ab = 11
    KEY_As = KEY_Bb = 12

    @staticmethod
    def to_string(key_value: int):
        if key_value < 1 or key_value > 12:
            return ""
        return ["", "C", "D", "E", "F", "G", "A", "B", "C#", "D#", "F#", "G#", "A#"][key_value]


class KeyMask:

    def __init__(self, contour, yf, yi=0):

        if len(contour) != 4:
            raise RuntimeError("Invalid key contours")

        self.yi = yi
        self.yf = yf

        contour.sort(key=lambda p: p[1])

        initial_points = contour[0:2]
        initial_points.sort(key=lambda p: p[0])

        final_points = contour[2:4]
        final_points.sort(key=lambda p: p[0])

        self.x1i = KeyMask.calc_x(yi, *initial_points[0], *final_points[0])
        self.x1f = KeyMask.calc_x(yf, *initial_points[0], *final_points[0])

        self.x2i = KeyMask.calc_x(yi, *initial_points[1], *final_points[1])
        self.x2f = KeyMask.calc_x(yf, *initial_points[1], *final_points[1])

    @property
    def contour(self):
        return [(self.x2i, self.yi), (self.x1i, self.yi), (self.x1f, self.yf), (self.x2f, self.yf)]

    @property
    def area(self):
        return cv2.contourArea(np.array(self.contour))

    @property
    def left_line(self):
        return [(self.x1i, self.yi), (self.x1f, self.yf)]

    @property
    def right_line(self):
        return [(self.x2i, self.yi), (self.x2f, self.yf)]

    @staticmethod
    def calc_x(y, x1, y1, x2, y2):
        return int((x2 - x1) * (y - y1) / (y2 - y1) + x1)


class KeyboardMask:

    edges: np.ndarray
    thresh: np.ndarray
    vlimit: int
    keys: List[KeyMask]

    def __init__(self, keyboard: np.ndarray):

        shape = keyboard.shape
        if len(shape) > 2:
            shape = (shape[0], shape[1])

        self.edges = np.zeros(shape)
        self.thresh = np.zeros(shape)

        self.vlimit = shape[0]

        self.keys = []

    def addKey(self, contour: list):

        if len(contour) != 4:
            raise RuntimeError("Invalid key contours")

        key = KeyMask(contour, self.vlimit)
        self.keys.append(key)

    def createMask(self):

        mask = np.zeros(self.thresh.shape).astype('uint8')

        for key in self.keys:
            # Create the key in mask
            cv2.fillPoly(mask, [np.intp(key.contour)], 255)

        return mask

    @property
    def top_x_array(self):
        return np.ravel(sorted([[k.x1i, k.x2i] for k in self.keys], key=lambda p: p[0]))

    @property
    def bottom_x_array(self):
        return np.ravel(sorted([[k.x1f, k.x2f] for k in self.keys], key=lambda p: p[0]))

    @property
    def black_x_array(self):
        return [i for i in zip(self.top_x_array, self.bottom_x_array)]


class Keyboard:

    resource: ResourceLoader
    image: np.ndarray
    cropped: np.ndarray
    mask: KeyboardMask

    def __init__(self, resource: ResourceLoader):
        """

        :param resource: Video
        """

        self.resource = resource
        resource.create("keyboard")

        self.image = None
        self.cropped = None
        self.mask = None

    def loadImage(self, auto=True, debug=False):

        image_name = self.resource["keyboard"]

        if not image_name.exists():

            if not auto:
                raise RuntimeError(str(image_name) + " doesn't exists!")

            if debug:
                print("Creating keyboard imagem ...")

            background = BackgroundExtractor(self.resource.videoname).run()
            cv2.imwrite(str(image_name), background)

            if debug:
                print("Done!")

            self.image = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        else:
            self.image = cv2.imread(str(image_name))
