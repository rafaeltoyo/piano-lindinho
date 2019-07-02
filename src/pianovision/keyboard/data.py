#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import IntEnum
from typing import List, Tuple

import cv2
import numpy as np

from utils.resourceloader import ResourceLoader
from utils.bgextractor import BackgroundExtractor


class KeyValue(IntEnum):
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
    def ordered_keys():
        return [KeyValue.KEY_C, KeyValue.KEY_Cs,
                KeyValue.KEY_D, KeyValue.KEY_Ds,
                KeyValue.KEY_E,
                KeyValue.KEY_F, KeyValue.KEY_Fs,
                KeyValue.KEY_G, KeyValue.KEY_Gs,
                KeyValue.KEY_A, KeyValue.KEY_As,
                KeyValue.KEY_B]

    @staticmethod
    def to_string(key_value: int) -> str:
        if key_value is None or key_value < 1 or key_value > 12:
            return ""
        return ["", "C", "D", "E", "F", "G", "A", "B", "C#", "D#", "F#", "G#", "A#"][key_value]

    @staticmethod
    def sharp(key_value: int) -> int:
        if key_value < 1 or key_value > 12:
            return 0
        keys = KeyValue.ordered_keys()
        return keys[(keys.index(key_value) + 1) % len(keys)]

    @staticmethod
    def flat(key_value: int) -> int:
        if key_value < 1 or key_value > 12:
            return 0
        keys = KeyValue.ordered_keys()
        return keys[(keys.index(key_value) + len(keys) - 1) % len(keys)]


class KeyMask:

    id: KeyValue
    octave: int

    def __init__(self, contour, yf, yi=0):
        if len(contour) != 4:
            raise RuntimeError("Invalid key contours")

        self.id = None
        self.octave = 0

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

        # utils
        self.xi = int((self.x2i + self.x1i) / 2)
        self.xf = int((self.x2f + self.x1f) / 2)
        self.pi = (self.xi, self.yi)
        self.pf = (self.xf, self.yf)

    @property
    def contour(self) -> List[Tuple[int, int]]:
        return [(self.x2i, self.yi), (self.x1i, self.yi), (self.x1f, self.yf), (self.x2f, self.yf)]

    @property
    def area(self) -> int:
        return cv2.contourArea(np.array(self.contour))

    @property
    def left_line(self) -> List[Tuple[int, int]]:
        return [(self.x1i, self.yi), (self.x1f, self.yf)]

    @property
    def right_line(self) -> List[Tuple[int, int]]:
        return [(self.x2i, self.yi), (self.x2f, self.yf)]

    @property
    def pixel_value(self) -> int:
        return KeyMask.encode(self.id, self.octave)

    @staticmethod
    def calc_x(y: float, x1: float, y1: float, x2: float, y2: float) -> int:
        return int((x2 - x1) * (y - y1) / (y2 - y1) + x1)

    @staticmethod
    def decode(pixel: int) -> Tuple[KeyValue, int]:
        octave = (pixel & 0b011110000) >> 4
        note = (pixel & 0b01111)
        return KeyValue(note), octave

    @staticmethod
    def encode(note: KeyValue, octave: int) -> int:
        if note is None:
            return 255
        return (int(note) & 0b01111) | ((octave & 0b01111) << 4)


class KeyboardMask:
    edges: np.ndarray
    thresh: np.ndarray
    vlimit: int
    bkeys: List[KeyMask]
    wkeys: List[KeyMask]

    def __init__(self, keyboard: np.ndarray):

        shape = keyboard.shape
        if len(shape) > 2:
            shape = (shape[0], shape[1])

        self.edges = np.zeros(shape)
        self.thresh = np.zeros(shape)

        self.vlimit = shape[0]

        self.bkeys = []
        self.wkeys = []

    def addKey(self, contour: list):

        if len(contour) != 4:
            raise RuntimeError("Invalid key contours")

        key = KeyMask(contour, self.vlimit)
        self.bkeys.append(key)

    def createMask(self, visual: bool = False):

        # Font cuz unicorns exist
        font = cv2.FONT_HERSHEY_SIMPLEX

        mask = np.zeros(self.thresh.shape).astype('uint8')

        swap = False
        for key in self.wkeys:
            # Create the key in mask
            if visual:
                cv2.fillPoly(mask, [np.intp(key.contour)], 64 if swap else 32)
                swap = not swap

                text = KeyValue.to_string(key.id) + str(key.octave)
                cv2.putText(mask, text, (key.xf, self.thresh.shape[0] - 50), font, 0.4, 255, 1, cv2.LINE_AA)
            else:
                cv2.fillPoly(mask, [np.intp(key.contour)], key.pixel_value)

        for key in self.bkeys:
            # Create the key in mask
            if visual:
                cv2.fillPoly(mask, [np.intp(key.contour)], 127)

                text = KeyValue.to_string(key.id) + str(key.octave)
                cv2.putText(mask, text, (key.xi, 50), font, 0.4, 255, 1, cv2.LINE_AA)
            else:
                cv2.fillPoly(mask, [np.intp(key.contour)], key.pixel_value)

        return mask

    @property
    def top_x_array(self):
        return np.ravel(sorted([[k.x1i, k.x2i] for k in self.bkeys], key=lambda p: p[0]))

    @property
    def bottom_x_array(self):
        return np.ravel(sorted([[k.x1f, k.x2f] for k in self.bkeys], key=lambda p: p[0]))

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
