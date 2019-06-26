#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from utils.resourceloader import ResourceLoader
from utils.bgextractor import BackgroundExtractor


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
        return [(self.x1i, self.yi), (self.x1f, self.yf), (self.x2f, self.yf), (self.x2i, self.yi)]

    @property
    def leftline(self):
        return [(self.x1i, self.yi), (self.x1f, self.yf)]

    @property
    def rightline(self):
        return [(self.x2i, self.yi), (self.x2f, self.yf)]

    @staticmethod
    def calc_x(y, x1, y1, x2, y2):
        # (y - y1) = (y2 - y1)/(x2 - x1) * (x - x1)
        return (x2 - x1) / (y2 - y1) * (y - y1) + x1

class KeyboardMask:

    mask: np.ndarray
    edges: np.ndarray
    thresh: np.ndarray

    def __init__(self, keyboard: np.ndarray):

        shape = keyboard.shape
        if len(shape) > 2:
            shape = (shape[0], shape[1])

        self.mask = np.zeros(shape)
        self.edges = np.zeros(shape)
        self.thresh = np.zeros(shape)

        self.vlimit = shape[0]

        self.keys = []

    def create_key(self, contour: list):

        if len(contour) != 4:
            raise RuntimeError("Invalid key contours")

        self.keys.append(KeyMask(contour, self.vlimit))


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
