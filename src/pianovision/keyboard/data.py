#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from utils.resourceloader import ResourceLoader
from utils.bgextractor import BackgroundExtractor


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

    def create_key(self, box):

        pass


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