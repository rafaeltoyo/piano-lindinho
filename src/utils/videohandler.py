#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from typing import Callable

from utils.pathbuilder import PathBuilder


class VideoHandler:

    def __init__(self, filename: str, fn: Callable[[np.ndarray], bool]):
        """
        Default constructor
        :param filename: Video filename
        :param fn: Frame handler
        :return: NA
        """

        self._fn = fn
        self._cap = cv2.VideoCapture(str(PathBuilder().miscdir(filename)))

        if not self._cap.isOpened():
            raise Exception("Error opening video stream or file")

    def _setup(self):
        """
        Setup video stream configuration before read a frame
        :return: NA
        """

        pass

    def run(self):
        """
        Process the video
        :return: NA
        """

        try:
            while self._cap.isOpened():

                self._setup()
                ret, frame = self._cap.read()

                if not ret or not self._fn(frame):
                    break

        finally:
            self._cap.release()
            cv2.destroyAllWindows()
