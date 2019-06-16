#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np


class KeyboardExtractor:

    def __init__(self, image: np.ndarray):
        """
        Keyboard Extractor
        :param image: Original image
        """

        self.__original = image

    def run(self) -> np.ndarray:
        """
        :return: Cropped image
        """

        # TODO Detect keyboard
        cropped = self.__original[550:804, :]
        # cropped = self.__original[524:726, :]

        gsimg = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gsimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # thresh = cv2.adaptiveThreshold(gsimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 81, 0)

        kernel_size = int(cropped.shape[1] * 0.005)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Closing roles
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # Denoising
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Connected components (black keys)
        ret, labels = cv2.connectedComponents(thresh)
        teste = np.zeros(labels.shape)
        teste[labels != 0] = 255

        # Find contours
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        final = cropped.copy()
        for cnt in contours:
            box = cv2.boxPoints(cv2.minAreaRect(cnt))
            cv2.drawContours(final, [np.intp(box)], 0, (0, 0, 255), 2)

        return final
