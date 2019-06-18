#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np


class BlackKeysDetector:

    def __init__(self, thresh: np.ndarray):
        """
        Keyboard Extractor
        :param thresh: Binarized keyboard image
        """

        self.edges = edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

        (h, w) = thresh.shape
        maxsize = int((h ** 2 + w ** 2) ** 0.5)

        # Vertical limit of black keys
        self.vlimit = h

        # Hough to get vertical and horizontal lines
        lines = cv2.HoughLines(edges, 1, np.pi / 2, int(w * 0.1))

        # Nothing to do
        if lines is None:
            return

        for line in lines:
            # Get rho and theta
            rho, theta = line[0]

            # Ignore vertical lines
            if not (np.pi * 3 / 4 >= (theta % np.pi) >= np.pi * 1 / 4):
                continue

            # Polar -> Rect
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho

            x1, y1 = int(x0 + maxsize * (-b)), int(y0 + maxsize * (a))
            x2, y2 = int(x0 - maxsize * (-b)), int(y0 - maxsize * (a))

            # Saving the new vertical limit
            self.vlimit = min(y1, y2)

            # Stop when a new limit is found
            break

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.nkeys = len(contours)

        self.mask = np.zeros(thresh.shape)

        for cnt in contours:

            # Fit a rectangle on the blob
            rect = cv2.minAreaRect(cnt)
            box = np.intp(cv2.boxPoints(rect))

            # Create a mask of the blob with original image shape
            blob = np.zeros(thresh.shape).astype(int)
            cv2.fillPoly(blob, [cnt], 1)

            # Create a mask of the rectangle with original image shape
            mask = np.zeros(thresh.shape).astype(int)
            cv2.fillPoly(mask, [box], 1)

            # Compute a error based in difference between blob mask and rectangle mask
            error = (mask * (1 - blob)).sum()

            # Error > 30% -> \( >.<")/
            if error / blob.sum() > 0.3:
                continue

            xf1, _ = box[0]
            xi1, _ = box[1]
            xf2, _ = box[2]
            xi2, _ = box[3]

            cv2.fillPoly(self.mask, [box], 1)

        self.mask[self.vlimit:, :] = 0


class FirstBlackKeyRecognition:

    def __init__(self, bk_detector: BlackKeysDetector):
        """
        FirstBlackKeyRecognition
        :param bk_detector: Black keys mask information
        """

        # Horizontal array with black keys mask
        self.__keys = bk_detector.mask[int(bk_detector.vlimit/2), :]

        # | i  c  iiii  c  c  iiii  c  iiii  c  c  i |   gaps (c = count, i = ignore)
        # |  C# D#    F# G# A#    C# D#    F# G# A#  |  black keys
        # | C  D  E  F  G  A  B  C  D  E  F  G  A  B |  white keys

        gaps = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]

        n_bkeys = 5
        n_gaps = 5

        gaps_fitness = []

        for shift in range(0, n_bkeys):
            # Start iter at first position
            self.__iter = 0

            if self.__keys[self.__iter] == 0:
                # Moving iterator to the first black key
                self.move_iter()

            # Keys size values
            keys_values = np.zeros(n_gaps)
            # Gaps size values
            gaps_values = np.zeros(n_gaps)
            # Mask to fitness function
            mask_values = np.array(gaps[shift:(shift + n_gaps)])

            # Find components (first element is a black key)
            for key in range(0, n_gaps):
                # Black key
                keys_values[key] = self.move_iter()
                # Gap between two black keys
                gaps_values[key] = self.move_iter()

            print(keys_values)
            print(gaps_values)
            print(mask_values)

            # Compute fitness value and save
            fitness = (gaps_values * mask_values).sum()
            gaps_fitness.append(fitness)

        # Minimum value of fitness
        self.first_key = np.array(gaps_fitness).argmin()

    @property
    def first_key_label(self):
        return ["C#", "D#", "F#", "G#", "A#"][self.first_key]

    def move_iter(self):
        keys = self.__keys
        iter = self.__iter

        # current value
        value = keys[iter]
        # count current value
        accu = 1

        for i in range((iter + 1), len(keys)):
            iter = i
            if keys[i] == value:
                accu += 1
            else:
                break

        self.__iter = iter
        return accu
