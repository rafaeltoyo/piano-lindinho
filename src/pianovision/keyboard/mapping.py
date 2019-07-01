#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from .data import Keyboard, KeyboardMask, KeyValue


class KeyboardMapping:

    def __init__(self, keyboard: Keyboard):

        self.kb = keyboard

        self.__mapping = np.zeros(self.kb.mask.thresh.shape)

        keysize = np.array([key.area for key in self.kb.mask.keys]).mean()

        self.first = self.estimate_first_black_key()

    def calc_gaps(self, corners=True):
        """
        Count the number of pixel between black keys starting at x=0.
        :return:
        """

        gaps = []
        left_key = 0

        keys = sorted([(k.x1i, k.x2i) for k in self.kb.mask.keys], key=lambda p: p[0])

        for (xi, xf) in keys:
            gaps.append(xi - left_key)
            left_key = xf

        if corners:
            gaps.append(self.kb.mask.thresh.shape[1] - left_key - 1)
            return gaps

        return gaps[1:]

    def estimate_first_black_key(self):
        """
        Estimate the first black key

        Keyboard layout
        | i  c  iiii  c  c  iiii  c  iiii  c  c  i |  gaps (c = count, i = ignore)
        |  C# D#    F# G# A#    C# D#    F# G# A#  |  black keys
        | C  D  E  F  G  A  B  C  D  E  F  G  A  B |  white keys

        Count the number of pixel on small gaps between black keys for fitness measure
        So the smallest value of fitness is the best mask fit and we found the first key note
        :return: ID of first black key
        """

        # Create a mask to extract that gaps (2 octaves)
        mask = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]

        # Gaps between black keys
        gaps = np.array(self.calc_gaps(corners=False))

        fitness = []

        for shift in range(0, 5):
            test = np.array(gaps[0:5] * mask[shift:(shift + 5)])
            fitness.append(test.sum())

        return np.array(fitness).argmin() + KeyValue.KEY_Cs.value

    def white_x_array(self):

        pass
