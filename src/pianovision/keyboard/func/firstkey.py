#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np


class FirstBlackKeyRecognition:

    def __init__(self, h_keyboard_sample: np.ndarray, debug: bool = False):
        """
        FirstBlackKeyRecognition
        :param bk_detector: Horizontal array with black keys mask
        """

        self.__keys = h_keyboard_sample

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

            if debug:
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
