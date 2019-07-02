#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import cv2
import numpy as np

from .data import Keyboard, KeyboardMask, KeyValue
from .func.findkeys import estimateBlackKey, createWhiteKeys, estimateOctaves


class KeyboardMapping:

    def __init__(self, keyboard: Keyboard):

        self.kb = keyboard

        self.__mapping = np.zeros(self.kb.mask.thresh.shape)

        keysize = np.array([key.area for key in self.kb.mask.bkeys]).mean()

        estimateBlackKey(keyboard)
        self.kb.mask.wkeys.extend(createWhiteKeys(self.kb))

        estimateOctaves(self.kb)

    def white_x_array(self):

        # C# 8 / D# 9 / F# 10 / G# 11 / A# 12
        first = self.first

        # Remove the offset
        # C# 0 / D# 1 / F# 2 / G# 3 / A# 4
        fixed_first = first - KeyValue.KEY_Cs

        # Estimate the number of keys in first octave
        # C# 5 / D# 4 / F# 3 / G# 2 / A# 1
        first_octave_keys = 5 - fixed_first

        # Get the number of black keys
        n_black_keys = len(self.kb.mask.bkeys)

        # Estimate number of octaves
        # The first octave may be incomplete so remove it keys
        # A octave has 5 black keys or less (a incomplete octave at the end of image)
        # Add the first octave (add 1)
        n_octaves = math.ceil((n_black_keys - first_octave_keys) / 5) + 1

        #                                          VVV The middle of image
        # | A0 B0 | C1  -  B1 | C2  -  B2 | C3  -  B3 | C4  -  B4 | C5  -  B5 | C6  -  B6 | C7  -  B7 | C8 |
        # |  A0#  | C#1 - A#1 | C#2 - A#2 | C#3 - A#3 | C#4 - A#4 | C#5 - A#5 | C#6 - A#6 | C#7 - A#7 |    |
        first_octave = 4 - int(n_octaves / 2)

        keys = sorted([(k.x1i, k.x2i) for k in self.kb.mask.bkeys], key=lambda p: p[0])

        # Iterators
        iter_key = first
        current_octave = first_octave

        # Previous key division
        previous = None

        mapping = np.zeros(self.kb.mask.thresh.shape)

        white_x_array = []

        for key in self.kb.mask.bkeys:

            # Estimate division on middle of black key
            xi = (key.x2i - key.x1i) / 2
            xf = (key.x2f - key.x1f) / 2

            if previous is None:
                previous = (xi, xf)
                continue

            (xip, xfp) = previous
            previous = (xi, xf)

            if iter_key == KeyValue.KEY_Cs or iter_key == KeyValue.KEY_Fs:
                # Estimate two white keys
                (xim, xfm) = ((xip-xi)/2, (xfp-xf)/2)

                white_x_array.append([(xim-xip)/2, (xfm-xfp)/2])
                white_x_array.append([(xi-xim)/2, (xf-xfm)/2])

            else:
                white_x_array.append([(xi-xip)/2, (xf-xfp)/2])

            # Go to the next black key
            if iter_key == KeyValue.KEY_As:
                iter_key = KeyValue.KEY_Cs
            else:
                iter_key += 1

        # White key
        current_key = KeyValue.flat(iter_key)

        # Pixel value
        pixel = int(current_key) & (current_octave << 4)

        # Draw the white key in mapping
        cv2.fillPoly(mapping, np.array([xi, xip, xfp, xf]), pixel)
