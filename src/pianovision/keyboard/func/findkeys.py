#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List

import math
import cv2
import numpy as np

from ..data import Keyboard, KeyMask, KeyValue


def countBlackKeysGaps(kb: Keyboard, corners=True) -> List[int]:
    """
    Count the number of pixel between black keys starting at x=0.
    :param kb: Keyboard
    :param corners: Return gaps between the keys and the edges of image
    :return:
    """

    gaps = []
    left_key = 0

    for (xi, xf) in sorted([(k.x1i, k.x2i) for k in kb.mask.bkeys], key=lambda p: p[0]):
        gaps.append(xi - left_key)
        left_key = xf

    if corners:
        gaps.append(kb.mask.thresh.shape[1] - left_key - 1)
        return gaps

    return gaps[1:]


def estimateBlackKey(kb: Keyboard):
    """
    Estimate all black key note

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
    gaps = np.array(countBlackKeysGaps(kb, corners=False))

    # All fitness values for different mask shift
    fitness = []

    # A octave has 5 gaps
    crop_size = min(5, len(gaps))

    # C# 0 / D# 1 / F# 2 / G# 3 / A# 4
    for shift in range(0, 5):
        test = np.array(gaps[0:crop_size] * mask[shift:(shift + crop_size)])
        fitness.append(test.sum())

    # The best fitness is the least value
    current = np.array(fitness).argmin()

    # Mapping black keys
    for key in sorted(kb.mask.bkeys, key=lambda k: k.x1i):
        key.id = KeyValue(current + int(KeyValue.KEY_Cs))
        current = (current + 1) % 5


def createWhiteKeys(kb: Keyboard):

    black_keys = sorted(kb.mask.bkeys, key=lambda k: k.x1i)
    white_keys = []

    height = kb.mask.thresh.shape[0] - 1

    for i in range(1, len(black_keys)):
        bkey1 = black_keys[i - 1]
        bkey2 = black_keys[i]

        # Gap with 2 white keys
        if bkey2.id == KeyValue.KEY_Cs or bkey2.id == KeyValue.KEY_Fs:
            # Find a division between black keys
            pi = int((bkey2.xi + bkey1.xi)/2), int((bkey2.yi + bkey1.yi)/2)
            pf = int((bkey2.xf + bkey1.xf)/2), int((bkey2.yf + bkey1.yf)/2)

            wkey1 = KeyMask([pi, bkey2.pi, bkey2.pf, pf], height)
            wkey1.id = KeyValue.flat(bkey2.id)
            white_keys.append(wkey1)

            wkey2 = KeyMask([bkey1.pi, pi, pf, bkey1.pf], height)
            wkey2.id = KeyValue.sharp(bkey1.id)
            white_keys.append(wkey2)

        else:
            wkey = KeyMask([bkey1.pi, bkey2.pi, bkey2.pf, bkey1.pf], height)
            wkey.id = KeyValue.flat(bkey2.id)
            white_keys.append(wkey)

    white_keys.sort(key=lambda k: k.xi)
    return white_keys


def estimateOctaves(kb: Keyboard):

    # Get the number of black keys
    n_black_keys = len(kb.mask.bkeys)

    # C# 8 / D# 9 / F# 10 / G# 11 / A# 12
    first = int(kb.mask.bkeys[0].id)

    # Remove the offset
    # C# 0 / D# 1 / F# 2 / G# 3 / A# 4
    fixed_first = first - KeyValue.KEY_Cs

    # Estimate the number of keys in first octave
    # C# 5 / D# 4 / F# 3 / G# 2 / A# 1
    first_octave_keys = 5 - fixed_first

    if n_black_keys <= 0:
        return

    # Estimate number of octaves
    # The first octave may be incomplete so remove it keys
    # A octave has 5 black keys or less (a incomplete octave at the end of image)
    # Add the first octave (add 1)
    n_octaves = math.ceil((n_black_keys - first_octave_keys) / 5) + 1

    #                                          VVV The middle of image
    # | A0 B0 | C1  -  B1 | C2  -  B2 | C3  -  B3 | C4  -  B4 | C5  -  B5 | C6  -  B6 | C7  -  B7 | C8 |
    # |  A0#  | C#1 - A#1 | C#2 - A#2 | C#3 - A#3 | C#4 - A#4 | C#5 - A#5 | C#6 - A#6 | C#7 - A#7 |    |
    first_octave = 4 - int(n_octaves / 2)

    current_octave = first_octave

    for key in kb.mask.bkeys:

        key.octave = current_octave
        if int(key.id) >= int(KeyValue.KEY_As):
            current_octave += 1

    current_octave = first_octave

    for key in kb.mask.wkeys:

        key.octave = current_octave
        if int(key.id) >= int(KeyValue.KEY_B):
            current_octave += 1
