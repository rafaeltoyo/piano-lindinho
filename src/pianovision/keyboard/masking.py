#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .data import Keyboard, KeyboardMask
from .func.binarize import keyboardHorizontalDivision, keyboardBlackKeys, keyboardThresholding


class KeyboardMasking:

    def __init__(self, keyboard: Keyboard):

        self.kb = keyboard

        if self.kb.cropped is None:
            raise RuntimeError("Masking need cropped image of keyboard")

        if self.kb.mask is None:
            self.kb.mask = KeyboardMask(self.kb.cropped)

        self.kb.mask.thresh = keyboardThresholding(self.kb.cropped)
        self.kb.mask.vlimit = keyboardHorizontalDivision(self.kb.mask.thresh)
        self.kb.mask.mask = keyboardBlackKeys(self.kb.mask.thresh, vlimit=self.kb.mask.vlimit)
