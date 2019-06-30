#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from .data import Keyboard, KeyboardMask


class KeyboardMapping:

    def __init__(self, kb: Keyboard):
        """

        :param mask: Black keys mask
        """

        self.__mask = mask
        pass

    def parse_key(self, x1, x2, y1, y2):

        pass
