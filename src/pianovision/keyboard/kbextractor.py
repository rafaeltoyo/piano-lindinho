#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np


class KeyboardDetector:

    def __init__(self, image: np.ndarray):
        """
        Keyboard Extractor
        :param image: Original image
        """

        # TODO implements

        self.cropped = image[550:804, :]
        # self.cropped = image[524:726, :]


class KeyboardBinarizer:

    def __init__(self, image: np.ndarray):
        """
        Keyboard binarizer
        :param image: Keyboard image
        """

        gsimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Binarização por Otsu
        ret, thresh = cv2.threshold(gsimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Queremos teclas pretas, logo devemos inverter a máscara
        # thresh = 255 - thresh

        # Criar um kernel para a morfologia
        kernel_size = min(int(image.shape[1] * 0.005), 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Morfologia para fechar buracos
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Morfologia para tirar ruído
        self.thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
