#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from utils.resourceloader import ResourceLoader

from .bgextractor import BackgroundExtractor
from .bkextractor import BlackKeysDetector, FirstBlackKeyRecognition
from .kbextractor import KeyboardBinarizer, KeyboardDetector


class KeyboardHandler:

    def __init__(self, resource: ResourceLoader):
        """
        Main class of keyboard module
        :param resource: video
        """

        self.__resource = resource

        resource.create("background")

        with resource["background"] as bgimg:
            if not bgimg.exists():
                background = BackgroundExtractor(resource.videoname).run()
                cv2.imwrite(str(bgimg), background)
            else:
                background = cv2.imread(str(bgimg), 0)

        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

        kb_detector = KeyboardDetector(background)

        kb_binarizer = KeyboardBinarizer(kb_detector.cropped)

        black_keys = BlackKeysDetector(kb_binarizer.thresh)

        cv2.imshow("teste", background)
        cv2.waitKey()
        cv2.imshow("teste", kb_detector.cropped)
        cv2.waitKey()
        cv2.imshow("teste", kb_binarizer.thresh)
        cv2.waitKey()
        cv2.imshow("teste", black_keys.edges)
        cv2.waitKey()
        cv2.imshow("teste", (black_keys.mask * 255).astype('uint8'))
        cv2.waitKey()

        first_black_keys = FirstBlackKeyRecognition(black_keys)

        print(first_black_keys.first_key_label)
