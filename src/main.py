#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from pianovision.keyboard import KeyboardHandler
from pianovision.sound import SoundAnalyser

from utils.videohandler import VideoHandler
from utils.resourceloader import ResourceLoader


def main():

    data = ResourceLoader("zenzenzense")

    kb_handler = KeyboardHandler(data)

    # sound_analysis = SoundAnalyser(data.audioname)
    # sound_analysis.plot()

    def behaviour(frame: np.ndarray) -> bool:
        cv2.imshow("teste", frame)
        return not (cv2.waitKey(5) & 0xFF == ord('q'))

    player = VideoHandler(data.videoname, behaviour)
    player.run()

    exit(0)


if __name__ == "__main__":
    main()
