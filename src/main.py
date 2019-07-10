#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from pianovision.keyboard import KeyboardHandler
from pianovision.video.base import MotionHandler

from utils.videohandler import VideoHandler
from utils.resourceloader import ResourceLoader


def main():
    np.seterr(divide='ignore', invalid='ignore')

    data = ResourceLoader("flamingo")

    MotionHandler(data, debug=True)

    exit(0)

    kb_handler = KeyboardHandler(data)
    kb_handler.print()

    # sound_analysis = SoundAnalyser(data.audioname)
    # sound_analysis.plot()

    def behaviour(frame: np.ndarray) -> bool:
        cv2.imshow("video", frame)
        return not (cv2.waitKey(5) & 0xFF == ord('q'))

    player = VideoHandler(data.videoname, behaviour)
    player.run()

    exit(0)


if __name__ ==  "__main__":
    main()
