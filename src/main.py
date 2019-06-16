#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

from pianovision.keyboard.bgextractor import BackgroundExtractor
from pianovision.keyboard.kbextractor import KeyboardExtractor

from utils.resourceloader import ResourceLoader


def main():

    data = ResourceLoader("flamingo")

    background = cv2.imread("background.bmp")

    if background is None:
        background = BackgroundExtractor(data.videoname, sample=110).run()
        cv2.imwrite("background.bmp", background)

    # sound_analysis = SoundAnalyser(data.audioname)
    # sound_analysis.plot()

    cv2.imshow("background", background)
    cv2.imshow("keyboard", KeyboardExtractor(background).run())
    cv2.waitKey()

    def behaviour(frame):
        cv2.imshow("teste", frame)
        return not (cv2.waitKey(5) & 0xFF == ord('q'))

    # player = VideoHandler(data.videoname, behaviour)
    # player.run()

    exit(0)


if __name__ == "__main__":
    main()
