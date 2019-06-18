#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

from pianovision.keyboard import KeyboardHandler

from utils.resourceloader import ResourceLoader


def main():

    data = ResourceLoader("flamingo")

    kb_handler = KeyboardHandler(data)

    # sound_analysis = SoundAnalyser(data.audioname)
    # sound_analysis.plot()

    def behaviour(frame):
        cv2.imshow("teste", frame)
        return not (cv2.waitKey(5) & 0xFF == ord('q'))

    # player = VideoHandler(data.videoname, behaviour)
    # player.run()

    exit(0)


if __name__ == "__main__":
    main()
