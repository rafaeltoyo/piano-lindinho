#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from utils.videohandler import VideoHandler


class BackgroundExtractor(VideoHandler):

    def __init__(self, filename: str, sample: int = 100):
        """
        Extract background from video
        :param filename: Video filename
        :param sample: Number of frames
        """

        super().__init__(filename, self.save_frame)

        self.__buffer = []

        self.__nframes = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__step = int(self.__nframes / sample)

        self.__iter = 0

    def _setup(self):
        """
        Change video stream pointer position
        :return: NA
        """
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.__iter)

    def run(self) -> np.ndarray:
        """
        Process the video
        :return: NA
        """

        super().run()
        buffer = np.array(self.__buffer)
        return np.median(buffer, axis=0).astype('uint8')

    def save_frame(self, frame: np.ndarray) -> bool:
        """
        Save current frame
        :param frame: current frame
        :return: keep processing
        """

        # Multiplicar as componentes R, G e B invertidas
        gsimg = (255 - frame).prod(axis=2).astype('float32')

        # Multiplicar pelo menor valor entre R, G e B invertidos
        gsimg *= np.amin((255 - frame), axis=2)

        # Normalize
        gsimg = ((gsimg * 255) / gsimg.max()).astype('uint8')

        self.__buffer.append(gsimg)
        self.__iter += self.__step
        return self.__iter < self.__nframes
