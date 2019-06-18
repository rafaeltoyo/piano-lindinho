#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from utils.pathbuilder import PathBuilder


class ResourceLoader:

    def __init__(self, name: str):

        self.path = PathBuilder()
        self.__name = name

        self.__video_file = self.path.miscdir(name + ".mp4")
        self.__audio_file = self.path.miscdir(name + ".wav")

        if not self.__audio_file.exists():
            os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(self.videoname, self.audioname))

        self.__images = {}

    @property
    def videoname(self) -> str:
        return str(self.__video_file)

    @property
    def audioname(self) -> str:
        return str(self.__audio_file)

    def create(self, sufix: str):
        self.__images[sufix] = self.path.miscdir("{}-{}.bmp".format(self.__name, sufix))

    def __getitem__(self, item: str) -> Path:
        if item not in self.__images.keys():
            raise IndexError
        return (self.__images[item])
