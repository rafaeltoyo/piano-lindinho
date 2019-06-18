#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path


class PathBuilder:

    def __init__(self):
        """
        Build absolute project path.
        """

        # First script called
        self.invoker = Path(os.getcwd()) / Path(sys.argv[0])

        # Project folder (root)
        sufix = 'src' + os.sep + str(__name__).replace('.', os.sep) + '.py'
        self.project = Path(__file__.replace(sufix, ''))

    def miscdir(self, file: str) -> Path:
        return self.project / 'misc' / file

    def __getimgdir(self, dir) -> Path:
        return self.project / 'imgs' / dir

    def inputdir(self, img: str) -> str:
        return str(self.__getimgdir('input') / img).replace('/', '\\')

    def outputdir(self, img: str) -> str:
        return str(self.__getimgdir('output') / img).replace('/', '\\')
