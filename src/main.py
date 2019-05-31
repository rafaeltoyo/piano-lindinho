#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from utils.pathbuilder import PathBuilder
import time

class Handler:

    def __init__(self, cap):
        self.cap = cap

        self.currentFrame = None
        self.lastFrame = None
        self.deltaFrame = None

    def first(self):
        # Get the data from the first frame:
        gray = cv2.cvtColor(self.currentFrame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 250, 500)

        cv2.imshow('First frame', edges)
        cv2.waitKey(0)
        lines = cv2.HoughLines(edges, 1, np.pi / 360, 200)
        #for line in lines:
        #    rho, theta = line[0]
        #    a = np.cos(theta)
        #    b = np.sin(theta)
        #    x0 = a * rho
        #    y0 = b * rho
        #    x1 = int(x0 + 1000 * (-b))
        #    y1 = int(y0 + 1000 * (a))
        #    x2 = int(x0 - 1000 * (-b))
        #    y2 = int(y0 - 1000 * (a))
        #    cv2.line(self.currentFrame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow('First frame', self.currentFrame)
        cv2.waitKey(0)

        self.lastFrame = self.currentFrame

    def process(self):
        blurred = cv2.blur(self.currentFrame, (3, 3))
        delta = cv2.absdiff(blurred, self.lastFrame)
        delta = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY)
        threshFrame = cv2.threshold(delta, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #threshFrame = cv2.dilate(threshFrame, None, iterations=2)
        cv2.imshow('Frame', cv2.UMat(threshFrame))
        self.lastFrame = blurred

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.currentFrame = frame

                if self.lastFrame is None:
                    self.first()
                else:
                    self.process()

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

    def end(self):
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(str(PathBuilder().miscdir("flamingo.mp4")))
    handler = Handler(cap)

    # Check if camera opened successfully
    if not handler.cap.isOpened():
        print("Error opening video stream or file")
        exit(1)

    try:
        handler.run()
    finally:
        handler.end()

    exit(0)


if __name__ == "__main__":
    main()
