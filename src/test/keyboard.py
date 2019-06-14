#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from utils.pathbuilder import PathBuilder


def main():
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    video_name = "believer.mp4"
    cap = cv2.VideoCapture(str(PathBuilder().miscdir(video_name)))

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        exit(1)

    ttl_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample = 200

    buffer = []

    try:
        for i in range(0, ttl_frames, int(ttl_frames/sample)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read(i)

            if not ret:
                break
            gs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            buffer.append(gs_frame)

        buffer = np.array(buffer)
        img = np.median(buffer, axis=0).astype('uint8')
        print(img.shape)

        cv2.imshow("mediana", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        cv2.imshow(video_name.replace(".mp4", "-median.bmp"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        cv2.waitKey(0)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    exit(0)


if __name__ == "__main__":
    main()
