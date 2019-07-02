#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def keyboardThresholding(image: np.ndarray) -> np.ndarray:
    """
    Keyboard binarizer
    :param image: Keyboard image
    """

    gsimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (h, w) = gsimg.shape

    # Tunning Otsu method: create a buffer with black pixels
    buffer = np.zeros((2 * h, w)).astype('uint8')
    buffer[:h, :] = gsimg

    # Thresholding with Otsu method
    ret, thresh = cv2.threshold(buffer, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = thresh[:h, :]

    # Invert values because we need mask of black keys
    thresh = 255 - thresh

    # Creating a kernel for Morphological Transformations
    kernel_size = min(int(image.shape[1] * 0.005), 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Closing roles inside keys
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Denoising
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return thresh


def keyboardHorizontalDivision(thresh: np.ndarray) -> int:
    """
    Find height of black keys
    :param thresh: Binarized keyboard image
    """

    (h, w) = thresh.shape
    maxsize = int((h ** 2 + w ** 2) ** 0.5)

    # Needs edges to apply Hough
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

    # Hough to get vertical and horizontal lines
    lines = cv2.HoughLines(edges, 1, np.pi / 2, int(w * 0.08))

    # Nothing to do
    if lines is None:
        return h

    for line in lines:
        # Get rho and theta
        rho, theta = line[0]

        # Ignore vertical lines
        if not (np.pi * 3 / 4 >= (theta % np.pi) >= np.pi * 1 / 4):
            continue

        # Polar -> Rect
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho

        x1, y1 = int(x0 + maxsize * (-b)), int(y0 + maxsize * (a))
        x2, y2 = int(x0 - maxsize * (-b)), int(y0 - maxsize * (a))

        # Division found
        return max(0, min(h, y1, y2))

    return h  # Nothing found :(


def keyboardBlackKeys(thresh: np.ndarray, tolerance: float = 0.5, vlimit: int = None):
    """
    Create a mask for black keys
    :param thresh: Binarized keyboard image
    :param tolerance: Error between blob and estimate rectangle
    :param vlimit: Vertical crop (height of black keys)
    :return:
    """

    if vlimit is None:
        vlimit = keyboardHorizontalDivision(thresh)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    median_area = np.median(np.array([cv2.contourArea(cnt) for cnt in contours]))

    for cnt in contours:

        # Fit a rectangle on the blob
        rect = cv2.minAreaRect(cnt)
        box = np.intp(cv2.boxPoints(rect))

        # Ignore small blobs
        blob_area = cv2.contourArea(box)
        if median_area > blob_area and (median_area - blob_area) / median_area > 0.1:
            continue

        # Create a mask of the blob with original image shape
        blob = np.zeros(thresh.shape).astype(int)
        cv2.fillPoly(blob, [cnt], 1)

        # Create a mask of the rectangle with original image shape
        rect = np.zeros(thresh.shape).astype(int)
        cv2.fillPoly(rect, [box], 1)

        # Compute a error based in difference between blob mask and rectangle mask
        error = float((rect * (1 - blob)).sum())

        # Ignore wrong rectangles
        if error / blob.sum() > tolerance:
            continue

        yield box
