import numpy as np
import cv2
from utils.pathbuilder import PathBuilder


cap = cv2.VideoCapture(str(PathBuilder().miscdir('flamingo.mp4')))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('frame', cv2.bitwise_and(frame, frame, mask=fgmask))
    k = cv2.waitKey(30) & 0xff
    if not ret or k == 27:
        break

cap.release()
cv2.destroyAllWindows()
