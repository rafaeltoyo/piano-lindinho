import cv2
import numpy as np

cap = cv2.VideoCapture('alice.mp4')
ret, current_frame = cap.read()
previous_frame = current_frame

previous_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
previous_frame_hsv =  cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

cv2.imshow('frame_test', current_frame)
cv2.waitKey(0)

# min_white_key_val = int(input('type in what the minimum white key value is: '))
# max_white_key_sat = int(input('type in what the maximum white key saturation is: '))
min_white_key_val = 180
max_white_key_sat = 7

red_difference = 50
red_threshold = 160

previous_hand_mask = np.zeros(previous_frame_gray.shape)



while(cap.isOpened()):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    current_frame_gray = cv2.GaussianBlur(current_frame_gray, (11,11), 3)

    current_frame_hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

    current_frame_hsv = cv2.GaussianBlur(current_frame_hsv, (11,11),3)

    current_frame_sat = current_frame_hsv[:,:,1]

    frame_diff_l = cv2.absdiff(current_frame_gray, previous_frame_gray)

    frame_diff_s = cv2.absdiff(current_frame_sat, previous_frame_hsv[:,:,1])

    white_key_mask = np.zeros(frame_diff_l.shape)

    white_key_mask[current_frame_hsv[:,:,1] < max_white_key_sat] = 1
    white_key_mask[previous_frame_hsv[:,:,1] < max_white_key_sat] = 1
    white_key_mask[current_frame_hsv[:,:,2] > min_white_key_val] = 1
    white_key_mask[previous_frame_hsv[:,:,2] > min_white_key_val] = 1

    hand_mask = np.zeros(frame_diff_l.shape)
    red_diff_frame = current_frame[:,:,2]*2  - current_frame[:,:,1] - current_frame[:,:,0]
    red_diff_frame[current_frame[:,:,2] < red_threshold] = 0
    red_diff_frame[red_diff_frame > 200] = 0
    hand_mask[red_diff_frame > red_difference] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    im = np.zeros((100, 100), dtype=np.uint8)
    im[50:, 50:] = 255

    mean, std_dev = cv2.meanStdDev(frame_diff_l)

    frame_diff_l[frame_diff_l < 2*std_dev] = 0


    hand_mask = cv2.dilate(hand_mask, kernel, iterations=2)

    frame_diff_l[previous_hand_mask == 1] = frame_diff_l[previous_hand_mask == 1]/100
    frame_diff_l[white_key_mask == 1] = frame_diff_l[white_key_mask == 1]*20
    frame_diff_l[white_key_mask == 0] = 0
    frame_diff_l[hand_mask == 1] = 0

    frame_diff_l = cv2.blur(frame_diff_l, (13,13))
    frame_diff_l = cv2.normalize(frame_diff_l, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    cv2.imshow('frame diff saturation',frame_diff_l)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    previous_frame_gray = current_frame_gray.copy()
    previous_frame_hsv = current_frame_hsv.copy()
    previous_hand_mask = hand_mask
    ret, current_frame = cap.read()


cap.release()
cv2.destroyAllWindows()