"""
Second Version of OpenCV Filters for ID Card detection and selection/cropping.

Code snippets and inspiration from: https://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/
"""

import cv2
import imutils
import numpy as np

# Setup variables:
IMG_NAME = 'id_front.jpeg'
IMG_INP = 'sample_data/' + IMG_NAME
IMG_OTP = 'output_data/' + IMG_NAME
BKG_THRESH = 70  # background black threshold

CARD_MIN_AREA = 25000

im = cv2.imread(IMG_INP)  # Load image
im_mod = im.copy()  # Copy image for later result

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 15, 225)

# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None
# ensure that at least one contour was found
if len(cnts) > 0:
    # sort the contours according to their size in
    # descending order
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # loop over the sorted contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        print(cv2.contourArea(c))
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points,
        # then we can assume we have found the paper
        if len(approx) == 4:
            docCnt = np.squeeze(approx, axis=1)
            break


cv2.polylines(im_mod,[docCnt],True,(0,255,255), 10)

cv2.imshow("Poly", im_mod)

cv2.imshow("Edge", edged)

cv2.waitKey(0)
cv2.destroyAllWindows()
