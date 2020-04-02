"""
OpenCV Filters for ID Card detection and selection/cropping.

based on work for a playing card detector: https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector
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
res_img = im.copy()  # Copy image for later result

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 15, 225)

# Display filtered image
cv2.imshow("ID Card with filters", edged)

# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cardCnt = None
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
            cardCnt = np.squeeze(approx, axis=1)
            break

cv2.polylines(im, [cardCnt], True, (0, 255, 255), 10)


def card_point(num):
    return int(cardCnt[num][0]), int(cardCnt[num][1])


# Draw box around card, from corner points
for i in [(0,1), (1,2), (2,3), (3,0)]:
    cv2.line(im, card_point(i[0]), card_point(i[1]), (255, 0, 0), 3)

# Perspective warp ID using corner points
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
M = 600
RAT = 1.5
pts2 = np.float32([[0, 0], [0, M*RAT], [M, M*RAT], [M, 0]])
trans = cv2.getPerspectiveTransform(np.float32(cardCnt), pts2)
dst = cv2.warpPerspective(res_img, trans, (M, int(M * RAT)))
cv2.imshow("Warped ID Card from contours", dst)
cv2.imwrite(IMG_OTP, dst)

cv2.imshow("ID Card with contours", im)

cv2.waitKey(0)
cv2.destroyAllWindows()

