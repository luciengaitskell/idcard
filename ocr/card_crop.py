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
edged = cv2.Canny(blurred, 30, 225)

# Display filtered image
cv2.imshow("ID Card with filters", edged)

# find contours in the edge map, then initialize
# the contour that corresponds to the document

if True:
    # https://stackoverflow.com/questions/49993616/multiple-line-detection-in-houghlinesp-opencv-function
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(edged.copy(), kernel)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    erosion = cv2.erode(dilated, kernel1, iterations = 1)
    cnt_edited = erosion
elif False:
    cnt_edited = cv2.blur(edged.copy(), (12, 12))
else:
    cnt_edited = edged.copy()
cv2.imshow("ID Card with filters 2", cnt_edited)


cnts = cv2.findContours(cnt_edited, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cardCnt = None
only_cnt = np.zeros(edged.shape, dtype=np.uint8)

# ensure that at least one contour was found
if len(cnts) > 0:
    # sort the contours according to their size in
    # descending order
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # loop over the sorted contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points,
        # then we can assume we have found the paper
        if len(approx) == 4:
            cv2.drawContours(only_cnt, [c], 0, 255, 4)
            cardCnt = np.squeeze(approx, axis=1)
        break


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilated = cv2.dilate(only_cnt.copy(), kernel)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
erosion = cv2.erode(dilated, kernel1, iterations = 1)
only_cnt = erosion
cv2.imshow("ID Card with filters and only one contour", only_cnt)
#cv2.polylines(im, [cardCnt], True, (0, 255, 255), 10)


minLineLength = 500
maxLineGap = 400

lines = cv2.HoughLinesP(
    only_cnt,
    rho=10,
    theta=0.01*np.pi/180,
    threshold=800,  # TODO: THIS IS making the operation very slow -- and needs a lot of refining ... find another way
    minLineLength=minLineLength,
    maxLineGap=maxLineGap)
print(len(lines))
for i in range(len(lines)):
    l = lines[i]
    for x1,y1,x2,y2 in l:
        color = [0,] * 3
        color[i%3] = 255
        cv2.line(im,(x1,y1),(x2,y2),color,10)


def card_point(num):
    return int(cardCnt[num][0]), int(cardCnt[num][1])


# Perspective warp ID using corner points
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
M = 600
RAT = 1.5
pts2 = np.float32([[0, 0], [0, M*RAT], [M, M*RAT], [M, 0]])
trans, status = cv2.findHomography(np.float32(cardCnt), pts2)
dst = cv2.warpPerspective(res_img, trans, (M, int(M * RAT)))
cv2.imshow("Warped ID Card from contours", dst)
cv2.imwrite(IMG_OTP, dst)

cv2.imshow("ID Card with contours", im)

cv2.waitKey(0)
cv2.destroyAllWindows()

