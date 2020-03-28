"""
OpenCV Filters for ID Card detection.

based on work for a playing card detector: https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector
"""

import cv2
import numpy as np

# Setup variables:
IMG = 'sample_data/id_front.jpeg'
BKG_THRESH = 70  # background black threshold

CARD_MIN_AREA = 25000

im = cv2.imread(IMG)  # Load image
res_img = im.copy()  # Copy image for later result

# Basic initial conversions
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # convert to grayscale
blur = cv2.GaussianBlur(gray, (5, 5), 0)  # apply blur

# The best threshold level depends on the ambient lighting conditions.
# For bright lighting, a high threshold must be used to isolate the cards
# from the background. For dim lighting, a low threshold must be used.
# To make the card detector independent of lighting conditions, the
# following adaptive threshold method is used.
#
# A background pixel in the center top of the image is sampled to determine
# its intensity. The adaptive threshold is set at 50 (THRESH_ADDER) higher
# than that. This allows the threshold to adapt to the lighting conditions.
img_w, img_h = np.shape(im)[:2]
bkg_level = gray[int(img_h / 100)][int(img_w / 2)]
thresh_level = bkg_level + BKG_THRESH

# Apply threshold to blurred image:
retval, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)

# Select contours on image
cnts, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Display filtered image
cv2.imshow("ID Card with filters", thresh)

# Display original image with contours:
#cv2.drawContours(im, cnts, -1, (255,0,0), 3)

if len(cnts) == 0:  # If there are no contours, do nothing
    print("NONE")
else:  # otherwise continue:

    # Sort contour indexes by the area of the contour represented
    index_sort = sorted(range(len(cnts)), key=lambda i: cv2.contourArea(cnts[i]), reverse=True)

    # initialize empty sorted contour and hierarchy lists
    cnts_sort = []
    hier_sort = []

    # Fill empty lists with sorted contour and sorted hierarchy. Now,
    # the indices of the contour list still correspond with those of
    # the hierarchy list. The hierarchy array can be used to check if
    # the contours have parents or not.
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    # Determine which of the contours are cards by applying the
    # following criteria: 1) bigger area than the minimum card size, 2) have no parents,
    # and 3) have four corners

    # Resulting card data
    card_approx = None
    card_idx = None
    card_contour = None

    for i in range(len(cnts_sort)):

        # Calculate area, perimeter, and approximate polygon for contour
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i], True)
        approx = cv2.approxPolyDP(cnts_sort[i], 0.05 * peri, True)

        # Clarify contour filters from above
        if ((size > CARD_MIN_AREA)
                and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            card_approx = approx
            card_idx = i
            break

    if card_idx is not None:
        # Draw the selected contour:
        cv2.drawContours(im, cnts_sort, card_idx, (255, 0, 0), 3)

        # Select corner points and remove extra dimensions
        corner_pts = np.float32(card_approx)
        corner_pts = np.squeeze(corner_pts, axis=1)

        def card_point(num):
            return int(corner_pts[num][0]), int(corner_pts[num][1])

        # Draw box around card, from corner points
        for i in [(0,1), (1,2), (2,3), (3,0)]:
            cv2.line(im, card_point(i[0]), card_point(i[1]), (255, 0, 0), 3)

        # Show non-rotated bounding rectangle
        x,y,w,h = cv2.boundingRect(cnts_sort[card_idx])
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Perspective warp ID using corner points
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
        M = 600
        RAT = 1.5
        pts2 = np.float32([[M, 0], [0, 0], [0, M*RAT], [M, M*RAT]])
        trans = cv2.getPerspectiveTransform(corner_pts, pts2)
        dst = cv2.warpPerspective(res_img, trans, (M, int(M * RAT)))
        cv2.imshow("Warped ID Card from contours", dst)

cv2.imshow("ID Card with contours", im)

# TODO: Select contour of card
# TODO: Select corners

cv2.waitKey(0)
cv2.destroyAllWindows()

