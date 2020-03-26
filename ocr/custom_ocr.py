"""
OpenCV Filters for ID Card detection.

based on work for a playing card detector: https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector
"""

import cv2
import numpy as np

# Setup variables:
IMG = 'sample_data/id_front.jpeg'
BKG_THRESH = 60  # background black threshold

im = cv2.imread(IMG)  # Load image

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
cv2.drawContours(im, cnts, -1, (255,0,0), 3)
cv2.imshow("ID Card with contours", im)

# TODO: Select contour of card
# TODO: Select corners

cv2.waitKey(0)
cv2.destroyAllWindows()

