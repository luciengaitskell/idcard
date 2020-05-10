"""
Card isolation example.

### USES OUTPUT from `example/cselect.py` ###
"""
import cv2
import os

# Example configuration for HAAR cascade locations
os.environ['OPENCV_HAAR'] = "/usr/local/Cellar/opencv/4.3.0/share/opencv4/haarcascades/"

from idcard.front import IDIsolate


# Image locations and names:
IMG_NAME = 'id_front.jpeg'
IMG_INP = 'output_data/' + IMG_NAME  # Grab from output of `card_crop` preprocessing

# Open source
res = cv2.imread(IMG_INP)

# Image processing
isolate = IDIsolate(res)  # Load image into isolator
res = isolate.process()

# Display
cv2.imshow('img', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
