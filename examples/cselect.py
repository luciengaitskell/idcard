""" Card selection example. """
import cv2
import os

# Example configuration for HAAR cascade locations
os.environ['OPENCV_HAAR'] = "/usr/local/Cellar/opencv/4.3.0/share/opencv4/haarcascades/"

from idcard.front import IDSelect


# Image locations and names:
IMG_NAME = 'id_front.jpeg'
IMG_INP = 'sample_data/' + IMG_NAME
IMG_OTP = 'output_data/' + IMG_NAME

# Open source
im = cv2.imread(IMG_INP)  # Load image

# Image processing
sample_card = IDSelect(im)
sample_card.process()

# Save
cv2.imwrite(IMG_OTP, sample_card.res_im)

# Display
cv2.imshow("Final", sample_card.res_im)
cv2.waitKey(0)
cv2.destroyAllWindows()
