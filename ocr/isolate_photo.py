"""
OpenCV Filters for ID Card photo isolation and blurring
"""

import cv2
import numpy as np

# Setup variables:
IMG_NAME = 'id_front.jpeg'
IMG_INP = 'output_data/' + IMG_NAME  # Grab from output of `card_crop` preprocessing

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


im_orig = cv2.imread(IMG_INP)  # Load image
im = im_orig.copy()  # Copy image for processing / debugging
im_result = im_orig.copy()  # Copy image for final result

gray = cv2.cvtColor(im_orig, cv2.COLOR_BGR2GRAY)  # Create gray image


# Detect faces:
face_coords = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in face_coords:  # Plot faces on display photo
    im = cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)

# Find areas of selected regions
areas = face_coords[:, 2] * face_coords[:, 3]
main_face_c = face_coords[np.argmax(areas)]  # Select largest as main face

# Mark the largest rectangle:
cv2.drawMarker(im, tuple(main_face_c[0:2]), (0, 255, 0), cv2.MARKER_CROSS, 30)

# Grow to fit whole head:
vert = int(1.3 * main_face_c[2])
horiz = int(1.1 * main_face_c[3])
large_face_c = np.array([
    main_face_c[0] - int((horiz - main_face_c[2]) / 2),
    main_face_c[1] - int((vert - main_face_c[3]) / 2),
    horiz,
    vert,
])

# Plot rectangle for enlarged face box:
im = cv2.rectangle(
    im,
    tuple(large_face_c[0:2]),  # Select corner coordinates
    tuple(i[0] + i[1] for i in zip(large_face_c[0:2], large_face_c[2:4])),  # Add width/height to x/y to find other corner
    (0,0,255),
    2
)

# Select enlarged face box pixels:
large_face = im_orig[large_face_c[1]:large_face_c[1]+large_face_c[3], large_face_c[0]:large_face_c[0]+large_face_c[2]]
cv2.imshow('face', large_face)  # Show face


# Obfuscate image
im_result = cv2.blur(im_result, (25,25))
cv2.imshow("Obfuscated", im_result)

# Add back enlarged unblurred face:
im_result[large_face_c[1]:large_face_c[1]+large_face_c[3], large_face_c[0]:large_face_c[0]+large_face_c[2]] = large_face
cv2.imshow("Obfuscated with face", im_result)


cv2.imshow('img',im)
cv2.waitKey(0)
cv2.destroyAllWindows()
