"""
OpenCV Filters for ID Card detection and selection/cropping.

based on work for a playing card detector: https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector
"""

import cv2
import imutils
import numpy as np
from scipy.spatial import cKDTree

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
    dilated = cv2.dilate(edged, kernel)
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
dilated = cv2.dilate(only_cnt, kernel)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
erosion = cv2.erode(dilated, kernel1, iterations = 1)
only_cnt = erosion
cv2.imshow("ID Card with filters and only one contour", only_cnt)
#cv2.polylines(im, [cardCnt], True, (0, 255, 255), 10)


minLineLength = 500
maxLineGap = 400

lines = cv2.HoughLines(
    only_cnt,
    rho=1,
    theta=1*np.pi/180,
    threshold=300)
print(len(lines))
for l in lines:
    rho, theta = l[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 2000*(-b))
    y1 = int(y0 + 2000*(a))
    x2 = int(x0 - 2000*(-b))
    y2 = int(y0 - 2000*(a))

    cv2.line(im,(x1,y1),(x2,y2),(0,0,255),2)

# Find intersections from HoughLines
# TODO: SLIM DOWN CODE, ADD COMMENTS
# https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
from collections import defaultdict
def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


segmented = segment_by_angle_kmeans(lines)

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections

intersections = segmented_intersections(segmented)

# Remove extra [] around elements (e.g. [[[a,b]], [c,d]]] -> [[a,b], [c,d]]
intersections = np.array(intersections)

# Finding close points:
tree = cKDTree(intersections)
rows_to_fuse = tree.query_pairs(r=30)  # Get indexes of close points
print(rows_to_fuse)


# https://stackoverflow.com/questions/36985185/fast-fuse-of-close-points-in-a-numpy-2d-vectorized
idxs_to_remove = []
for row in rows_to_fuse:
    els = []
    for idx in row:
        els.append(intersections[idx])
    min_idx = min(row)
    for excess in row:
        if not excess == min_idx:
            idxs_to_remove.append(excess)
    m = np.mean(els, axis=0)
    intersections[min_idx] = m

intersections = np.delete(intersections, idxs_to_remove, axis=0)

for item in intersections:
    cv2.drawMarker(im, (item[0], item[1]),(0,0,255), markerType=cv2.MARKER_TILTED_CROSS,
                   markerSize=40, thickness=2, line_type=cv2.LINE_AA)

cv2.imshow("ID Card with contours", im)

x_sort, y_sort = np.argsort(np.transpose(intersections))

# Sort corners
for i in range(4):  # Iterate through all values in x_sort and y_sort
    x = np.squeeze(np.argwhere(x_sort == i))  # Get position in x sorted array
    y = np.squeeze(np.argwhere(y_sort == i))  # Get position in y sorted array

    if x > 1:  # On right side (based on position in array)
        if y > 1:  # On lower side
            bottomRight = i
        else:  # On upper side
            topRight = i
    else:  # On left side
        if y > 1:  # On lower side
            bottomLeft = i
        else:  # On upper side
            topLeft = i

#print(intersections)

# Perspective warp ID using corner points
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
M = 600
RAT = 1.5
dstPts = np.float32([[0, 0], [M, 0], [0, M*RAT], [M, M*RAT]])  # Proportional size of card
srcPts = np.float32(intersections)[[topLeft, topRight, bottomLeft, bottomRight]]  # Convert and sort source points

# Warp perspective:
trans, status = cv2.findHomography(srcPts, dstPts)
dst = cv2.warpPerspective(res_img, trans, (M, int(M * RAT)))
cv2.imshow("Warped ID Card from contours", dst)
cv2.imwrite(IMG_OTP, dst)

cv2.waitKey(0)
cv2.destroyAllWindows()

