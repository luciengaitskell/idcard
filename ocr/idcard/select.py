"""
Utility to select and crop an ID Card in a photo
"""

import cv2
import imutils
import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict


class IDSelect:
    def __init__(self, source: np.array):
        self.orig_im = source
        self.res_im = source.copy()  # Copy image for later result

    def _get_edged(self):
        """ Use canny filter on black and white blurred image to detected edges. """
        gray = cv2.cvtColor(self.orig_im, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, 30, 225)

    def _process_edged(self, edged):
        """ Dilate and erode edged image to improve detection. """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated = cv2.dilate(edged, kernel)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        erosion = cv2.erode(dilated, kernel1, iterations=1)
        return erosion

    @staticmethod
    def _get_contours(edged):
        """ Get contours from edged image. """
        cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return imutils.grab_contours(cnts)

    @staticmethod
    def _find_card_contour(edged, cnts):
        """ Select biggest contour, which will be ID card. """
        single_cnt = np.zeros(edged.shape, dtype=np.uint8)  # Save single, final contour of card

        # ensure that at least one contour was found
        if len(cnts) > 0:
            # sort the contours according to their size in
            # descending order
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            # loop over the sorted contours
            for c in cnts:
                # approximate the contour
                peri = cv2.arcLength(c, True)  # Perimeter
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)  # Bounding box
                # if our approximated contour has four points,
                # then we can assume we have found the card
                if len(approx) == 4:
                    cv2.drawContours(single_cnt, [c], 0, 255, 4)
                break
        return single_cnt

    @staticmethod
    def _process_card_contour(id_contour):
        """ Dilate and erode id_contour for better HoughLines detection """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(id_contour, kernel)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        erosion = cv2.erode(dilated, kernel1, iterations=1)
        return erosion

    @staticmethod
    def _get_edge_lines(id_contour):
        return cv2.HoughLines(
            id_contour,
            rho=1,
            theta=1 * np.pi / 180,
            threshold=200
        )

    @staticmethod
    def _segment_lines_by_angle_kmeans(lines, k=2, **kwargs):
        """ Groups lines based on angle with k-means.

        Uses k-means on the coordinates of the angle on the unit circle
        to segment `k` angles inside `lines`.

        See: https://stackoverflow.com/a/46572063/3954632
        """

        # Define criteria = (type, max_iter, epsilon)
        default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
        criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
        flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
        attempts = kwargs.get('attempts', 10)

        # returns angles in [0, pi] in radians
        angles = np.array([line[0][1] for line in lines])
        # multiply the angles by two and find coordinates of that angle
        pts = np.array([[np.cos(2 * angle), np.sin(2 * angle)]
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

    @staticmethod
    def _line_intersection(line1, line2):
        """ Finds the intersection of two lines given in Hesse normal form.

        Returns closest integer pixel locations.
        See: https://stackoverflow.com/a/383527/5087436
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

    @classmethod
    def _segmented_intersections(cls, seg_lines):
        """ Finds the intersections between groups of lines. """

        intersections = []
        for i, group in enumerate(seg_lines[:-1]):
            for next_group in seg_lines[i + 1:]:
                for line1 in group:
                    for line2 in next_group:
                        intersections.append(cls._line_intersection(line1, line2))

        intersections = np.array(intersections)
        return intersections

    @staticmethod
    def _fuse_close_points(intersections):
        """ Fuse close points to delete duplicate intersections.

        See: https://stackoverflow.com/a/36989440/3954632
        """
        # Finding close points:
        tree = cKDTree(intersections)
        rows_to_fuse = tree.query_pairs(r=30)  # Get set of index tuples of close points

        idxs_to_remove = []  # Save indexes to remove
        for row in rows_to_fuse:  # Each tuple of close points
            close_values = []  # Collect all elements
            for idx in row:
                close_values.append(intersections[idx])

            min_idx = min(row)  # Use smallest index for final value

            for excess in row:  # Save indexes of extra elements
                if not excess == min_idx:
                    idxs_to_remove.append(excess)

            m = np.mean(close_values, axis=0)  # Calculate average of all values

            intersections[min_idx] = m  # Save mean

        intersections = np.delete(intersections, idxs_to_remove, axis=0)  # Delete all extra elements

        return intersections

    @staticmethod
    def _sort_corners(intersections):
        """ Sort each of four corners. """
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

        # Return sorted corners
        return np.float32(intersections)[[topLeft, topRight, bottomLeft, bottomRight]]

    def _perspective_warp(self, corners):
        """ Use corners to crop card and warp to a flat image. """
        M = 600
        RAT = 1.5
        dstPts = np.float32([[0, 0], [M, 0], [0, M * RAT], [M, M * RAT]])  # Proportional size of card
        srcPts = corners  # Convert and sort source points

        # Preform perspective warp:
        trans, status = cv2.findHomography(srcPts, dstPts)
        return cv2.warpPerspective(self.orig_im, trans, (M, int(M * RAT)))

    def process(self):
        edged = self._get_edged()
        edged = self._process_edged(edged)
        contours = self._get_contours(edged)
        id_contour = self._find_card_contour(edged, contours)
        id_contour = self._process_card_contour(id_contour)
        lines = self._get_edge_lines(id_contour)
        segmented = self._segment_lines_by_angle_kmeans(lines)
        intersections = self._segmented_intersections(segmented)
        intersections = self._fuse_close_points(intersections)
        corners = self._sort_corners(intersections)
        self.res_im = self._perspective_warp(corners)


if __name__ == "__main__":
    IMG_NAME = 'id_front.jpeg'
    IMG_INP = 'sample_data/' + IMG_NAME
    IMG_OTP = 'output_data/' + IMG_NAME
    im = cv2.imread(IMG_INP)  # Load image
    sample_card = IDSelect(im)
    sample_card.process()
    cv2.imshow("Final", sample_card.res_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


