"""
Utility to censor a cropped ID Card photo

NOTE -- if using system install of OpenCV (not from pip)
    Define the folder containing the OpenCV `haarcascades` in the environment var "OPENCV_HAAR".
    Include trailing slash.
        example: "/usr/local/Cellar/opencv/4.3.0/share/opencv4/haarcascades/"
"""

import cv2
import numpy as np
import os


try:
    FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except AttributeError:
    classifier_path = os.environ["OPENCV_HAAR"] + 'haarcascade_frontalface_default.xml'
    print("Loading classifier from: '{}'".format(classifier_path))
    FACE_CASCADE = cv2.CascadeClassifier(classifier_path)


class IDIsolate:
    def __init__(self, source: np.array):
        self.orig_im = source
        self.res_im = None
        self._all_faces = None  # Will hold all face coords from detection

    @property
    def faces(self):
        """ Get coordinates of face coords  """
        if self._all_faces is None:
            gray = cv2.cvtColor(self.orig_im, cv2.COLOR_BGR2GRAY)  # Create gray image
            self._all_faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)  # Detect faces

        return self._all_faces

    def display_faces(self):
        """ Display all faces on original image copy. """
        im = self.orig_im.copy()
        for (x, y, w, h) in self.faces:  # Plot faces on display photo
            im = cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return im

    @property
    def largest_face(self):
        """ Get the largest face from coordinates using  """
        areas = self.faces[:, 2] * self.faces[:, 3]  # Calculate all areas
        return self.faces[np.argmax(areas)]  # Select largest as main face

    @staticmethod
    def enlarge_face_coords(face_coords):
        """ Take single face coordinates and enlarge to fit entire head. """
        vert = int(1.3 * face_coords[2])
        horiz = int(1.1 * face_coords[3])
        return np.array([
            face_coords[0] - int((horiz - face_coords[2]) / 2),
            face_coords[1] - int((vert - face_coords[3]) / 2),
            horiz,
            vert,
            ])

    def _select_face(self, face_coords):
        """ Get pixels at face coords """
        return self.orig_im[face_coords[1]:face_coords[1] + face_coords[3],
               face_coords[0]:face_coords[0] + face_coords[2]]

    @property
    def obfuscated(self):
        """ Create obfuscated version of original image. """
        return cv2.blur(self.orig_im, (25,25))


    def overlay_unblurred(self, face_coords):
        """ Overlay unblurred face on resulting image. """
        self.res_im[face_coords[1]:face_coords[1] + face_coords[3],
        face_coords[0]:face_coords[0] + face_coords[2]] = self._select_face(face_coords)

    def process(self):
        """ Run all major operations to isolate face and produce obfuscated final image with clear face. """
        f = self.largest_face
        f = self.enlarge_face_coords(f)
        self.res_im = self.obfuscated
        self.overlay_unblurred(f)

        return self.res_im


if __name__ == "__main__":
    # Setup variables:
    IMG_NAME = 'id_front.jpeg'
    IMG_INP = 'output_data/' + IMG_NAME  # Grab from output of `card_crop` preprocessing

    res = cv2.imread(IMG_INP)
    isolate = IDIsolate(res)  # Load image into isolator
    res = isolate.process()
    cv2.imshow('img', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



