"""
Barcode reading example.

NOTE: From testing, the barcode MUST be in landscape orientation for detection.

"""
import cv2
from idcard.barcode import IDBarcode

# Image location:
IMG = 'sample_data/id_back.png'

# Image processing
bc = IDBarcode(IMG)

# Open image for display
im = cv2.imread(IMG)

#   Get the barcode top-left position
x1 = int(bc.barcode.points[0][0])
y1 = int(bc.barcode.points[0][1])


def bc_point(num):  # Helper function to get barcode point pair
    return int(bc.barcode.points[num][0]), int(bc.barcode.points[num][1])


#   Draw box around barcode on the image
for i in [(0, 2), (2, 3), (3, 1), (1, 0)]:
    cv2.line(im, bc_point(i[0]), bc_point(i[1]), (255, 0, 0), 3)

#   Draw the barcode data and barcode type on the image
cv2.putText(im, bc.barcode.raw, (x1, y1 - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Print parsed data:
print("First: ", bc.firstname)
print("Last: ", bc.lastname)
print("DOB: ", bc.datebirth)

# Display annotated image
cv2.imshow("Reading PDF417 with Webcam", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
