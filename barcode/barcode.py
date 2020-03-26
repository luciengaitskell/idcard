#  https://github.com/mmalecki/zbar/issues/2
#  New Try

""" From: https://github.com/brdamico/PDF417BarcodeReader/blob/master/ReadingBarcodePDF417WithWebcam.py

Edited for static photo, instead of video stream

Note: Found it necessary to use png -- jpeg photo DID NOT work
"""

# import the necessary packages
import zxing
import cv2

IMG = 'sample_data/id_back.png'


# initialize the zxing BarCodeReader
reader = zxing.BarCodeReader()

im = cv2.imread(IMG)
# find the barcode in the frame and decode it
barcodes = reader.decode(IMG)

# check if some info was found
if barcodes is not None:
    # print the value read
    print(barcodes.raw)

    # get the barcode top-left position
    x1 = int(barcodes.points[0][0])
    y1 = int(barcodes.points[0][1])

    def bc_point(num):  # Get barcode point pair
        return int(barcodes.points[num][0]), int(barcodes.points[num][1])

    # Draw box around barcode
    for i in [(0,2), (2,3), (3,1), (1,0)]:
        cv2.line(im, bc_point(i[0]), bc_point(i[1]), (255, 0, 0), 3)


    # draw the barcode data and barcode type on the image
    cv2.putText(im, barcodes.raw, (x1, y1 - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(barcodes.parsed)

# show the output image (with edits if applicable)
cv2.imshow("Reading PDF417 with Webcam", im)
cv2.waitKey(0);
cv2.destroyAllWindows()
