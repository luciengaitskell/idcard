""" Read barcode on the reverse of driver's licenses to extract required information. """

import zxing
import datetime


reader = zxing.BarCodeReader()


class _IDCardData:
    """
    Define values to collect from encoded data on barcode.

    In format of (start_sequence, data_type)
    """
    firstname = ("DAC", str)
    lastname = ("DCS", str)
    middlename = ("DAD", str)
    datebirth = ("DBB", datetime.date)


class IDBarcode(_IDCardData):
    def __init__(self, img: str):
        """
        Initialize and run ID card decoder.

        NOTE: From testing, the barcode MUST be in landscape orientation for detection.
        Can be .png or .jpg/.jpeg

        :param img: File path to decode
        """
        self.barcode = reader.decode(img)
        if self.barcode is not None:
            self.__parse()
        else:
            raise ValueError("None detected")

    def __parse(self):
        """ Iterate encoded barcode data and parse known lines. """

        # Select defined keys (all non-private members) in _IDCardData
        keys = list(filter(lambda x: not x[0].startswith('__'), vars(_IDCardData).items()))

        # Operate over each line in parsed barcode data:
        for l in self.barcode.parsed.splitlines():
            # Iterate through collected ID card keys
            for valname, (startseq, keytype) in keys:
                # Find if matching start sequence for this key
                if l.startswith(startseq):
                    # Select data after start sequence
                    data = l[len(startseq):]

                    # -- Custom data processing -- #
                    if keytype == datetime.date:  # if key is date
                        data = datetime.datetime.strptime(data, "%m%d%Y").date()

                    # Add attribute to `self`
                    setattr(self, valname, data)
                    break


if __name__ == "__main__":
    import cv2
    IMG = 'sample_data/id_back.png'

    bc = IDBarcode(IMG)

    im = cv2.imread(IMG)

    # get the barcode top-left position
    x1 = int(bc.barcode.points[0][0])
    y1 = int(bc.barcode.points[0][1])


    def bc_point(num):  # Get barcode point pair
        return int(bc.barcode.points[num][0]), int(bc.barcode.points[num][1])


    # Draw box around barcode
    for i in [(0, 2), (2, 3), (3, 1), (1, 0)]:
        cv2.line(im, bc_point(i[0]), bc_point(i[1]), (255, 0, 0), 3)

    # draw the barcode data and barcode type on the image
    cv2.putText(im, bc.barcode.raw, (x1, y1 - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    print(bc.datebirth)

    cv2.imshow("Reading PDF417 with Webcam", im)
    cv2.waitKey(0);
    cv2.destroyAllWindows()
