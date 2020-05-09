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


class IDCardBarcode(_IDCardData):
    def __init__(self, img: str):
        """
        Initialize and run ID card decoder.

        :param img: File path to decode
        """
        self.barcode = reader.decode(img)
        if self.barcode is not None:
            self.__parse()

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
    IMG = 'sample_data/id_back.png'

    bc = IDCardBarcode(IMG)
    print(bc.datebirth)
