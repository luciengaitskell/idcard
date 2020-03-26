from pyzbar.pyzbar import decode
from PIL import Image

d = decode(Image.open('sample_data/id_back.png'))
print(d)
