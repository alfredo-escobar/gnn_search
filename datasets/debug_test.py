import os
import skimage.io as io
import numpy

filename = os.path.join(os.path.normpath("./datasets/Pepeganga"), "train", "Vajilla Esmaltada Macuira 12 Piezas - Coral Café_62585.jpg".strip())
print(filename)
assert(os.path.exists(filename))