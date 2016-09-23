from scipy import ndimage
from scipy import misc

# input image
lena = misc.imread('Lena.png')

# rotate image
rotate_lena = ndimage.rotate(lena, 180)

# save image
misc.imsave("ans2.png", rotate_lena);
