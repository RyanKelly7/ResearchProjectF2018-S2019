import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

#filename = 'TARGET__00001.fit'

#hdul = fits.open(filename)
'''
print(hdul.info())

hdu = hdul[0]
image_shape = hdul[0].data.shape
print(hdu.data[:])
plt.show(hdu.data)
'''


def generate_circle(xPixels, yPixels, center, radius):
    bool_map = np.empty(shape = (xPixels, yPixels))
    for x in range(xPixels):
        for y in range(yPixels):
            if ((x - center[0])**2 + (y - center[1])**2) <= radius**2:
                bool_map[x][y] = False
            else:
                bool_map[x][y] = True
    return bool_map

'''
circle_map = generate_circle(image_shape[0], image_shape[1], (0,0), 2**(1/2) * 4)
print(circle_map)   
print(hdu.data[circle_map == 1])
'''
def get_sub_image(image_data, image_map):
    xP, yP = image_data.shape
    image = np.zeros(shape = (xP, yP))
    for x in range(xP):
        for y in range(yP):
            if image_map[x][y]:
                image[x][y] = image_data[x][y]
    return image
'''
circle_image = get_sub_image(hdu.data, circle_map)

print(circle_image)
'''  
