import numpy as np
import sep
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse
import cv2
import fits_manager
from astropy.visualization import (ZScaleInterval, MinMaxInterval, ImageNormalize, LinearStretch)

# 2 test files 2nd of higher quality, first more starts and wider view
filename = 'TARGET__00161.fit'
# filename = 'mscience0217.fits'


# open fits file using astropy.io fits tool
hdul = fits.open(filename)

# block un-used in current implementation; calculates useful data information.
hdu = hdul[0]
image_shape = hdul[0].data.shape
data = hdu.data
m, s = np.mean(data), np.std(data)

# NECESSARY step for sep.extract: convert uint-16 (or whatever datatype from images) to float
np_data = np.array(data, dtype= float)


# get data using atropy.io.fits tool; include header information
### UNUSED: data, header = fits.getdata(filename, 0, header = True)
# 


# rename main data structure to data for easier use
data = np_data


# Block generates a mask for sep.extract to utilize, should be based of data from fits file
dim = data.shape
my_mask = fits_manager.generate_circle(dim[0], dim[1], (dim[0]//2, dim[1]//2), 250)

# Not entirley sure what this is for but the likeliness of a memory error occurring
# in call to sep.extract is inversely proportional to the pixstack...
sep.set_extract_pixstack(100000)

# generate a background map of data
bkg = sep.Background(data)
# subtract the background map from data
data_sub = data-bkg



# Unecessary display of image data (debugging)
'''
norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=LinearStretch())
plt.imshow(data_sub, origin='lower', norm=norm)

plt.colorbar()
plt.show()
'''

background_std = np.std(data_sub)

## get objects from sep.extract; counts the stars in the image (hopefully)
# argument info:
# (data with background subtracted, minimum value (within data_sub (parameter 0)) for object detection (float),
# minimum area to be considered as a star, ??read Documentation?? -----)
objects = sep.extract(data_sub, 1.5*background_std, minarea = 9, mask = fits_manager.generate_mask([data_sub]),
                      gain = 3, deblend_nthresh=32, deblend_cont = 0.0005)

# print object information for debugging purposes
# DEBUGGING
'''
print(len(objects))
print(objects['x'], objects['y'], objects['npix'])
'''

good_objects = objects[objects['flag']<8]
print(good_objects['flag'])
objects = good_objects

# code block provided in sep.extract example (give link!!)
# code block generates ellipses on data_sub image display;
# ellipses should give positions and sizes of stars found in sep.extract
# Entire block should be abscent in final product; however,
# is entirely necessary for debugging
fig, ax = plt.subplots()
m, s = np.mean(data_sub), np.std(data_sub)
im = ax.imshow(data_sub, interpolation='nearest', cmap='gray',
               vmin=m-s, vmax=m+s, origin='lower')
# plot an ellipse for each object
for i in range(len(objects)):
    e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                width=6*objects['a'][i],
                height=6*objects['b'][i],
                angle=objects['theta'][i] * 180. / np.pi)
    e.set_facecolor('none')
    e.set_edgecolor('red')
    ax.add_artist(e)

plt.colorbar(im)
plt.show()
