import numpy as np
import sep
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse
import pyfits
import cv2
import fits_manager
filename = 'TARGET__00161.fit'




hdul = fits.open(filename)

print(hdul.info())

hdu = hdul[0]

image_shape = hdul[0].data.shape
print(hdu.data)
data = hdu.data
m, s = np.mean(data), np.std(data)
np_data = np.array(data, dtype= float)
print(np_data)

print(np.dtype(np_data[0][0]))
data, header = pyfits.getdata(filename, 0, header = True)
## not working!

from astropy.visualization import (ZScaleInterval, MinMaxInterval, ImageNormalize, LinearStretch)


norm = ImageNormalize(data, interval=ZScaleInterval(),
                      stretch=LinearStretch())

plt.imshow(data, origin='lower', norm=norm)

plt.colorbar()
plt.show()

##



#data = data //(data.max()//255)
#data = v2.convertScaleAbs(data)
data = np_data
#data = data.byteswap(inplace=True).newbyteorder()
print(np.dtype(data[695][520]))
print(data)
dim = data.shape
my_mask = fits_manager.generate_circle(dim[0], dim[1], (dim[0]//2, dim[1]//2), 250)
bkg = sep.Background(data)

sep.set_extract_pixstack(10000000)

data_sub = data-bkg
print(my_mask)
objects = sep.extract(data_sub, 0.5, minarea = 30, deblend_nthresh=1, mask = my_mask, deblend_cont = 0.1)
(data_sub, 0.5)
#print(objects)
print(len(objects))
print(objects['x'], objects['y'], objects['npix'])



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

plt.show()
