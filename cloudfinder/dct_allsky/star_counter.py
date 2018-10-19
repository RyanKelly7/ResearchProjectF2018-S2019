import numpy as np
import sep
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Ellipse
import cv2
from .fits_manager import FitsManager
from astropy.visualization import (ZScaleInterval, MinMaxInterval, ImageNormalize, LinearStretch)


class StarCounter:
    def __init__(self, fits_file_name):

        self.fits_file = fits_file_name
        self.fits_manager = FitsManager(fits_file_name)
        self.image_data = self.fits_manager.get_image_data()
        self.image_data_formatted = np.array(self.image_data, dtype=float)
        self.image_mask = FitsManager.image_mask

        self.star_objects = StarCounter.run_sep_extractor()

    def run_sep_extractor(self):
        # Not entirley sure what this is for but the likeliness of a memory
        # error occurring in call to sep.extract is inversely proportional to
        # the pixstack...
        sep.set_extract_pixstack(100000)

        data = self.image_data_formatted
        # generate a background map of data
        bkg = sep.Background(data)
        # subtract the background map from data
        data_sub = data - bkg
        threshold = np.std(data_sub) + np.min(data_sub)
        star_objects = sep.extract(data_sub, threshold, minarea=9,
                              mask=self.image_mask,
                              gain=3, deblend_nthresh=32, deblend_cont=0.0005)

        good_objects = star_objects[star_objects['flag'] < 8]

        return good_objects

    def debug_pyplotter(self):
        data_sub = self.image_data_formatted - \
                   sep.Background(self.image_data_formatted)
        fig, ax = plt.subplots()
        m, s = np.mean(data_sub), np.std(data_sub)
        im = ax.imshow(data_sub, interpolation='nearest', cmap='gray',
                       vmin=m - s, vmax=m + s, origin='lower')
        # plot an ellipse for each object
        objects = self.star_objects
        for i in range(len(self.star_objects)):
            e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                        width=6 * objects['a'][i],
                        height=6 * objects['b'][i],
                        angle=objects['theta'][i] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax.add_artist(e)

        plt.colorbar(im)
        plt.show()



