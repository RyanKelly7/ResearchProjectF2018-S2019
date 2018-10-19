from .fits_manager import FitsManager
import matplotlib.pyplot as plt


fits_manager = FitsManager("TARGET__0161.fir")

valid_image = fits_manager.get_sub_image(fits_manager.image_data)

fig, ax = plt.subplots()
m, s = np.mean(fits_manager.data_sub), np.std(data_sub)

im = ax.imshow(data_sub, interpolation='nearest', cmap='gray',
                       vmin=m - s, vmax=m + s, origin='lower')
