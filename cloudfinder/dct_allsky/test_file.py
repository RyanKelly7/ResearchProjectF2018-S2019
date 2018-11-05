
from fits_manager import FitsManager
from star_counter import StarCounter


'''
data_sub = Star_Counter.data_sub_bkg

Fits_Manager = FitsManager("TARGET__00161.fit")
valid_image = Fits_Manager.get_sub_image(Fits_Manager.image_data,
                                         Fits_Manager.image_mask)

fig, ax = plt.subplots()
m, s = np.mean(Star_Counter.data_sub_bkg), np.std(data_sub)

im = ax.imshow(data_sub, interpolation='nearest', cmap='gray',
                       vmin=m - s, vmax=m + s, origin='lower')
'''

Star_Counter = StarCounter("TARGET__00161.fit")
# print(Star_Counter.star_objects)
Star_Counter.debug_pyplotter()
