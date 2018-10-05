import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

filename = 'TARGET__00161.fit'

hdul = fits.open(filename)
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
    x_pix, y_pix = image_data.shape
    image = np.zeros(shape=(x_pix, y_pix))
    for x in range(x_pix):
        for y in range(y_pix):
            if image_map[x][y]:
                image[x][y] = image_data[x][y]
    return image


'''
circle_image = get_sub_image(hdu.data, circle_map)

print(circle_image)
'''


# Generate mask (not just circular but data based)
# Should get all valid data within horizon of allsky image

# Generates a mask given a small array of images from batch
# Images must have same dimensions
def generate_mask(images):
    x_pix, y_pix = images[0].shape

    # checks for images that are not the same size
    invalid_image_indices = []
    for index, each_image in enumerate(images):
        temp_x_pix, temp_y_pix = each_image.shape
        if temp_x_pix != x_pix or temp_y_pix != y_pix:
            invalid_image_indices.append(index)

    # Generates a singular image which is just the sum of every image value
    # at each pixel
    sum_of_images = np.zeros(shape=(x_pix, y_pix))
    for index, each_image in enumerate(images):
        if index in invalid_image_indices:
            continue
        sum_of_images += each_image

    # compute data to use for masking
    mean = np.mean(sum_of_images)
    std = np.std(sum_of_images)

    # generate mask
    mask = np.zeros(shape=(x_pix, y_pix))
    mask_2 = np.zeros(shape=(x_pix, y_pix))
    mask_f = np.zeros(shape=(x_pix, y_pix))
    
    max_val = sum_of_images.max()
    mask = sum_of_images < mean - std
    mask2 = sum_of_images > (mean + std / 2)

    for i in range(x_pix):
        for j in range(y_pix):
            mask_f[i][j] = mask[i][j] and mask2[i][j]

    return mask_f


# generates a region for each NSEW, then concentric circles. Total of 4 * param(circular_regions) regions
# all within the range of the image mask with some data outside mask range
# relatively computationally intensive, should only be called on images used for training
def generate_sub_regions_mask(image_mask, circular_regions):
    # get largest circle radius
    largest_circle_radius = 10;
    radius_increment = 10
    while region_has_outer_value(largest_circle_radius, image_mask):
        largest_circle_radius += radius_increment

    # get all region's radii
    region_radii = np.linspace(0, largest_circle_radius, circular_regions)

    # create mask of dimensions image.shape() for each desired mask region
    # z-ordering is given by N, S, W, E for circular region of radius 0 - region_radii[1] ...
    # ragion_radii[end -1] - region_radii[end]
    regions_matrix = np.zeros(image_mask.shape().append(4 * circular_regions))
    x_pix, y_pix = image_mask.shape()
    center = [x_pix / 2, y_pix / 2]
    truth_val = 1
    direction_dictionary = {'N': 0, 'S': 1, 'W': 2, 'E': 3}
    for x in range(x_pix):
        for y in range(y_pix):
            radius_number, direction = pix_in_region(x, y, region_radii, center)
            direction = direction[direction]
            regions_matrix[x][y][radius_number * 4 + direction]
    return regions_matrix


# TODO: finish this and check that regions are calculated appropriately
def pix_in_region(x, y, radii, center):
    radius_val = ((x-center[0])**2 + (y-center[1])**2)**(1/2)

    radius_return_index = False
    for radius_number in range(len(radii) - 1):
        if radii[radius_number] < radius_val < radii[radius_number]:
            radius_return_index = radius_number

    # TODO: double check functions are appropriate and that use is good
    line_slope_1 = 1
    line_slope_2 = -1
    y_function_1 = lambda x_val: line_slope_1 * (x_val - center[0])
    y_function_2 = lambda x_val: line_slope_2 * (x_val - center[0])

    x_function_1 = lambda y_val: (y_val - center[1]) / line_slope_1
    x_function_2 = lambda y_val: (y_val - center[1]) / line_slope_2

    if x_function_1(y) > x > x_function_2(y) and y_function_1(x) > y > y_function_2(x):
        direction = 'N'
    elif x_function_1(y) < x < x_function_2(y) and y_function_1(x) < y < y_function_2(x):
        direction = 'S'
    elif x_function_1(y) < x < x_function_2(y) and y_function_1(x) > y > y_function_2(x):
        direction = 'W'
    else:
        direction = 'E'

    return radius_number, direction


# helper function for generate_sub_regions
# should return true if there exits some value, in the mask, outside the circular region
# of the given radius
def region_has_outer_value(region_radius, mask):
    x_pix, y_pix = mask.shape()
    center_x = x_pix / 2
    center_y = y_pix / 2
    for x in range(x_pix):
        for y in range(y_pix):
            # mask might have inverted values
            # TODO: check mask true/false orientation and change "mask[x][y]" to "not mask[x][y]" if needed
            if mask[x][y] and (x - center_x)**2 + (y - center_y)**2 > region_radius:
                return True
    return False


# print(generate_mask([hdul[0].data]))

