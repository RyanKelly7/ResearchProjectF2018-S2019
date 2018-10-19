from astropy.io import fits
import numpy as np


class FitsManager:
    def __init__(self, fits_file_name):
        """

        """
        self.fits_file = fits_file_name
        self.mask_threshold = 10
        self.data_sub = None
        self.image_data = FitsManager.read_image(fits_file_name)
        self.image_mask = FitsManager.generate_mask(self.image_data,
                                                    self.mask_threshold)
        self.lattice_region_masks = FitsManager.generate_lattice_masks(
            self.image_data, 10, 10)


    def get_image_data(self):
        return self.image_data

    def get_image_mask(self):
        return self.image_data

    def get_lattice_region_masks(self):
        return self.lattice_region_masks

    def read_image(fits_file_name):
        """

        :param fits_file_name: file path with name included of the desired fits
         image
        :return: None
        """
        # TODO: this method!!! and error handling???
        try:
            # Try to open file (link to GUI at some point)
            image_fits_formatted = fits.open(fits_file_name)
        except IOError:
            return
            # request new image file name from user (through GUI?)

        return image_fits_formatted[0].data

    def write_image(self, file_path):
        """

        :param file_path:
        :return:
        """
        # TODO: whats up with this method???
        return

    def get_sub_image(self, data, mask, mask_false_value=False):
        """
        returns an image of same shape as data but with zero where there
        are false mask values

        :param data: image data values
        :param mask: map of valid image data values
        :param mask_false_value: value within image_mask (normally False) to not
        include in valid image
        :return: data without masked values
        """
        # TODO: ask about error handling!!! Also this method might be useless!
        assert data.shape == mask.shape
        return np.where(mask, mask_false_value, data)

    def generate_mask(image, threshold):
        """
        Automatically create a mask of an image given the image and a threshold
        value

        :param image: the AllSky image to create a mask from
        :param threshold: the minimum pixel value within the image to be mapped
        to the mask
        :return: A mask with same shape as the image, True if the image value is
        greater than threshold and less than the maximum pixel value minus one
        standard deviation of the image values.
        """
        # TODO: Test this and optimize to only include pixels inward of the
        # horizon
        x_pix, y_pix = image.shape
        image_median = np.median(image) # TODO: this vs np.median(image, axis=0)
        image_mean = np.mean(image)
        image_std = np.std(image)
        image_max = image.max()

        # generate mask
        mask = np.where(threshold < image < image_max - image_std, True, False)
        return mask

    def generate_lattice_masks(image_data, num_x_divisions, num_y_divisions):
        total_masks = num_y_divisions * num_x_divisions

        image_x_pix, image_y_pix = image_data.shape

        masks = np.zeros([total_masks, image_x_pix, image_y_pix], bool)

        x_borders = np.linspace(0, image_x_pix, num_x_divisions)
        y_borders = np.linspace(0, image_y_pix, num_y_divisions)

        for x_region in range(num_x_divisions):
            for y_region in range(num_y_divisions):
                x_lower_index = np.floor(x_borders[x_region])
                x_upper_index = np.floor(x_borders[x_region + 1])

                y_lower_index = np.floor(y_borders[y_region])
                y_upper_index = np.floor(y_borders[y_region + 1])

                mask_number = x_region * num_y_divisions + y_region

                masks[mask_number][x_lower_index:x_upper_index][
                    y_lower_index:y_upper_index] = True
        return masks


# TODO: maybe delete all code after this, possibly not necessary
# generates a region for each NSEW, then concentric circles.
#  Total of 4 * param(circular_regions) regions
# all within the range of the image mask with some data outside mask range
# relatively computationally intensive, should only be called on images
# used for training
# TODO: ask about , nseg_az=10, nseg_el=3 as potential parameters
def generate_sub_region_masks(image_mask, n_circular_regions, n_slice_regions, theta_offset=0):
    # get largest circle radius
    largest_circle_radius = 10;
    radius_increment = 10
    while region_has_outer_value(largest_circle_radius, image_mask):
        largest_circle_radius += radius_increment

    # get all region's radii and region angles
    region_radii = np.linspace(0, largest_circle_radius, n_circular_regions + 1)
    region_theta = np.linspace(0, 2 * np.pi, n_slice_regions + 1)

    # create mask of dimensions image.shape() for each desired mask region
    # z-ordering is given by N, S, W, E for circular region
    # of radius 0 - region_radii[1] ...
    # region_radii[end -1] - region_radii[end]
    x_pix, y_pix = image_mask.shape
    regions_matrix = np.zeros([n_circular_regions * n_slice_regions, x_pix,
                               y_pix], bool)

    center = [x_pix / 2, y_pix / 2]
    truth_val = True


    # todo: check numpy.where or numpy.ma (masked array)
    for x in range(x_pix):
        for y in range(y_pix):
            radius_number, direction = pix_in_region(x, y, region_radii, center)
            # direction = direction_dictionary[direction]
            regions_matrix[x][y][radius_number * 4 + direction] = truth_val
    return regions_matrix


def pixel_in_region(x, y, direction, lower_radius, upper_radius, center):
    """
    Function to test if a given pixel is within the desired region.
    Helper function for generate_sub_regions_mask.

    :param x: pixel x value
    :param y: pixel y value
    :param direction: NSEW direction of image (might want to change to n-regions around center)
    :param lower_radius: lower radius of region
    :param upper_radius: upper radius of region
    :param center: center of the image
    :return: boolean value, true if between upper and lower radii and false
    """
    pixel_radius = ((x - center[0])**2 + (y - center[1])**2)**(1/2)

    if not (lower_radius < pixel_radius < upper_radius):
        return False

    # return direction == get_direction(x, y)


def get_theta(x, y):
    """
    helper function for pixel_in_region method
    determines direction according to top of image being N, and right being E
    (will need to be changed later according
    to pixel_in_region specifications

    :param x: pixel x value
    :param y: pixel y value
    :return: string of direction, ('N', 'S', 'E', or 'W')
    """


# TODO: finish this and check that regions are calculated appropriately
def pix_in_region(x, y, radii, center):
    radius_val = ((x-center[0])**2 + (y-center[1])**2)**(1/2)

    radius_return_index = False
    # todo: numpy.where
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
# should return true if there exits some value, in the mask, outside the
# circular region of the given radius
def region_has_outer_value(region_radius, mask):
    x_pix, y_pix = mask.shape()
    center_x = x_pix / 2
    center_y = y_pix / 2
    for x in range(x_pix):
        for y in range(y_pix):
            # mask might have inverted values
            # TODO: check mask true/false orientation and change "mask[x][y]" to
            # "not mask[x][y]" if needed or return False
            if mask[x][y] and (x - center_x)**2 + (y - center_y)**2 > \
                    region_radius:
                return True
    return False


