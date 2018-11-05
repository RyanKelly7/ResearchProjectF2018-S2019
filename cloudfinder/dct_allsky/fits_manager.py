from astropy.io import fits
import numpy as np
import math


class FitsManager:
    def __init__(self, fits_file_name, create_region_masks=True):

        self.fits_file = fits_file_name
        self.mask_threshold = 700
        self.image_data, self.fits_image = \
            FitsManager.read_image(fits_file_name)
        self.image_mask = FitsManager.generate_mask(self.image_data,
                                                    self.mask_threshold)
        if create_region_masks:
            self.lattice_region_masks = FitsManager.generate_lattice_masks(
                self.image_data, 10, 10)
            self.circular_region_masks = FitsManager.generate_circular_masks(
                self.image_data, 10, 10)

    @staticmethod
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
        fits_image = image_fits_formatted
        data = image_fits_formatted[0].data
        return data, fits_image

    def write_image(self, file_path):
        """

        :param file_path:
        :return:
        """
        # TODO: whats up with this method???
        return

    @staticmethod
    def get_sub_image(data, mask, mask_false_value=False):
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

    @staticmethod
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
        image_median = np.median(image)
        image_mean = np.mean(image)
        image_std = np.std(image)
        image_max = image.max()

        # generate mask
        mask = np.where(threshold < image, False, True)
        return mask

    @staticmethod
    def generate_lattice_masks(image_data, num_x_divisions, num_y_divisions):
        total_masks = num_y_divisions * num_x_divisions

        image_x_pix, image_y_pix = image_data.shape

        masks = np.zeros([total_masks, image_x_pix, image_y_pix], bool)

        x_borders = np.linspace(0, image_x_pix, num_x_divisions + 1)
        y_borders = np.linspace(0, image_y_pix, num_y_divisions + 1)

        for x_region in range(num_x_divisions):
            for y_region in range(num_y_divisions):
                x_lower_index = np.floor(x_borders[x_region])
                x_upper_index = np.floor(x_borders[x_region + 1])

                y_lower_index = np.floor(y_borders[y_region])
                y_upper_index = np.floor(y_borders[y_region + 1])

                mask_number = x_region * num_y_divisions + y_region

                masks[mask_number][
                math.floor(x_lower_index):math.floor(x_upper_index)][
                math.floor(y_lower_index):math.floor(y_upper_index)] = True
        return masks

    @staticmethod
    def generate_circular_masks(image_data, num_concentric_circle_div,
                                num_theta_div):
        total_masks = num_concentric_circle_div * num_theta_div

        image_x_pix, image_y_pix = image_data.shape
        center_x = image_x_pix / 2
        center_y = image_y_pix / 2

        max_radius = ((image_x_pix / 2) ** 2 + (image_y_pix / 2) ** 2) ** (
                    1 / 2)
        max_theta = math.pi * 2

        theta_borders = np.linspace(0, max_theta, num_theta_div + 1)
        circle_borders = np.linspace(0, max_radius,
                                     num_concentric_circle_div + 1)

        masks = np.zeros([total_masks, image_x_pix, image_y_pix], bool)

        for x in range(image_x_pix):
            for y in range(image_y_pix):
                pix_r, pix_theta = \
                    FitsManager.get_pixel_rad_theta_vals(x, y, center_x,
                                                         center_y)
                r_region, theta_region = \
                    FitsManager.get_pixel_rad_theta_regions(pix_r, pix_theta,
                                                            circle_borders,
                                                            theta_borders)
                mask_number = r_region * num_theta_div + theta_region
                masks[mask_number][x][y] = True

        return masks

    @staticmethod
    def get_pixel_rad_theta_vals(pix_x, pix_y, center_x, center_y):
        """
        helper function for generate_circular_masks()

        :param pix_x: pixel x value
        :param pix_y: pixel y value
        :param center_x: image center x value
        :param center_y: image center y value
        :return: radius, theta relative to terminal angle (radius in pixels,
        theta in radians)
        """
        relative_x = pix_x - center_x
        relative_y = pix_y - center_y

        radius = (relative_x ** 0.5 + relative_y ** 0.5) ** 0.5

        if relative_x == 0:
            theta = 0
        else:
            theta = math.atan(abs(relative_y / relative_x))

        if relative_x < 0 < relative_y:
            theta += math.pi / 2
        elif relative_x < 0 and relative_y < 0:
            theta += math.pi
        elif relative_x > 0 > relative_y:
            theta += math.pi * 1.5

        return radius, theta

    @staticmethod
    def get_pixel_rad_theta_regions(r, theta, r_borders, theta_borders):
        r_regions = len(r_borders) - 1
        theta_regions = len(theta_borders) - 1

        r_region = theta_region = None
        for r_val in range(r_regions):
            if r_borders[r_val] <= r <= r_borders[r_val + 1]:
                r_region = r_val
                break

        for theta_val in range(theta_regions):
            if r_borders[theta_val] <= theta <= r_borders[theta_val + 1]:
                theta_region = theta_val
                break

        return r_region, theta_region
