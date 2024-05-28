#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:06:31 2024

@author: ruby
"""
"""
Turn real galaxies into Lenstronomy inputs.

This module contains the default class for transforming the objects of a
source catalog into sources to be passed to lenstronomy.

"""
import cosmology
import numpy as np
import scipy

# default redshift of source galaxy 
DEFAULT_Z = 1.5


class GalaxyCatalog:
    """Class for turning real galaxy images into Lenstronomy inputs.


    Args:
        smoothing_sigma (float): Width of Gaussian image smoothing filter
            in arcseconds.
        random_rotation (bool): Whether to randomly rotate images or not
            if no rotation angle is given
    """

    DEFAULT_PIXEL_WIDTH: float

    def __init__(self, smoothing_sigma=0.08, random_rotation=False):
        self.cosmo = cosmology.setCosmology("planck18")
        self.smoothing_sigma = smoothing_sigma
        self.random_rotation = random_rotation

    def __len__(self):
        """Returns the length of the catalog"""
        raise NotImplementedError

    def image_and_metadata(self, catalog_i):
        """Returns the image array and metadata for one galaxy

        Parameters:
            catalog_i (int): The catalog index

        Returns
            ([np.array, np.void]) A numpy array containing the image
            metadata and a numpy void type that acts as a dictionary with
            the metadata.
        """
        raise NotImplementedError

    def sample_indices(self, n_galaxies, selection_mask=None):
        """Return n_galaxies array of catalog indices to sample

        Args:
            n_galaxies (int): Number of indices to return
            selection_mask (array): Boolean array indicating which galaxies
                to sample from.

        Returns:
            (np.array): Array of ints of catalog indices to sample.

        Notes:
            The minimum apparent magnitude, minimum size in pixels, and
            minimum redshift are all set by the source parameters dict.
        """
        if selection_mask is None:
            selection_mask = np.ones(len(self), dtype=np.bool_)
        return np.random.choice(
            np.where(selection_mask)[0], size=n_galaxies, replace=True
        )

    def smooth_image(self, img):
        """Return img smoothed by Gaussian filter with sigma set
        by self.smoothing_sigma
        """
        if self.smoothing_sigma > 0:
            return scipy.ndimage.gaussian_filter(
                img, sigma=self.smoothing_sigma / self.pixel_width
            )
        return img

    def fill_catalog_i_phi_defaults(self, catalog_i=None, phi=None):
        """Return catalog index and source rotation angle.

        Args:
            catalog_i (int): Index of image in catalog
                If not provided or None, will be sampled randomly from
                the catalog.
            phi (float): Rotation to apply to the image.
                If not provided or None, will either randomize or use 0,
                depending on self.random_rotation.
        """
        # If no index is provided pick one at random
        if catalog_i is None:
            catalog_i = self.sample_indices(1).item()
        # If no rotation is provided, pick one at random or use original
        # orientation
        if phi is None:
            if self.random_rotation:
                phi = self.draw_phi()
            else:
                phi = 0
        return catalog_i, phi

    def draw_source(self, catalog_i=None, z_new=DEFAULT_Z, phi=None):
        """Creates lenstronomy interpolation lightmodel kwargs from
            a catalog image.

        Args:
            catalog_i (int): Index of image in catalog
            z_new (float): Redshift to place image at
            phi (float): Rotation to apply to the image.
                If not provided, randomize or use 0, depending on
                source_parameters['random_rotation']

        Returns:
            (list,list) A list containing the model ['INTERPOL'] and
                the kwargs for an instance of the class
                lenstronomy.LightModel.Profiles.interpolation.Interpol

        Notes:
            If not catalog_i is provided, one that meets the cuts will be
            selected at random.
        """
        catalog_i, phi = self.fill_catalog_i_phi_defaults(catalog_i, phi)
        img, metadata = self.image_and_metadata(catalog_i)
        pixel_width = metadata["pixel_width"]

        # With this, lenstronomy will preserve the scale/units of
        # the input image (in a configuration without lensing, same pixel widths)
        img = img / pixel_width ** 2

        pixel_width *= self.z_scale_factor(metadata["z"], z_new)

        # Convert to kwargs for lenstronomy
        return (
            ["INTERPOL"],
            [dict(image=img, center_x=0, center_y=0, phi_G=phi, scale=pixel_width)],
        )

    @staticmethod
    def draw_phi():
        """Draws a random rotation angle for the interpolation of the source.

        Returns:
            (float): The new angle to use in the interpolation class.
        """
        return np.random.rand() * 2 * np.pi

    def z_scale_factor(self, z_old, z_new):
        """Return multiplication factor for object/pixel size for moving its
        redshift from z_old to z_new.

        Args:
            z_old (float): The original redshift of the object.
            z_new (float): The redshift the object will be placed at.

        Returns:
            (float): The multiplicative pixel size.
        """
        # Pixel length ~ angular diameter distance
        # (colossus uses funny /h units, but for ratios it doesn't matter)
        return (
            self.cosmo.angularDiameterDistance(z_old)
            / self.cosmo.angularDiameterDistance(z_new))