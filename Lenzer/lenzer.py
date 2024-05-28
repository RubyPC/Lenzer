#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:08:31 2024

@author: ruby
"""
# Simulate lenses from source galaxies taken from COSMOS

from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Util.simulation_util import data_configure_simple
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel


class Lenzer:
    """Class that makes lensed images

    Args:
        catalog (GalaxyCatalog): catalog that produces source images
        n_pixels (int): image width and height in pixels
        pixel_width (float): pixel width of the *produced* images in arcsec
            (the catalog images may have a different resolution)
        psf_fwhm (float): Full width half maximum of the Gaussian
            point spread function of the observation.
        z_lens (float): Redshift at which to place the lens
        z_source (float): Redshift at which to place the source galaxy
    """

    def __init__(self, catalog, n_pixels=64, pixel_width=None, 
                 psf_fwhm=0.1, z_lens=0.5, z_source=1.5):
         self.catalog = catalog
         self.n_pixels = n_pixels
         if pixel_width is None:
             pixel_width = catalog.pixel_width
         self.pixel_width = pixel_width
         self.psf_fwhm = psf_fwhm
         self.z_lens = z_lens
         self.z_source = z_source
         self.image_length = self.pixel_width * self.n_pixels

    def lensed_image(self, lenses=None, lens_light=None, catalog_i=None, phi=None, z_source=None):

        """Return numpy array describing lensed image.

        Args:
            lenses: list of (lens model name, lens kwargs)
            lens_light: None, or (light model name, light model kwargs)
            catalog_i (int): image index from the catalog.
                If not provided, a random index will be chosen.
            phi (float): rotation to apply to the source galaxy.
                If not provided, it is left up to the catalog whether to
                randomize this or not.
        """
        if z_source is None:
            z_source = self.z_source
        if lenses is None:
            # Do not lens
            lenses = [("SIS", dict(theta_E=0.0))]
        if lens_light is None:
            lens_light_model = None
            lens_light_kwargs = None
        else:
            lens_light_model = LightModel([lens_light[0]])
            lens_light_kwargs = [lens_light[1]]
        catalog_i, phi = self.catalog.fill_catalog_i_phi_defaults(catalog_i, phi)

        lens_model_names, lens_kwargs = list(zip(*lenses))

        source_model_class, kwargs_source = self.catalog.draw_source(
            catalog_i=catalog_i, phi=phi, z_new=z_source
        )

        return ImageModel(
            data_class=ImageData(
                **data_configure_simple(numPix=self.n_pixels, deltaPix=self.pixel_width)
            ),
            psf_class=PSF(psf_type="GAUSSIAN", fwhm=self.psf_fwhm),
            lens_model_class=LensModel(lens_model_names),
            lens_light_model_class=lens_light_model,
            source_model_class=LightModel(source_model_class),
        ).image(
            kwargs_lens=lens_kwargs,
            kwargs_source=kwargs_source,
            kwargs_lens_light=lens_light_kwargs,
        )