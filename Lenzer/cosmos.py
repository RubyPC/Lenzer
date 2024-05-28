#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 08:06:31 2024

@author: ruby
"""
"""
Turn COSMOS galaxies into Lenstronomy inputs.

This module contains the default class for transforming the objects of the
COSMOS catalog into sources to be passed to lenstronomy.

adapted from manada et al, 18 April 2021
"""
from pathlib import Path

import astropy
import astropy.table
from astropy.io import fits
from lenstronomy.Util.param_util import phi_q2_ellipticity
import numpy as np
import numpy.lib.recfunctions
from tqdm import tqdm

from galaxy_catalog import GalaxyCatalog, DEFAULT_Z


class COSMOSCatalog(GalaxyCatalog):
    """Interface to the COSMOS 23.5 magnitude catalog

    This is the catalog used for real galaxies in galsim, see
    https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data.
    The catalog must be downloaded and unzipped.

    Args:
        folder (str): Path to the folder with the catalog files
        smoothing_sigma (float): Width of Gaussian image smoothing filter
            in arcseconds.
        random_rotation (bool): Whether to randomly rotate images or not
            if no rotation angle is given
    """

    pixel_width = 0.03  # Hubble ACS

    def __init__(self, folder, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Store the path as a Path object.
        self.folder = Path(folder)

        # Check if we've already populated the catalog
        self.catalog_path = self.folder / "manada_catalog.npy"
        self.npy_files_path = self.folder / "npy_files"
        if self.catalog_path.exists() and self.npy_files_path.exists():
            self.catalog = np.load(str(self.catalog_path))

        else:
            # Make the directory where I'm going to save the npy files
            self.npy_files_path.mkdir()
            # Combine all partial catalog files
            catalogs = [
                unfits(str(self.folder / fn))
                for fn in [
                    "real_galaxy_catalog_23.5.fits",
                    "real_galaxy_catalog_23.5_fits.fits",
                ]
            ]

            # Duplicate IDENT field crashes numpy's silly merge function.
            catalogs[1] = numpy.lib.recfunctions.drop_fields(catalogs[1], "IDENT")

            # Custom fields
            catalogs += [
                np.zeros(
                    len(catalogs[0]),
                    dtype=[
                        ("size_x", np.int),
                        ("size_y", np.int),
                        ("z", np.float),
                        ("pixel_width", np.float),
                    ],
                )
            ]

            self.catalog = numpy.lib.recfunctions.merge_arrays(catalogs, flatten=True)

            self.catalog["pixel_width"] = self.pixel_width
            self.catalog["z"] = self.catalog["zphot"]

            # Loop over the images to find their sizes.
            catalog_i = 0
            for img, meta in self.iter_image_metadata_bulk(
                message="One-time COSMOS startup"
            ):
                # Grab the shape of each image.
                meta["size_x"], meta["size_y"] = img.shape
                # Save the image as its own image.
                img = img.astype(np.float)
                np.save(str(self.npy_files_path / ("img_%d.npy" % (catalog_i))), img)
                catalog_i += 1

            np.save(self.catalog_path, self.catalog)

    def __len__(self):
        return len(self.catalog)

    @staticmethod
    def _file_number(fn):
        """Return integer X in X.fits filename fn.
        X can be more than one digit, not necessarily zero padded.
        """
        return int(str(fn).split("_n")[-1].split(".")[0])

    def iter_image_metadata_bulk(self, message=""):
        """Yields the image array and metadata for all of the images
        in the catalog.

        Args:
            message (str): Progress bar message to display

        Returns:
            (generator): A generator that can be iterated over to give
            lenstronomy kwargs.

        Notes:
            This will read the fits files.
        """
        catalog_i = 0
        _pattern = f"real_galaxy_images_23.5_n*.fits"  
        files = list(sorted(self.folder.glob(_pattern), key=self._file_number))

        # Iterate over all the matching files.
        for fn in tqdm(files, desc=message):
            with fits.open(fn) as hdul:
                for img in hdul:
                    yield img.data, self.catalog[catalog_i]
                    catalog_i += 1

    def image_and_metadata(self, catalog_i):
        """Returns the image array and metadata for one galaxy.

        Parameters:
            catalog_i (int): The catalog index

        Returns
            ([np.array, np.void]) A numpy array containing the image
            metadata and a numpy void type that acts as a dictionary with
            the metadata.

        Notes:
            This will read the numpy files made during initialization. This is
            much faster on average than going for the fits files.
        """
        img = np.load(str(self.npy_files_path / ("img_%d.npy" % (catalog_i))))
        return self.smooth_image(img), self.catalog[catalog_i]


class COSMOSSersicCatalog(COSMOSCatalog):
    """As COSMOSCatalog, but produces the best-fit single elliptic Sersic
    profiles instead of real galaxy images.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        _fit_results = self.catalog["sersicfit"].astype(np.float)
        self.sercic_info = {
            p: _fit_results[:, i]
            for i, p in enumerate("intensity r_half n q boxiness x0 y0 phi".split())
        }
        # Convert half-light radius from pixels to arcseconds
        self.sercic_info["r_half"] *= self.pixel_width

    def draw_source(self, catalog_i=None, z_new=DEFAULT_Z, phi=None):
        """Creates lenstronomy interpolation lightmodel kwargs from
            a catalog image.

        Args:
            catalog_i (int): Index of image in catalog
            z_new (float): Redshift to place image at
            phi (float): Rotation to apply to the image.
                If not provided, use random or original rotation
                depending on source_parameters['random_rotation']

        Returns:
            (list,list) A list containing the model ['INTERPOL'] and
                the kwargs for an instance of the class
                lenstronomy.LightModel.Profiles.interpolation.Interpol

        Notes:
            If not catalog_i is provided, one that meets the cuts will be
            selected at random.
        """
        catalog_i, phi = self.fill_catalog_i_phi_defaults(catalog_i, phi)
        metadata = self.catalog[catalog_i]

        z_scaling = self.z_scale_factor(metadata["z"], z_new)

        # Get sercic info for this particular galaxy
        sercic_info = {p: self.sercic_info[p][catalog_i] for p in self.sercic_info}

        # Convert (phi, q) -> (e1, e2), after applying possible random rotation
        # Using py_func to avoid numba caching trouble
        # (and it's a trivial func anyway)
        e1, e2 = phi_q2_ellipticity.py_func(
            (sercic_info["phi"] + phi) % (2 * np.pi), sercic_info["q"]
        )

        # Convert to kwargs for lenstronomy
        return (
            ["SERSIC_ELLIPSE"],
            [
                dict(
                    # Scale by pixel area before z-scaling, as in parent draw_source
                    amp=sercic_info["intensity"] / metadata["pixel_width"] ** 2,
                    e1=e1,
                    e2=e2,
                    R_sersic=sercic_info["r_half"] * z_scaling,
                    n_sersic=sercic_info["n"],
                )
            ],
        )

    def image_and_metadata(self, catalog_i):
        raise NotImplementedError

    def iter_image_metadata_bulk(self, message=""):
        raise NotImplementedError


def unfits(fn, pandas=False):
    """Returns numpy record array from fits catalog file fn.

    Args:
        fn (str): filename of fits file to load
        pandas (bool): If True, return pandas DataFrame instead of an array

    Returns:
        (np.array):
    """
    if pandas:
        astropy.table.Table.read(fn, format="fits").to_pandas()
    else:
        with fits.open(fn) as hdul:
            data = hdul[1].data
            # Remove fitsyness from record array
            return np.array(data, dtype=data.dtype)