# This file is part of dms-bias-correction for enhancing sharpened LST imagery
# Copyright 2023 Hector Nieto and contributors listed in the README.md file.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
@author: Hector Nieto (hector.nieto@ica.csic.es)

DESCRIPTION
===========
This module contains the main lwoer-level functions to perform the enhancement
the dynamic range in sharpened LST images based on [Guzinski_2023]_

MODULE CONTENTS
================

* :func:`bias_correction` Main function for LST bias correction to a sharpened
LST.
* :func:`apply_bias_correction` Bias correction conserving original mean values.
* :func:`local_variability` Computes the local coefficient of variation at a
given local spatial window`Bias correction conserving original mean values.
* :func:`compute_scale` Computes factor scale based on the ration of CV between
the reference and the sharpened.
* :func:`resample_factor` Resamples the bias correction factors to the original
sharpened resolution.
* :func:`raster_data` Helper to read a GDAL image file

REFERENCES
==========
.. [Guzinski_2023] R. Guzinski, H. Nieto, R. Ramo-Sánchez, J.M. Sánchez, I. Joma,
  R. Zitouna-Chebbi, O. Roupsard, R. López-Urrea",
  Improving field-scale crop actual evapotranspiration monitoring with
  Sentinel-3, Sentinel-2, and Landsat data fusion (2023)
  International Journal of Applied Earth Observation and Geoinformation",
  125, 103587, DOI:10.1016/j.jag.2023.103587,
'''

from osgeo import gdal
import numpy as np
import multiprocessing as mp
from pyDMS import pyDMSUtils as gu


WARP_OPTIONS = {"multithread": True,
                "warpOptions": ["NUM_THREADS=%i" % mp.cpu_count()]}


def bias_correction(lst_dms,
                    cv_ref,
                    cv_dms,
                    mean_dms,
                    proj,
                    lr_gt,
                    hr_res,
                    smooth_scale=True):
    """Main function for LST bias correction to a sharpened LST.

    Parameters
    ----------
    lst_dms : 2D array
        Sharpened LST array at fine resolution
    cv_ref, cv_dms : 2D array_like
        LST coefficient of variation at coarse resolution for the
        reference `cv_ref` and the sharpened `cv_dms` products
    mean_dms : 2D array_like
        Mean sharpened LST at coarse resolution
    proj : str
        Working projection system in WKT
    lr_gt : tuple of float
        Low resolution GDAL geotransform
    hr_res : tuple of float
        Spatial resolution (x, y) of the sharpened LST
    smooth_scale : bool, optional
        Whether smooth the coarse resolution factors
        to minimize box-shaped artifacts in the outputs

    Returns
    -------
    lst_corr : 2D array
        Corrected sharpened LST array at fine resolution

    References
    ----------
    .. [Guzinski_2023] R. Guzinski, H. Nieto, R. Ramo-Sánchez, J.M. Sánchez, I. Joma,
      R. Zitouna-Chebbi, O. Roupsard, R. López-Urrea",
      Improving field-scale crop actual evapotranspiration monitoring with
      Sentinel-3, Sentinel-2, and Landsat data fusion (2023)
      International Journal of Applied Earth Observation and Geoinformation",
      125, 103587, DOI:10.1016/j.jag.2023.103587,
    """

    # Compute the scale and resample the scale and mean value
    scale = compute_scale(cv_ref, cv_dms)
    scale = resample_factor(scale, proj, lr_gt,  hr_res,
                               smoothed=smooth_scale)

    mean_dms = resample_factor(mean_dms, proj, lr_gt, hr_res,
                                  smoothed=smooth_scale)

    lst_corr = lst_dms.copy()
    valid = np.logical_and(np.isfinite(lst_dms), np.isfinite(scale))
    lst_corr[valid] = apply_bias_correction(lst_dms[valid],
                                            mean_dms[valid],
                                            scale[valid])

    return lst_corr


def apply_bias_correction(src_array, mean_src, scale):
    """Bias correction conserving original mean values

    Parameters
    ----------
    src_array : array_like
        Array to be corrected
    mean_src : array_like
        Mean values that need to be conserved
    scale : array_like
        Scale factor of the bias correction

    Returns
    -------
    array_like
        Corrected array
    """
    output = mean_src + scale * (src_array - mean_src)
    return output


def local_variability(src_file,
                      target_proj,
                      target_extent,
                      window_size=(1000, 1000)):
    """Computes the local coefficient of variation at a given local spatial window

    Parameters
    ----------
    src_file : str or Path object
        Input fine resolution file
    target_proj : str
        Working projection system in WKT
    target_extent : tuplo of float
        Destination extension (xmin, ymin, xmax, ymax)
        in `targe_proj` coordinate system
    window_size
        Spatial window (x, y) in referenced units that will be used to compute
        the spatial statistics

    Returns
    -------
    cv : 2D array
        Coefficient of variation image array at the given `target_extent`  and
        `window_size` resolution
    mean : 2S array
        Average image array at the given `target_extent`  and
        `window_size` resolution
    lr_gt : tuploe of floag
        GDAL GeoTransform for CV image array
    """

    lr_fid = gdal.Warp("MEM",
                       str(src_file),
                       format="MEM",
                       dstSRS=target_proj,
                       xRes=window_size[1],
                       yRes=-window_size[0],
                       outputBounds=target_extent,
                       resampleAlg="near",
                       **WARP_OPTIONS)

    lr_gt = lr_fid.GetGeoTransform()
    in_fid = gdal.Open(str(src_file), gdal.GA_ReadOnly)
    # Compute image local statistics
    mean, std = gu.resampleHighResToLowRes(in_fid, lr_fid)
    mean = mean[:, :, 0]
    std = std[:, :, 0]
    valid = np.logical_and(np.isfinite(std), mean != 0)
    cv = np.full_like(mean, np.nan)
    cv[valid] = std[valid] / mean[valid]
    return cv, mean, lr_gt


def compute_scale(cv_ref, cv_dms):
    """Computes factor scale based on the ration of CV between 
    the reference and the sharpened 
    
    Parameters
    ----------
    cv_ref : array_like
        Reference Coefficient of Variation array.
    cv_dms : array_like
        Sharpened Coefficient of Variation array. 
    Returns
    -------
    scale : array_like
        
    """""
    valid = np.logical_and(np.isfinite(cv_ref), np.isfinite(cv_dms))
    scale = np.full(cv_ref.shape, 1.0)
    scale[valid] = cv_ref[valid] / cv_dms[valid]
    # We assume that the reference always yields a larger dynamic range due to
    # sharpening limitations, and thus scale is always >= 1
    scale = np.maximum(scale, 1)
    return scale


def resample_factor(factor, lr_gt, template_file, smoothed=True):
    """Resamples the bias correction factors to
    the original sharpened resolution

    Parameters
    ----------
    factor : 2D array
        Bias correction factor (scale or mean)
    template_file : str or Path object
        Reference image file that will be used as template for the resampling
    lr_gt : tuple of float
        Low resolution GDAL geotransform
    hr_res : tuple of float
        Sptial resolution of the original sharpened image, (in referenced units)
    smoothed : bool, optional
        Whether or not smooth the coarse resolution factors
        to minimize box-shaped artifacts in the outputs

    Returns
    -------
    resampled : 2D array
        Resampled factor at the original sharpened image resolution
    """
    # Resample the local image scale to the output resolution
    if smoothed:
        factor = gu.binomialSmoother(factor)
        resample_alg = "bilinear"
    else:
        resample_alg = "near"

    proj, *_ = gu.getRasterInfo(str(template_file))
    fid = gu.saveImg(factor, lr_gt, proj, "MEM", noDataValue=np.nan)
    _, _, _, _, extent, _ = gu.getRasterInfo(fid)
    res_fid = gu.resampleWithGdalWarp(fid, str(template_file), "MEM",
                                      resampleAlg=resample_alg)

    resampled = res_fid.GetRasterBand(1).ReadAsArray()
    # Sometimes there can be 1 HR pixel NaN border around LR invalid pixel
    # due to resampling.
    # Fuction below fixes this.
    # Image border pixels are excluded due to numba stencil limitations.
    resampled[1:-1, 1:-1] = gu.removeEdgeNaNs(resampled)[1:-1, 1:-1]
    return resampled


def raster_data(input_file_path, bands=None):
    """Helper to read a GDAL image file

    Parameters
    ----------
    input_file_path : str of Path object
        Path to the input GDAL image file
    bands : list of int, optional
        List of bands (1-based) to extract from input_file_path.
        If None will extract all bands

    Returns
    -------
    array : 2D array or 3D array
        Output numpy array
    """
    fid = gdal.Open(str(input_file_path), gdal.GA_ReadOnly)
    if isinstance(bands, type(None)):
        bands = range(1, fid.RasterCount + 1)
    array = []
    for band in bands:
        array.append(fid.GetRasterBand(band).ReadAsArray())
    del fid
    array = np.squeeze(np.array(array))
    return array
