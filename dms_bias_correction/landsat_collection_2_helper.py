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
This module contains higher-level functions to perform the enhancement
the dynamic range in sharpened LST images using Landsat 2 Collection 2
as reference LST source based on [Guzinski_2023]_.

MODULE CONTENTS
================
* :func:`rescale_lst` Main function to rescale LST dynamic range based on a set of
Landsat Collection 2 Level 2 scenes.
* :func:`cv_landsat_collection_2` Computers the Coefficient of Variation of a
Landsat 2 Collection 2 scene.
* :func:`landsat_temp_factors` Get scale and offset for scaling LST to Kelvin.
* :func:`make_composite` Prepares the composite image locations using the closest
available clear-sky observation.
* :func:`stack_image_timeseries` Stacks all images that ar within two dates.
* :func:`reclassify_espamask` Reclassifies fmask output map into a binary
clear sky array.
* :func:`composite_reference_variability` Builds the reference
coefficient of variation

REFERENCES
==========
.. [Guzinski_2023] R. Guzinski, H. Nieto, R. Ramo-Sánchez, J.M. Sánchez, I. Joma,
  R. Zitouna-Chebbi, O. Roupsard, R. López-Urrea",
  Improving field-scale crop actual evapotranspiration monitoring with
  Sentinel-3, Sentinel-2, and Landsat data fusion (2023)
  International Journal of Applied Earth Observation and Geoinformation",
  125, 103587, DOI:10.1016/j.jag.2023.103587,
'''

import datetime as dt
from pathlib import Path
import numpy as np
from dms_bias_correction import scale_lst as sl
from skimage import morphology as morph

IMAGE_NAME_TEMPLATE = "L???_????_%s_%s_*"
LST_BAND_NAME = {"LE07": "ST_B6", "LC08": "ST_B10", "LC09": "ST_B10"}


def rescale_lst(landsat_base_folder,
                ref_date,
                target_file,
                tiles,
                max_days_offset=-16,
                output_file=None,
                window_size=(1000, 1000),
                smooth_scale=False):
    """Main function to rescale LST dynamic range based on a set of
    Landsat Collection 2 Level 2 scenes.

    It aims to equalize the spatial variability of temperatures from a target image
    using another image as reference.

    Parameters
    ----------
    landsat_base_folder : str
        Path to the input Landsat granule location
    ref_date : object
        Datetime object with the reference date for the compositing period
    target_file : str
        Path to the sharpened image that will be used for warping
    tiles : list
        Names of the WNS tiles to be used during the compositing
    max_days_offset : int
        Maximum number of days backwards allowed to search cloud-free pixels.
        If positive search both backwards and forwards in the future of
        ref_date, for NTC applications
        If negative only search backwards in the past of
        ref_date, for NRT applications
    output_file : str, optional
        Path to the output composite image
    window_size : tuple of float
        Spatial extent in target_file projection units of the matching window
    smooth_scale : bool, optional
        Whether smooth the coarse resolution factors
        to minimize box-shaped artifacts in the outputs

    Returns
    -------
    lst_corr : array
        Rescaled LST array
    """
    proj, hr_gt, xsize, ysize, extent, *_ = sl.gu.getRasterInfo(str(target_file))
    hr_res = hr_gt[1], hr_gt[5]
    cv_dms, mean_dms, lr_gt = sl.local_variability(target_file, proj, extent,
                                                   window_size=window_size)
    cv_ref, days_offset = composite_reference_variability(
        landsat_base_folder, ref_date, target_file,  tiles,
        max_days_offset=max_days_offset, window_size=window_size)

    scale = sl.compute_scale(cv_ref, cv_dms)
    if output_file:
        scale_file = output_file.parent / f"{output_file.stem}_scale.tif"
        sl.gu.saveImg(scale, lr_gt, proj, str(scale_file), noDataValue=np.nan)

    scale = sl.resample_factor(scale, lr_gt, target_file,
                               smoothed=smooth_scale)

    mean_dms = sl.resample_factor(mean_dms, lr_gt, target_file,
                                  smoothed=smooth_scale)
    lst_dms = sl.raster_data(target_file)
    lst_corr = lst_dms.copy()
    valid = np.logical_and(np.isfinite(lst_dms), np.isfinite(scale))
    lst_corr[valid] = sl.apply_bias_correction(lst_dms[valid], mean_dms[valid], scale[valid])
    if output_file:
        sl.gu.saveImg(lst_corr, hr_gt, proj, str(output_file),
                      noDataValue=np.nan)

    return lst_corr


def cv_landsat_collection_2(input_image,
                            image_template,
                            window_size=(1000, 1000)):
    """Computers the Coefficient of Variation of a Landsat 2 Collection 2 scene

    Parameters
    ----------
    input_image : str or Path object
        Input Landsat Collection 2 Level 2 Surface Temperature File
    image_template : str
        Path to the sharpened image that will be used for warping
    window_size : tuple of float, optional
        Spatial extent in target_file projection units of the matching window

    Returns
    -------
    cv : array_like
        Coefficient of Variation composite
    """

    proj, gt, xsize, ysize, extent, _ = sl.gu.getRasterInfo(str(image_template))
    # Transform UINT16 to Kelvin
    mult, add = landsat_temp_factors(input_image)
    raster = sl.raster_data(str(input_image / f"{input_image.stem}_QA_PIXEL.TIF"))
    valid = reclassify_espamask(raster, buffer_clouds=1, buffer_shadow=0)
    sensor = input_image.stem[:4]
    raster = input_image / f"{input_image.stem}_{LST_BAND_NAME[sensor]}.TIF"
    lst = sl.raster_data(str(raster)).astype(float)
    lst[valid] = add + mult * lst[valid]
    lst[~valid] = np.nan
    # Save temporary LST file in K
    temp_file = input_image / f"{input_image.stem}_ST.TIF"
    sl.gu.saveImg(lst, gt, proj, str(temp_file), noDataValue=np.nan)
    cv, *_ = sl.local_variability(temp_file, proj, extent,
                                  window_size=window_size)
    # Delete temporary LST file in K
    temp_file.unlink()
    return cv


def landsat_temp_factors(image_path):
    """Get scale and offset for scaling LST to Kelvin

    Parameters
    ----------
    image_path : str or Path object
        Path to the Landsat folder scene

    Returns
    -------
    mult, add : float
        Offset and Scale to convert digital numbers to LST
    """
    image_path = Path(image_path)
    image_name = image_path.stem
    sensor = image_name[0:4]
    mtl_file = image_path / f"{image_name}_MTL.txt"
    with open(mtl_file, "r") as fid:
        for line in fid:
            if f"TEMPERATURE_MULT_BAND_{LST_BAND_NAME[sensor]}" in line:
                mult = float(line.rstrip("\r\n").split("=")[1])
            elif f"TEMPERATURE_ADD_BAND_{LST_BAND_NAME[sensor]}" in line:
                add = float(line.rstrip("\r\n").split("=")[1])

    return mult, add


def make_composite(image_stack,
                   date_obj,
                   dates,
                   image_template,
                   window_size=(1000, 1000),
                   method="last"):
    """
    Prepares the composite image locations using the closest
    available clear-sky observation

    Parameters
    ----------
    image_stack : list of str
        List with the path of the images that fall betwen the search dates
    date_obj : datetime Object
        Reference date that will be used during the search of the closest clear-sky pixel
    dates : list of datetime
        List of the p dates that correspond for each of the stack array elements
    iamge_template : str
        Path to the sharpened image that will be used for warping
    window_size : tuple of float, optional
        Spatial extent in target_file projection units of the matching window
    method : {"last", "closest"}
        Method that will be used during the compositing:
        > "closest" search both backwards and forwards in the future to
        find the closest date to ref_date, for NTC applications
        > "last" only search backwards in the past of ref_date,
        for NRT applications

    Returns
    -------
    cv : array_like
        Coefficient of Variation composite
    day_offset : array_like
        Days offset between the reference date and the date used for the given pixel
    """
    proj, gt, xsize, ysize, extent, _ = sl.gu.getRasterInfo(str(image_template))
    days_diff = np.array([date - date_obj for date in dates])
    if method == "last":
        days_diff, image_stack = zip(*list([(i[1], image_stack[i[0]])
                                            for i in sorted(enumerate(days_diff), reverse=True,
                                                            key=lambda x: x[1])]))
    elif method == "closest":
        days_diff, image_stack = zip(*list([(days_diff[i[0]], image_stack[i[0]])
                                            for i in sorted(enumerate(np.abs(days_diff)),
                                                            key=lambda x: x[1])]))

    for i, image in enumerate(image_stack):
        array = cv_landsat_collection_2(image,
                                        image_template,
                                        window_size=window_size)

        if i == 0:
            cv = array.copy()
            day_offset = np.full(cv.shape, np.nan)
            valid = np.isfinite(cv)
            day_offset[valid] = days_diff[i].days
        else:
            valid = np.logical_and(np.isnan(cv), np.isfinite(array))
            cv[valid] = array[valid]
            day_offset[valid] = days_diff[i].days

        # Check for all pixels filled
        finished = np.isfinite(cv)
        if np.all(finished):
            break

    return cv, day_offset


def stack_image_timeseries(input_folder, date_ini, date_end, tiles):
    """Stacks all images that ar within two dates

    Parameters
    ----------
    input_folder : str or Path object
    date_ini : datetime object
        Start date
    date_end : datetime object
        End date
    tiles : list of str
        Names of the WNS tiles to be used during the compositing

    Returns
    -------
    image_stack : list of str
        List with the path of the images that fall between the search dates
    dates : list of datetime
        List of the p dates that correspond for each of the stack array elements
    """
    image_stack = []
    dates = []
    while date_ini <= date_end:
        date_str = date_ini.strftime("%Y%m%d")
        for tile in tiles:
            subfolders = input_folder / tile
            match = list(subfolders.glob(IMAGE_NAME_TEMPLATE % (tile, date_str)))
            if len(match) > 0:
                image_stack.append(match[0])
                dates.append(date_ini)
        date_ini = date_ini + dt.timedelta(1)

    return image_stack, dates


def reclassify_espamask(lc_array, buffer_clouds=1, buffer_shadow=0):
    """Reclassifies fmask output map into a binary clear sky array

    All the pixels that should be masked have a value of zero,
    and pixels which should not be masked have a value of 1.

    Parameters
    ----------
    lc_array : array_like
        FMask QA_PIXEL array
    buffer_clouds : int, optional
        Buffer in pixel units that will be applied to clouds
    buffer_shadow : int, optional
        Buffer in pixel units that will be applied to shadows

    Returns
    -------
    valid : array_like
        Boolean mask with True the valid pixels.
    """

    # if clouds (bit 3) and low/medium/high probability (bit 8 and 9) then clouds
    clouds = ((lc_array & (1 << 3)) > 1) & ((lc_array & ((1 << 8) | (1 << 9))) > 1)
    # if cirrus (bit 2) and low/medium/high probability shadows (bit 14 and 15) then shadows
    cirrus = ((lc_array & (1 << 2)) > 1) & ((lc_array & ((1 << 14) | (1 << 15))) > 1)
    # if shadows (bit 4) and low/medium/high probability shadows (bit 10 and 11) then shadows
    shadow = ((lc_array & (1 << 4)) > 1) & ((lc_array & ((1 << 10) | (1 << 11))) > 1)

    if buffer_clouds > 0:
        se = morph.disk(buffer_clouds)
        clouds = morph.binary_dilation(clouds, se)
        cirrus = morph.binary_dilation(cirrus, se)

    # Find shadow flags
    if buffer_shadow > 0:
        se = morph.disk(buffer_shadow)
        shadow = morph.binary_dilation(shadow, se)

    # Final mask is without clouds nor snow, with sensor observations and succesful AOT
    valid = np.logical_and.reduce((~clouds, ~shadow, ~cirrus))

    valid[lc_array==1] = False
    return valid


def composite_reference_variability(landsat_base_folder,
                                    ref_date,
                                    image_template,
                                    tiles,
                                    max_days_offset=-16,
                                    output_file=None,
                                    window_size=(1000, 1000)):
    """ Builds the reference coefficient of variation

    The CV is built by time cmpositing all clear pixels that the closest
    nearest in time to a reference date.

    Parameters
    ----------
    landsat_base_folder : str
        Path to the input Landsat granule location
    ref_date : object
        Datetime object with the reference date for the compositing period
    image_template : str
        Path to the sharpened image that will be used for warping
    tiles : list
        Names of the WNS tiles to be used during the compositing
    max_days_offset : int
        Maximum number of days backwards allowed to search cloud-free pixels.
        If positive search both backwards and forwards in the future of ref_date, for NTC applications
        If negative only search backwards in the past of ref_date, for NRT applications
    output_file : str, optional
        Path to the output composite image

    Returns
    -------
    cv_ref : array_like
        Coefficient of Variation compoiste
    day_offset : array_like
        Days offset between the reference date and the date used for the given pixel
    """

    # Read and stack the available images for preparing the composite
    if max_days_offset <= 0:
        date_start = ref_date + dt.timedelta(max_days_offset)
        date_end = ref_date
        method = "last"

    else:
        date_start = ref_date - dt.timedelta(max_days_offset)
        date_end = ref_date + dt.timedelta(max_days_offset)
        method = "closest"

    images, dates = stack_image_timeseries(landsat_base_folder,
                                           date_start,
                                           date_end,
                                           tiles)

    cv_ref, day_offset = make_composite(images,
                                        ref_date,
                                        dates,
                                        image_template,
                                        window_size=window_size,
                                        method=method)

    if output_file:
        proj, gt, *_ = sl.gu.getRasterInfo(str(image_template))
        gt = list(gt)
        gt[1] = window_size[0]
        gt[5] = -window_size[1]
        sl.gu.saveImg(cv_ref, gt, proj, str(output_file), noDataValue=np.nan)

    return cv_ref, day_offset

