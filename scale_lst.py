from pathlib import Path
from osgeo import gdal
import agrotig_input_creator.gdal_utils as gu  # Replace for your standard senet gdal_utils
import numpy as np
import multiprocessing as mp
from numpy.lib.stride_tricks import as_strided
from pyDMS import pyDMSUtils as utils
import datetime as dt
from scipy import stats as st
import skimage.morphology as morph
# Libraries for plotting results
import matplotlib.pyplot as plt
import model_evaluation.double_collocation as dc

WARP_OPTIONS = {"multithread": True,
                "warpOptions": ["NUM_THREADS=%i" % mp.cpu_count()]}

IMAGE_NAME_TEMPLATE = "L???_????_%s_%s_*"
LST_BAND_NAME = {"LE07": "ST_B6", "LC08": "ST_B10"}


def reclassify_espamask(lc_array, buffer_clouds=1, buffer_shadow=0):
    """Run FMask 4.0 through Docker on input SAFE.
    Reclassifies fmask output map such that all the pixels which should be masked have a value of
    zero, and pixels which should not be masked have a value of 1.
    Parameters:
        mask_map (str):
            Path to fmask output map.
        mask_clouds (bool):
            True if clouds should be masked.
        mask_shadow (bool):
            True if shadows should be masked.
        mask_water (bool):
            True if water should be masked.
        mask_snow (bool):
            True if snow should be masked.
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


def moving_window_filter(in_array, win_size):
    if ~ win_size % 2 == 1:
        print('Warning, VAR_FILTER_SIZE must be ODD Integer number')

    size = int((win_size - 1) / 2)
    # Extend the array and replicate the outer borders
    in_array = np.pad(in_array, pad_width=size, mode='reflect')

    # Create Strided array to vectorize the filter window
    img_strided = as_strided(in_array, shape=(in_array.shape[0] - win_size + 1,
                                              in_array.shape[1] - win_size + 1,
                                              win_size,
                                              win_size),
                             strides=in_array.strides * 2)
    std_img = np.nanstd(img_strided, axis=(-1, -2))
    mean_img = np.nanmean(img_strided, axis=(-1, -2))
    return mean_img, std_img


def rescale_lst(landsat_folder, target_file, ref_date, tiles,
                output_file=None, max_days_offset=-16, window_size=(1000, 1000),
                smoth_scale=False):
    """ Rescale LST dynamic range based on a reference LST.

    It aims to equalize the spatial variability of temperatures from a target image
    using another image as reference.

    Parameters
    ----------
    reference_file : str
        Path to the reference image which will be used to match its dynamic range
    target_file : str
        Path to the objective image which will be rescaled to match the dynamic
        range of reference_file
    output_file : str, optional
        If provided, path to the output file where the corrected LST will be saved
    window_size : tuple
        tuple of two floats with the spatial extent, in target_file projection units
         of the matching window

    Returns
    -------
    lst_corr : array
        Rescaled LST array
    """
    # Reproject reference file to target extent and resolution
    proj, gt, xsize, ysize, extent, _ = gu.raster_info(str(target_file))

    # Aggregate target_file to match the resolution of window_size
    lr_fid = gdal.Warp("MEM",
                       str(target_file),
                       format="MEM",
                       dstSRS=proj,
                       xRes=window_size[0],
                       yRes=-window_size[1],
                       outputBounds=extent,
                       resampleAlg="average",
                       **WARP_OPTIONS)

    lst_dms = gu.raster_data(str(target_file))
    dms_fid = gdal.Open(str(target_file), gdal.GA_ReadOnly)
    # Compute the relative scale between reference and target, based of the ratio
    # between coefficients of variation
    mean_dms, std_dms = utils.resampleHighResToLowRes(dms_fid, lr_fid)
    mean_dms = mean_dms[:, :, 0]
    std_dms = std_dms[:, :, 0]
    cv_dms = np.full_like(std_dms, np.nan)
    valid = np.logical_and(np.isfinite(std_dms), mean_dms != 0)
    cv_dms[valid] = std_dms[valid] / mean_dms[valid]
    del std_dms
    cv_ref, days_offset = composite_reference_variability(landsat_folder,
                                                          ref_date,
                                                          dms_file,
                                                          tiles,
                                                          max_days_offset=max_days_offset,
                                                          window_size=win_size)

    valid = np.logical_and(np.isfinite(cv_ref), np.isfinite(cv_dms))
    scale = np.full(cv_ref.shape, 1.0)
    scale[valid] = cv_ref[valid] / cv_dms[valid]
    # Resample the local image scale to the output resolution
    if smoth_scale:
        scale = utils.binomialSmoother(scale)
        scale_fid = utils.saveImg(scale, lr_fid.GetGeoTransform(),
                                  lr_fid.GetProjection(), "MEM", noDataValue=np.nan)

        scale_res = utils.resampleWithGdalWarp(scale_fid, dms_fid,
                                               resampleAlg="bilinear")

        # Bilinear resampling extrapolates by half a pixel, so need to clean it up
        scale = scale_res.GetRasterBand(1).ReadAsArray()
    else:
        scale_fid = utils.saveImg(scale, lr_fid.GetGeoTransform(),
                                  lr_fid.GetProjection(), "MEM", noDataValue=np.nan)

        scale_res = utils.resampleWithGdalWarp(scale_fid, dms_fid, resampleAlg="near")
        scale = scale_res.GetRasterBand(1).ReadAsArray()

    del scale_fid, scale_res
    # Sometimes there can be 1 HR pixel NaN border arond LR invalid pixels due to resampling.
    # Fuction below fixes this. Image border pixels are excluded due to numba stencil
    # limitations.
    scale[1:-1, 1:-1] = utils.removeEdgeNaNs(scale)[1:-1, 1:-1]
    if output_file:
        scale_file = output_file.parent / f"{output_file.stem}_scale.tif"
        gu.save_image(scale, gt, proj, str(scale_file), no_data_value=np.nan)

    if smoth_scale:
        # Resample the local window average to the output resolution
        mean_dms = utils.binomialSmoother(mean_dms)
        mean_fid = utils.saveImg(mean_dms, lr_fid.GetGeoTransform(),
                                 lr_fid.GetProjection(), "MEM", noDataValue=np.nan)

        mean_res = utils.resampleWithGdalWarp(mean_fid, dms_fid, resampleAlg="bilinear")
        mean_dms = mean_res.GetRasterBand(1).ReadAsArray()

    else:
        # Resample the local window average to the output resolution
        mean_fid = utils.saveImg(mean_dms, lr_fid.GetGeoTransform(),
                                 lr_fid.GetProjection(), "MEM", noDataValue=np.nan)

        mean_res = utils.resampleWithGdalWarp(mean_fid, dms_fid, resampleAlg="near")
        mean_dms = mean_res.GetRasterBand(1).ReadAsArray()

    del mean_res, mean_fid, lr_fid, dms_fid
    # Sometimes there can be 1 HR pixel NaN border arond LR invalid pixels due to resampling.
    # Fuction below fixes this. Image border pixels are excluded due to numba stencil
    # limitations.
    mean_dms[1:-1, 1:-1] = utils.removeEdgeNaNs(mean_dms)[1:-1, 1:-1]

    lst_corr = lst_dms.copy()
    valid = np.logical_and(np.isfinite(lst_dms), np.isfinite(scale))
    lst_corr[valid] = apply_bias_correction(lst_dms[valid], mean_dms[valid], scale[valid])
    if output_file:
        gu.save_image(lst_corr, gt, proj, str(output_file), no_data_value=np.nan)

    return lst_corr


def composite_reference_variability(reference_folder, ref_date, image_template, tiles,
                                    max_days_offset=-16, output_file=None,
                                    window_size=(1000, 1000)):
    """ Builds temporal composites using several compositing methods

    Parameters
    ----------
    reference_folder : str
        Path to the input granule location
    ref_date : object
        Datetime object with the reference date for the compositing period
    max_days_offset : int
        Maximum number of days backwards allowed to search cloud-free pixels.
        If positive search both backwards and forwards in the future of ref_date, for NTC applications
        If negative only search backwards in the past of ref_date, for NRT applications
    tiles : list
        Names of the tiles to be used during the compositing
    output_file : str
        Path to the output composite image

    Returns
    -------
    pos : bytearray
        Locations (M) in the stack for the composite.
        If return_date is True it will return the date in days since 1 January 1970
    mask_composite : bytearray
        Masked array (M) of valid observations in the composite
    output_composite : array
        Composite image
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

    images, dates = stack_image_timeseries(reference_folder, date_start, date_end, tiles)
    cv_ref, day_offset = make_composite(images, ref_date, dates, image_template,
                                        window_size=window_size, method=method)

    if output_file:
        proj, gt, *_ = gu.raster_info(str(image_template))
        gt = list(gt)
        gt[1] = window_size[0]
        gt[5] = -window_size[1]
        gu.save_image(cv_ref, gt, proj, str(output_file), no_data_value=np.nan)

    return cv_ref, day_offset


def apply_bias_correction(src_array, mean_src, scale):
    output = mean_src + scale * (src_array - mean_src)
    return output


def local_variability(input_image, image_template, window_size=(1000, 1000)):
    proj, gt, xsize, ysize, extent, _ = gu.raster_info(str(image_template))
    lr_fid = gdal.Warp("MEM",
                       str(image_template),
                       format="MEM",
                       dstSRS=proj,
                       xRes=window_size[1],
                       yRes=-window_size[0],
                       outputBounds=extent,
                       resampleAlg="near",
                       **WARP_OPTIONS)

    # Transform UINT16 to Kelvin
    mult, add = landsat_temp_factors(input_image)
    raster = gu.raster_data(str(input_image / f"{input_image.stem}_QA_PIXEL.TIF"))
    valid = reclassify_espamask(raster, buffer_clouds=1, buffer_shadow=0)
    sensor = input_image.stem[:4]
    raster = input_image / f"{input_image.stem}_{LST_BAND_NAME[sensor]}.TIF"
    lst = gu.raster_data(str(raster)).astype(float)
    lst[valid] = add + mult * lst[valid]
    lst[~valid] = np.nan
    landsat_proj, landsat_gt, *_ = gu.raster_info(str(raster))
    in_fid = gu.save_image(lst, landsat_gt, landsat_proj, "MEM")
    # Compute image local statistics
    mean, std = utils.resampleHighResToLowRes(in_fid, lr_fid)
    mean = mean[:, :, 0]
    std = std[:, :, 0]
    valid = np.logical_and(np.isfinite(std), mean != 0)
    cv = np.full_like(mean, np.nan)
    cv[valid] = std[valid] / mean[valid]
    return cv


def make_composite(image_stack, date_obj, dates, image_template, window_size=(1000, 1000),
                   method="last"):
    """
    Prepares the composite image locations using the latest available observation

    Parameters
    ----------
    image_stack : array
        Input Stack of images (n, M) with n observations and M pixels
    no_data_value : float
        Fill value for `image_stack`. Default NaN
    mask_stack: array or None
        Stack of masks (n, M) for n observations  and M pixels

    Returns
    -------
    pos : bytearray
        Locations (M) in the stack for the composite
    mask_composite : bytearray
        Masked array (M) of valid observations in the composite
    """
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
        array = local_variability(image, image_template, window_size=window_size)
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


def landsat_temp_factors(image_path):
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


def prostprocess_dms(dms_orig, dms_corr, output_file=None):

    lst_orig = gu.raster_data(str(dms_orig))
    lst_mean = np.nanmean(lst_orig)
    lst_std = np.nanstd(lst_orig)
    lst_corr = gu.raster_data(str(dms_corr))
    z_score = st.norm.cdf(lst_orig, loc=lst_mean, scale=lst_std)
    w = np.abs(z_score - 0.5) / 0.5
    lst_final = lst_corr.copy()
    valid = np.logical_and(np.isfinite(lst_orig), np.isfinite(lst_final))
    lst_final[valid] = lst_orig[valid] * (1. - w[valid]) + lst_corr[valid] * w[valid]
    del w, lst_corr, lst_orig
    if output_file:
        proj, gt, *_ = gu.raster_info(str(dms_orig))
        gu.save_image(lst_final, gt, proj, str(output_file))
    return lst_final


def _plot_results(ref, orig, corr, outfile, q=20):
    """

    Parameters
    ----------
    ref
    orig
    corr
    outfile
    q

    Returns
    -------

    """
    plot_params = {"s": 3, "alpha": 0.01, "marker": '.'}

    valid = np.logical_and.reduce((np.isfinite(ref),
                                   np.isfinite(orig),
                                   np.isfinite(corr)))
    ref = ref[valid]
    orig = orig[valid]
    corr = corr[valid]
    lst_min = np.nanmin([ref, orig, corr])
    lst_max = np.nanmax([ref, orig, corr])
    lst_lims = (lst_min, lst_max)

    fig, axs = plt.subplots(nrows=2, figsize=(10, 10), sharex=True)
    dc.density_plot(ref, orig, axs[0], **plot_params)
    axs[0].plot(lst_lims, lst_lims, "k--")

    mean_bias, mae, rmse = dc.error_metrics(ref, orig)
    rmse_s, rmse_u = dc.rmse_wilmott_decomposition(ref, orig)
    scale = dc.scaling_factor(ref, orig)
    cor, p_value, slope, intercept, d = dc.agreement_metrics(ref, orig)

    string = f"N    = {np.sum(valid)}\n"\
             f"RMSE = {rmse:>5.2f} K\n"\
             f"    RMSE$_s$ = {rmse_s:>5.2f}\n"\
             f"    RMSE$_u$ = {rmse_u:>5.2f}\n"\
             f"bias = {mean_bias:>5.2f} K\n"\
             f"scale= {scale:>5.2f}\n"\
             f"r    = {cor:>5.2f}\n"\
             f"d    = {d:>5.2f}"

    axs[0].text(0.05, 0.95,
                string,
                fontsize=10,
                backgroundcolor='white',
                family='monospace',
                linespacing=1,
                verticalalignment="top",
                transform=axs[0].transAxes)

    valid = np.isfinite(ref)
    extreme_thres = np.percentile(ref[valid], q), \
                    np.percentile(ref[valid], 100 - q)
    hot_col = np.logical_or(ref < extreme_thres[0],
                            ref > extreme_thres[1])

    mean_bias, mae, rmse = dc.error_metrics(ref[hot_col], orig[hot_col])
    rmse_s, rmse_u = dc.rmse_wilmott_decomposition(ref[hot_col], orig[hot_col])
    scale = dc.scaling_factor(ref[hot_col], orig[hot_col])
    cor, p_value, slope, intercept, d = dc.agreement_metrics(ref[hot_col], orig[hot_col])

    string = "Only extreme temperatures\n"\
             f"N    = {np.sum(hot_col)}\n"\
             f"RMSE = {rmse:>5.2f} K\n"\
             f"    RMSE$_s$ = {rmse_s:>5.2f}\n"\
             f"    RMSE$_u$ = {rmse_u:>5.2f}\n"\
             f"bias = {mean_bias:>5.2f} K\n"\
             f"scale= {scale:>5.2f}\n"\
             f"r    = {cor:>5.2f}\n"\
             f"d    = {d:>5.2f}"

    axs[0].text(0.65, 0.05,
                string,
                fontsize=10,
                backgroundcolor='white',
                family='monospace',
                linespacing=1,
                verticalalignment="bottom",
                transform=axs[0].transAxes)

    axs[0].set_ylim(lst_lims)
    axs[0].grid(color='silver', linestyle=':', linewidth=0.5)
    axs[0].set_ylabel("LST DMS (K)")

    dc.density_plot(ref, corr, axs[1], **plot_params)
    axs[1].plot(lst_lims, lst_lims, "k--")

    mean_bias, mae, rmse = dc.error_metrics(ref, corr)
    rmse_s, rmse_u = dc.rmse_wilmott_decomposition(ref, corr)
    scale = dc.scaling_factor(ref, corr)
    cor, p_value, slope, intercept, d = dc.agreement_metrics(ref, corr)

    string = f"N    = {np.sum(valid)}\n"\
             f"RMSE = {rmse:>5.2f} K\n"\
             f"    RMSE$_s$ = {rmse_s:>5.2f}\n"\
             f"    RMSE$_u$ = {rmse_u:>5.2f}\n"\
             f"bias = {mean_bias:>5.2f} K\n"\
             f"scale= {scale:>5.2f}\n"\
             f"r    = {cor:>5.2f}\n"\
             f"d    = {d:>5.2f}"

    axs[1].text(0.05, 0.95,
                string,
                fontsize=10,
                backgroundcolor='white',
                family='monospace',
                linespacing=1,
                verticalalignment="top",
                transform=axs[1].transAxes)
    mean_bias, mae, rmse = dc.error_metrics(ref[hot_col], corr[hot_col])
    rmse_s, rmse_u = dc.rmse_wilmott_decomposition(ref[hot_col], corr[hot_col])
    scale = dc.scaling_factor(ref[hot_col], corr[hot_col])
    cor, p_value, slope, intercept, d = dc.agreement_metrics(ref[hot_col], corr[hot_col])

    string = "Only extreme temperatures\n"\
             f"N    = {np.sum(hot_col)}\n"\
             f"RMSE = {rmse:>5.2f} K\n"\
             f"    RMSE$_s$ = {rmse_s:>5.2f}\n"\
             f"    RMSE$_u$ = {rmse_u:>5.2f}\n"\
             f"bias = {mean_bias:>5.2f} K\n"\
             f"scale= {scale:>5.2f}\n"\
             f"r    = {cor:>5.2f}\n"\
             f"d    = {d:>5.2f}"

    axs[1].text(0.65, 0.05,
                string,
                fontsize=10,
                backgroundcolor='white',
                family='monospace',
                linespacing=1,
                verticalalignment="bottom",
                transform=axs[1].transAxes)

    axs[1].set_ylim(lst_lims)
    axs[1].grid(color='silver', linestyle=':', linewidth=0.5)
    axs[1].set_ylabel("corrected LST DMS (K)")

    axs[1].set_xlabel("Landsat LST (LST)")
    axs[1].set_xlim(lst_lims)
    plt.suptitle(datestr)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(outfile)
    plt.close()


if __name__ == "__main__":
    tiles = ["199033", "200033"]
    workdir = Path() / "test"
    landsat_dir = workdir / "landsat/"  # Input folder with Landsat LST tiles used for correction
    dms_dir = workdir / "dms_raw"  # Input Folder with original sharpened images
    out_dir = workdir / "dms_corr_best_case"  # Output folder with corrected sharpened images
    final_correction = True  # Postprocessing flag, if true apply the a weighted correction based on how extreme the temperature is
    out_dir_final = workdir / "dms_corr_best_case_final"  # Output folder with postprocessed sharpened images
    lst_ref_dir = workdir / "landsat_reference"  # Folder with reference Landsat LST images for validation
    win_size = (1000, 1000)  # Local window size in DMS reference units to apply the correction, typically use the original S3 LST resolution size

    overwrite = False

    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)

    dms_files = dms_dir.glob("*.tif")
    for dms_file in dms_files:
        filename = dms_file.stem
        datestr = filename[5:13]

        ref_date = dt.datetime.strptime(datestr, "%Y%m%d")
        output_file = out_dir / f"{filename}_corrected.tif"
        if output_file.exists() and not overwrite:
            lst_corr = gu.raster_data(str(output_file))

        else:
            print(f"LST Bias correction for {filename}")
            lst_corr = rescale_lst(landsat_dir, dms_file, ref_date, tiles,
                                   output_file=output_file, max_days_offset=-16,
                                   window_size=win_size, smoth_scale=True)

        if final_correction:
            lst_corr = prostprocess_dms(dms_file, output_file,
                                        output_file=out_dir_final / f"{filename}_corrected_final.tif")

        lst_corr = lst_corr.reshape(-1)
        landsat_file = list(lst_ref_dir.glob(f"L???_????_??????_{datestr}_*_LST.tif"))
        l_fid = gdal.Open(str(landsat_file[0]), gdal.GA_ReadOnly)
        dms_fid = gdal.Open(str(dms_file), gdal.GA_ReadOnly)
        lst_orig = dms_fid.GetRasterBand(1).ReadAsArray().reshape(-1)
        lst_ref = gu.resampleWithGdalWarp(l_fid, dms_fid, "near")
        del dms_fid, l_fid
        lst_ref = lst_ref.GetRasterBand(1).ReadAsArray().reshape(-1)
        valid = np.logical_and.reduce((np.isfinite(lst_orig),
                                       np.isfinite(lst_ref),
                                       np.isfinite(lst_corr)))
        lst_ref = lst_ref[valid]
        lst_orig = lst_orig[valid]
        lst_corr = lst_corr[valid]


        outfile = out_dir_final / f"DMS_comparison_{datestr}.png"
        print(f"Plotting comparison graphs for {datestr}")
        _plot_results(lst_ref, lst_orig, lst_corr, outfile, q=10)

        del lst_ref, lst_orig, lst_corr

