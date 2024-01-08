from pathlib import Path
import datetime as dt
import numpy as np
from pyDMS import pyDMSUtils as gu
from osgeo import gdal
from dms_bias_correction import landsat_collection_2_helper as hlp
import matplotlib.pyplot as plt
import model_evaluation.double_collocation as dc

tiles = ["199033", "200033"]
workdir = Path() / "test"
landsat_dir = workdir / "landsat/"  # Input folder with Landsat LST tiles used for correction
# Input Folder with original sharpened images
dms_dir = workdir / "dms_raw"
# Output folder with corrected sharpened images
out_dir = workdir / "dms_corr_best_case"
# Folder with reference Landsat LST images for validation
lst_ref_dir = workdir / "landsat_reference"
# Local window size in DMS reference units to apply the correction,
# typically use the original S3 LST resolution size
win_size = (1000,  1000)
overwrite = True

def run_test(dms_dir, landsat_dir, tiles, out_dir, lst_ref_dir):
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)

    dms_files = dms_dir.glob("*.tif")
    for dms_file in dms_files:
        filename = dms_file.stem
        datestr = filename[5:13]

        ref_date = dt.datetime.strptime(datestr, "%Y%m%d")
        output_file = out_dir / f"{filename}_corrected.tif"
        if output_file.exists() and not overwrite:
            lst_corr = hlp.sl.raster_data(str(output_file))

        else:
            print(f"LST Bias correction for {filename}")
            lst_corr = hlp.rescale_lst(landsat_dir,
                                       ref_date,
                                       dms_file,
                                       tiles,
                                       output_file=output_file,
                                       max_days_offset=-16,
                                       window_size=win_size,
                                       smooth_scale=True)

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
        outfile = out_dir / f"DMS_comparison_{datestr}.png"
        print(f"Plotting comparison graphs for {datestr}")
        _plot_results(lst_ref, lst_orig, lst_corr, datestr, outfile, q=5)

        del lst_ref, lst_orig, lst_corr


def _plot_results(ref, orig, corr, datestr, outfile, q=10):
    """

    Parameters
    ----------
    ref : array_like
        reference Landsat LST array
    orig : array_like
        Original Sharpened LST array
    corr : array_like
        Corrected Sharpened LST array
    outfile : str of Path object
        Output graph file
    q : float
        Percentile to evaluate tha hottest and coldest temperatures in the scene

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
    run_test(dms_dir, landsat_dir, tiles, out_dir, lst_ref_dir)


