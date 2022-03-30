"""Summarize hourly outputs from 1km WRF runs for
SERDP Fish and Fire to the monthly scale
"""

import argparse
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import rasterio as rio
import xarray as xr
from rasterio.warp import reproject, Resampling


def summarize_day_array(arr, varname):
    """Summarize the hourly outputs array of a daily WRF file to
    the daily scale based on the summary variable supplied.
    
    Args:
        arr (numpy.ndarray): data array with shape (24, 209, 339)
        varname (str): name of the variable being summarized
        
    Returns:
        out_arr (numpy.ndarray): Single raster with shape (209, 339)
            containing summarized values
    """
    if varname in ["tsk", "t2"]:
        return arr.mean(axis=0)
    elif varname in ["tskmax", "t2max"]:
        return arr.max(axis=0)
    elif varname in ["prtot", "prday"]:
        return arr.sum(axis=0)
    
    
def summarize_daily(fp):
    """Open a daily WRF netcdf file and summarize hourly
    outputs to the daily scale for desired variables
    
    Args:
        fp (path-like): file path of WRF output file to be summarized
        
    Returns:
        summary_di (dict): dict with summary variable names as keys and
            the daily summary arrays as values
    """
    # names of target summary variables: WRF variable names
    varnames = {
        "tsk": "TSK", # mean skin temp
        "tskmax": "TSK", # max skin temp of the month
        "t2": "T2", # max 2m temp
        "t2max": "T2", # max 2m temp value of the month
        "prtot": "PCPNC", # total accumulation over month
        "prday": "PCPNC", # monthly mean daily precip
    }
    with xr.open_dataset(fp) as ds:
        summary_di = {}
        for varname in varnames:
            wrf_varname = varnames[varname]
            summary_di[varname] = summarize_day_array(ds[wrf_varname].values, varname)
    
    date = fp.name.split("_")[1]
    out_di = {
        "year": date.split("-")[0],
        "month": date.split("-")[1],
        "summary_dict": summary_di,
    }
    
    return out_di


def summarize_month_array(arr, varname):
    """Summarize the daily summaries of all daily WRF files om a month to
    the monthly scale based on the summary variable supplied.
    
    Args:
        arr (numpy.ndarray): data array with shape (~30, 209, 339)
        varname (str): name of the variable being summarized
        
    Returns:
        out_arr (numpy.ndarray): Single raster with shape (209, 339)
            containing summarized monthly values
    """
    if varname in ["tsk", "t2", "prday"]:
        return arr.mean(axis=0)
    elif varname in ["tskmax", "t2max"]:
        return arr.max(axis=0)
    elif varname == "prtot":
        return arr.sum(axis=0)


def write_raster(arr, dst_fp, spatial_di):
    """Reproject a numpy array that is on the original WRF grid to EPSG:3338
    and write to disk
    
    Args:
        arr (np.ndarray): summarized monthly array to be written
        dst_fp (pathlib.PosixPath): destination filepath
        spatial_di (dict): dict of spatial info derived from processing
            notebook including metadata for writing geotiffs with rasterio
        
    Returns:
        None, writes raster to disk
    """
    # using global transform and CRS info defined above for convenience
    height = spatial_di["meta"]["height"]
    width = spatial_di["meta"]["width"]
    meta = spatial_di["meta"]
    dst_arr = np.zeros((height, width), np.float32)
    # WRF transform was created with decreasing Y coordinates along
    #  increasing array row indices, but the summarized data will be
    #  on the native grid with increasing Y coordinates so the array
    #  needs to be flipped vertically
    arr = np.flipud(arr)
    reproject(
        arr,
        dst_arr,
        src_transform=spatial_di["src_transform"],
        src_crs=spatial_di["wrf_crs"],
        dst_transform=meta["transform"],
        dst_crs=meta["crs"],
        dst_nodata=meta["nodata"],
        resampling=Resampling.bilinear
    )
    
    # clean up data for writing - convert and round
    varname = dst_fp.name.split("_")[0]
    if varname in ["tsk", "tskmax", "t2", "t2max"]:
        dst_arr = dst_arr - 273.15
    # delete
    month = dst_fp.name.split("_")[-1].split(".")[0]
    if (varname == "t2") & (month == "06"):
        new_meta = meta.copy()
        new_meta.update({"height": 209, "width": 339})
        with rio.open(dst_fp.parent.joinpath("test_era_1980_06_summary.tif"), "w", **new_meta) as dst:
            dst.write(arr, 1)
        
    dst_arr = np.round(dst_arr, 1)
    
    
    #write it
    with rio.open(dst_fp, "w", **meta) as dst:
        dst.write(dst_arr, 1)

        
    return


def write_summaries(summary_dicts, model, out_dir, spatial_di):
    """Computes the monthly summary results for a particular year
    (stored in a list of dicts of daily summaries) and writes it
    to a file in the output directory.
    
    Args:
        summary_dicts (list): list of dicts of daily summaries of hourly
            WRF outputs
        model (str): name of model being worked on as seen in wrf_dir paths
        out_dir (pathlib.PosixPath): path to directory containing directories
            named by variable, where summary files should be written
        spatial_di (dict): dict of spatial info derived from processing
            notebook including metadata for writing geotiffs with rasterio
        
    Returns:
        None, writes files.
    """
    if len(summary_dicts) == 0:
        return ["No data"]
    
    # take the month year and varname from the first entry in the dict
    month, year = [summary_dicts[0][key] for key in ["month", "year"]]
    # iterate over variable names in first summary_dict
    dst_fps = []
    for varname in summary_dicts[0]["summary_dict"]:
        dst_fp = out_dir.joinpath(
            model, varname, f"{varname}_monthly_wrf_{model}_{year}_{month}.tif"
        )
        # stack arrays and summarize the daily to monthly
        arr = np.array([di["summary_dict"][varname] for di in summary_dicts])
        arr = summarize_month_array(arr, varname)
        write_raster(arr, dst_fp, spatial_di)
        dst_fps.append(dst_fp)
        
    return dst_fps


def run_monthly_summary(wrf_dir, out_dir, spatial_di, ncores):
    """Run the summarization to monthly scale in parallel,
    iterating over years (for progress)
    
    Args:
        wrf_dir (pathlib.PosixPath): path to the directory containing the
            daily WRF outputs to be summarized for a single model / year
        out_dir (pathlib.PosixPath): path to directory containing directories
            named by variable where summary files should be written
        spatial_di (dict): dict of spatial info derived from processing
            notebook including metadata for writing geotiffs with rasterio
        ncores (int): number of cores to use with multiprocessing.Pool
            
    Returns:
        None, only writes the files.
    """
    fps = list(wrf_dir.glob("*.nc"))
    
    with Pool(ncores) as pool:
        summary_dicts = pool.map(summarize_daily, fps)
        
    # breakup summary dicts by month and summarize and write
    args = [
        (
            [
                out_di for out_di in summary_dicts
                if str(month).zfill(2) in list(out_di.values())
            ], 
            wrf_dir.parent.name,
            out_dir,
            spatial_di,
        )
        for month in range(1, 13)
    ]
 

    with Pool(12) as pool:
        dst_fps = pool.starmap(write_summaries, args)

    return dst_fps


if __name__ == "__main__":
    # track time
    tic = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="program to summarize some hourly WRF outputs (in daily files) to monthly scale"
    )
    parser.add_argument(
        "-w",
        "--wrf_dir",
        action="store",
        dest="wrf_dir",
        type=str,
        help=(
            "path to WRF outputs directory containing the raw WRF"
            "outputs to be summarized (should be a single model / year)"
        ),
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        action="store",
        dest="out_dir",
        type=str,
        help=(
            "path to output directory (should be directory of subdirs"
            "named by variable, for a single model)"
        ),
    )
    parser.add_argument(
        "-p",
        "--proj_pkl_fp",
        action="store",
        dest="proj_pkl_fp",
        type=str,
        help="path to pickled sptial info derived in the processing notebook",
    )
    parser.add_argument(
        "-nc",
        "--ncores",
        action="store",
        dest="ncores",
        type=int,
        help="number of cores",
    )

    args = parser.parse_args()
    wrf_dir = Path(args.wrf_dir)
    out_dir = Path(args.out_dir)
    proj_pkl_fp = Path(args.proj_pkl_fp)
    ncores = args.ncores
    
    # read in the spatial info derived from processing notebook
    with open(proj_pkl_fp, "rb") as pkl:
        spatial_di = pickle.load(pkl)
        
    dst_fps = run_monthly_summary(wrf_dir, out_dir, spatial_di, ncores)
    print("Following files written:")
    _ = [print(fp) for var_fps in dst_fps for fp in var_fps]
    print(f"\nTime elapsed: {round(time.perf_counter() - tic)}s")
