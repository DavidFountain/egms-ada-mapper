import geopandas as gpd
import pandas as pd
import os


def get_ada_location(
        base_dir: str,
        model_date: str | int,
        egms_date: str | int,
        product: str,
        country: str,
        geohaz_type: str,
        aoi_name: str,
        s1_path: str,
        ada_type: str,
        avg_vel_thr: str | float
):
    """Get the local folder destination for
    classified ADAs"""
    base_dir = base_dir.replace("{model_date}", str(model_date))
    base_dir = base_dir.replace("{egms_date}", str(egms_date))
    base_dir = base_dir.replace("{product}", product)
    base_dir = base_dir.replace("{country}", country)
    base_dir = base_dir.replace("{geohaz_type}", geohaz_type)
    base_dir = base_dir.replace("{aoi_name}", aoi_name)
    base_dir = base_dir.replace("{s1_path}", s1_path)
    base_dir = base_dir.replace("{ada_type}", ada_type)
    base_dir = base_dir.replace("{avg_vel_thr}", str(avg_vel_thr))

    if ada_type == "avgvel+":
        base_dir += "union/"
    return base_dir


def get_aoi_bounds_location(
        base_dir: str,
        country: str,
        geohaz_type: str,
        aoi_name: str,
):
    """Get the local folder destination for
    AOI locations"""
    base_dir = base_dir.replace("{country}", country)
    base_dir = base_dir.replace("{geohaz_type}", geohaz_type)
    base_dir = base_dir.replace("{aoi_name}", aoi_name)
    return base_dir


def get_pid_lookup_location(
        base_dir: str,
        egms_date: str,
        product: str,
        country: str,
        geohaz_type: str,
        aoi_name: str,
        s1_path: str,
):
    """Get the local folder destination for
    AOI locations"""
    base_dir = base_dir.replace("{egms_date}", egms_date)
    base_dir = base_dir.replace("{product}", product)
    base_dir = base_dir.replace("{country}", country)
    base_dir = base_dir.replace("{geohaz_type}", geohaz_type)
    base_dir = base_dir.replace("{aoi_name}", aoi_name)
    base_dir = base_dir.replace("{s1_path}", s1_path)
    return base_dir


def read_geo_data(fname: str, **kwargs):
    return gpd.read_parquet(
        fname, **kwargs
    )


def read_data(fname: str, **kwargs):
    return pd.read_parquet(
        fname, **kwargs
    )


def get_ts_filename(
        pid: str, pid_fname_df, egms_ts_dir
):
    fname = pid_fname_df.loc[
        pid_fname_df["pid"] == pid,
        "filename"
    ].item()
    return os.path.join(egms_ts_dir, fname)


def read_single_ts(
        fname: str, pid: str
):
    return read_data(
        fname, filters=[("pid", "==", pid)]
    )
