# Functions to manipulate data sets
import geopandas as gpd
import pandas as pd
import numpy as np
import json
from json import JSONEncoder
import os
import re
from collections import Counter
from datetime import datetime, timedelta
from dotenv import load_dotenv

gpd.options.io_engine = "pyogrio"

# Set global variables and read boundary files
load_dotenv("src/.env")
PROJECT_CRS = os.getenv("PROJECT_CRS")


def get_point_data(click_data):
    data = pd.read_json(json.dumps(click_data))
    return data.loc["pid", "properties"]


# def get_pid_from_pointclick(json_data):
#     gdf = convert_json_to_geodataframe(json_data)
#     return gdf["pid"].values


def convert_json_to_dataframe(json_dict) -> pd.DataFrame:
    """Convert JSON dict to GeoPandas GeoDataframe

    Parameters
    ----------
    json_dict : a JSON dict object with 'features' key

    Returns
    ----------
    GeoPandas GeoDataFrame
    """
    data = json.loads(json_dict)
    return (pd.DataFrame.from_features(data["features"]))


def get_timeseries_from_pid(df: pd.DataFrame, pid: str) -> pd.DataFrame:
    """Return the time series from EGMS dataset
    from a specific pid

    Parameters
    ----------
    df : EGMS data in pandas DataFrame format
    pid : pid value for time series to be filtered

    Returns
    ----------
    DataFrame containing only time series measurements for specific pid
    """
    date_cols = get_date_cols(df)
    return df[df["pid"] == pid][date_cols]


def get_date_cols(df: pd.DataFrame, date_format: str=r"^\d{8}$"):
    """Return the date columns from a dataframe
    that match the date format pattern

    Parameters
    ----------
    df : pandas DataFrame with dates in columns
    date_format : specific format of the date in the dataframe

    Returns
    ----------
    ordered list of date columns
    """
    date_cols = [col for col in df.columns if re.match(date_format, col)]
    return sorted(date_cols)


def create_date_range(
        sdate: str | datetime,
        edate: str | datetime,
        tdelta: int,
        date_format: str = "%Y%m%d") -> list:
    if isinstance(sdate, str):
        sdate = datetime.strptime(sdate, date_format)
    if isinstance(edate, str):
        edate = datetime.strptime(edate, date_format)

    total_num_samples = (edate - sdate).days // tdelta
    date_range = [sdate + timedelta(days=(i*tdelta))
                  for i in range(total_num_samples+1)]
    return [datetime.strftime(d, "%Y%m%d") for d in date_range]


def get_pids_to_process(df, trend_type, exclude_pids=None):
    """Get the PIDs matching a particular trend type"""
    pids = df[df["class_label"] == trend_type]["pid"].values.tolist()
    if exclude_pids is not None:
        pids = list(set(pids).difference(exclude_pids))
    return pids


def filter_egms_pid(df, pid):
    """Return a single row matching the PID"""
    return df[df["pid"] == pid]


def get_most_common_sr(date_cols):
    """Get the most common time interval between EGMS dates"""
    acquisition_days = Counter(
            [td.days for td in np.diff(
                [datetime.strptime(d, "%Y%m%d") for d in date_cols]
            )]
    )
    return list(acquisition_days.keys())[0]


def resample_egms_data(df, date_cols):
    """Create constant time series columns for EGMS data"""
    sr = get_most_common_sr(date_cols)
    resampled_dates = create_date_range(
        date_cols[0], date_cols[-1], sr
    )
    return df.reindex(columns=resampled_dates)


def filter_and_resample_egms(df, pid, date_cols):
    df = filter_egms_pid(df)
    df = resample_egms_data(df, date_cols)
    return df


# From https://pynative.com/python-serialize-numpy-ndarray-into-json/
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
