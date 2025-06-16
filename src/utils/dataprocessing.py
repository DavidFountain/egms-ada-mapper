import re
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter


def prettify_egms_date(egms_date: str) -> str:
    return egms_date[:4] + "-" + egms_date[4:]


def get_date_cols(egms_df) -> list:
    return [col for col in egms_df.columns
            if re.match(r"^\d{8}$", col)]


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


def convert_json_to_dataframe(json_dict) -> pd.DataFrame:
    """Convert JSON dict to GeoPandas GeoDataframe

    Parameters
    ----------
    json_dict : a JSON dict object with 'features' key

    Returns
    ----------
    GeoPandas GeoDataFrame
    """
    return pd.DataFrame.from_dict(json.loads(json_dict))


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


def create_velocity_groups(velocities):
    return np.where(
        velocities < -10, "<-10",
        np.where(
            velocities < -6, "<-6",
            np.where(
                velocities < -2, "<-2",
                np.where(
                    (velocities >= -2) & (velocities <= 2), "[-2, 2]",
                    np.where(
                        (velocities > 2) & (velocities <= 6), ">2",
                        np.where(
                            (velocities > 6) & (velocities <= 10), ">6", ">10"
                        )
                    )
                )
            )
        )
    )
