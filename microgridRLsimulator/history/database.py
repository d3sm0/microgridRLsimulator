import numpy as np

from microgridRLsimulator.utils import MICROGRID_DATA_FILE

import pandas as pd

to_datetime = pd.to_datetime


def _slice_dataset(date_slice, df):
    data_start = df.first_valid_index()
    data_end = df.last_valid_index()
    start_date, end_date = date_slice
    start_date = max(data_start, start_date)
    check_date(start_date, end_date, data_start, data_end)
    df = df.loc[start_date:end_date]
    return df


class Database:
    def __init__(self, path_to_csv, date_slice, freq=None):
        self.time_to_idx = None
        self.device = None
        self.values = None

        self.start_date, self.end_date = date_slice
        self.freq = freq
        self._read_data(path_to_csv)

    def _read_data(self, path):
        df = pd.read_csv(path, sep=";|,", parse_dates=True, index_col='DateTime', engine='python')

        df = _slice_dataset((self.start_date, self.end_date), df)
        if self.freq is not None:
            df = df.resample(str(self.freq) + 'h').apply(np.mean)

        assert df.index.is_monotonic, f"DateTime index is not monotonic for {path}"
        df = check_na(df)
        self.values = df.values.astype(np.float32)
        self.time_to_idx = df.index.tolist()
        self.device = df.columns.tolist()
        self.max_steps = len(self.time_to_idx)

    def get(self, device, idx):
        assert isinstance(idx, int) or isinstance(idx, slice), "idx must be int or a slice"
        assert device in self.device, "device not in columns"
        assert idx < self.max_steps, "out of samples"
        device = self.device.index(device)
        return self.values[idx, device].astype(np.float32).item()


def load_db(start_date, end_date, case, freq=1):
    assert isinstance(start_date, str) and isinstance(end_date, str), "Dates should be str"
    assert isinstance(freq, int), "freq must be an int"

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    database = Database(MICROGRID_DATA_FILE(case), date_slice=(start_date, end_date), freq=freq)
    return database


def check_na(df):
    total_nas = pd.isna(df)
    if total_nas.sum().sum() != 0:
        print(f"Found nas {total_nas}. Dropping them. Check the dataset.")
        df.dropna(inplace=True)
    return df

def check_date(start_date, end_date, data_start_date, data_end_date):
    assert (start_date < end_date), "The end date is before the start date."
    assert (data_start_date <= start_date < data_end_date), "Invalid start date."
    assert (data_start_date < end_date <= data_end_date), "Invalid end date."
