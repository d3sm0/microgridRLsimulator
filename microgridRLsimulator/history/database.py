import numpy as np

from microgridRLsimulator.utils import MICROGRID_DATA_FILE

import pandas as pd


class Database:
    def __init__(self, path_to_csv, date_slice, freq=None):
        self.time_to_idx = None
        self.device = None
        self.values = None

        self.start_date, self.end_date = date_slice
        self.freq = freq
        self._read_data(path_to_csv)
        self.max_steps = len(self.time_to_idx)

    def _read_data(self, path):
        df = pd.read_csv(path, sep=";|,", parse_dates=True, index_col='DateTime', engine='python')

        df = self._slice_dataset((self.start_date, self.end_date), df)
        if self.freq is not None:
            df = df.resample(str(self.freq) + 'h').apply(np.mean)

        assert df.index.is_monotonic, f"DateTime index is not monotonic for {path}"
        assert pd.isna(df).sum().sum() == 0, "Found Nan in dataset"
        self.values = df.values.astype(np.float32)
        self.time_to_idx = df.index.tolist()
        self.device = df.columns.tolist()

    def _slice_dataset(self, date_slice, df):
        data_start = df.first_valid_index()
        data_end = df.last_valid_index()
        start_date, end_date = date_slice
        check_date(start_date, end_date, data_start, data_end)
        df = df.loc[start_date:end_date]
        return df

    def get(self, device, idx):
        assert not isinstance(idx, pd.datetime), "idx must be int or a slice"
        assert device in self.device, "device not in columns"
        assert idx < self.max_steps
        # this could be avoided by using enum
        device = self.device.index(device)
        return self.values[idx, device].astype(np.float32)


def load_db(start_date, end_date, case, freq=1):
    assert isinstance(start_date, str) and isinstance(end_date, str), "Dates should be str"
    assert isinstance(freq, int), "freq must be an int"

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    database = Database(MICROGRID_DATA_FILE(case), date_slice=(start_date, end_date), freq=freq)
    return database


def check_date(start_date, end_date, data_start_date, data_end_date):
    assert (start_date < end_date), "The end date is before the start date."
    assert (data_start_date <= start_date < data_end_date), "Invalid start date."
    assert (data_start_date < end_date <= data_end_date), "Invalid end date."


