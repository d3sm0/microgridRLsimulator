import numpy as np
import pandas as pd


class Database:
    _inputs_ = ['Year', 'Month', 'Day', 'Hour', 'Minutes', 'Seconds', 'IsoDayOfWeek', 'IsoWeekNumber']

    def __init__(self, path_to_csv, device_names, freq=None, date_slice=None):
        """
        A Database objects holds the realized data of the microgridRLsimulator in a pandas dataframe.

        The CSV file values are separated by ';' and the first line must contain series names.
        It must contain

        * a 'DateTime' column with values interpretable as python date time objects.
        * a 'Price' column with values interpretable as floats.
        * All the non-flexible quantities (load and generation) described in the microgridRLsimulator configuration

        Some new columns are generated from the DateTime column to indicate e.g. whether
        a datetime corresponds to a day of the week or not.

        :param path_to_csv: Path to csv containing realized data
        :param device_names: A Grid object describing the configuration of the microgridRLsimulator
        """
        self._output_ = device_names  # + ['Price'] # Add the price when working on-grid
        self.read_data(path_to_csv, date_slice, freq=freq)
        self.max_steps = len(self.time_to_idx)

    def read_data(self, path, date_slice=None, freq=None):
        """
        Read data and generate new columns based on the DateTime column.

        :param path: Path to the csv data file
        :return: A pandas dataframe
        """
        df = pd.read_csv(path, sep=";|,", parse_dates=True, index_col='DateTime', engine='python')

        if date_slice is not None:
            df = self._slice_dataset(date_slice, df)
        if freq is not None:
            df = df.resample(str(int(freq)) + 'h').apply(np.mean)

        assert df.index.is_monotonic, f"DateTime index is not monotonic for {path}"
        self.columns_name = df.columns.tolist()
        self.values = df.values.astype(np.float32)
        self.time_to_idx = df.index.tolist()

        # Assert required columns are defined
        for tag in self._output_:
            if tag not in self.columns_name:
                raise ValueError(f"Column name {tag} not defined in {path}")

        return df

    def _slice_dataset(self, date_slice, df):
        data_start = df.first_valid_index()
        data_end = df.last_valid_index()
        start_date, end_date = date_slice
        check_date(start_date, end_date, data_start, data_end)
        df = df.loc[start_date:end_date]
        return df

    def get_columns(self, column_indexer, time_indexer):
        """

        :param column_indexer: The name of a column
        :param time_indexer: A datetime
        :return: The realized value of the series column_indexer at time time_indexer
        """

        idx = self.time_to_idx.index(time_indexer)
        column = self.columns_name.index(column_indexer)

        return self.values[idx, column]

    def get_column(self, column_indexer, dt_from, dt_to):
        """

        :param column_indexer: The name of a column
        :param dt_from: A start datetime
        :param dt_to: An end datetime
        :return: A list of values of the column_indexer series between dt_from and dt_to
        """

        start_dt = self.time_to_idx.index(dt_from)
        end_dt = self.time_to_idx.index(dt_to)
        column = self.columns_name.index(column_indexer)

        return self.values[start_dt:end_dt + 1, column]

    def get_times(self, time_indexer):
        """

        :param time_indexer: A date time
        :return: A list containing the value of all the series at time time_indexer
        """
        raise NotImplementedError()


def check_date(start_date, end_date, data_start_date, data_end_date):
    assert (start_date < end_date), "The end date is before the start date."
    assert (data_start_date <= start_date < data_end_date), "Invalid start date."
    assert (data_start_date < end_date <= data_end_date), "Invalid end date."
