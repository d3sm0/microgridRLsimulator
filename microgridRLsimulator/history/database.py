import numpy as np
import pandas as pd


class Database:
    _inputs_ = ['Year', 'Month', 'Day', 'Hour', 'Minutes', 'Seconds', 'IsoDayOfWeek', 'IsoWeekNumber']

    def __init__(self, path_to_csv, device_names):
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
        :param grid: A Grid object describing the configuration of the microgridRLsimulator
        """
        self._output_ = device_names
        self.read_data(path_to_csv)

    def read_data(self, path):
        """
        Read data and generate new columns based on the DateTime column.

        :param path: Path to the csv data file
        :return: A pandas dataframe
        """
        df = pd.read_csv(path, sep=";|,", parse_dates=True, index_col='DateTime', engine='python')

        assert df.index.is_monotonic, f"DateTime index is not monotonic for {path}"
        self.columns_name = df.columns.tolist()
        self.values = df.values.astype(np.float32)
        self.time_to_idx = df.index.tolist()
        self.first_valid_index = df.first_valid_index()
        self.last_valid_index = df.last_valid_index()

        # df['Year'] = df.index.map(lambda x: x.year)
        # df['Month'] = df.index.map(lambda x: x.month)
        # df['Day'] = df.index.map(lambda x: x.day)
        # df['Hour'] = df.index.map(lambda x: x.hour)
        # df['Minutes'] = df.index.map(lambda x: x.minute)
        # df['Seconds'] = df.index.map(lambda x: x.second)
        # df['IsoDayOfWeek'] = df.index.map(lambda x: x.isoweekday())
        # df['IsoWeekNumber'] = df.index.map(lambda x: x.isocalendar()[1])

        # Assert required columns are defined
        for tag in self._output_:
            if tag not in self.columns_name:
                raise ValueError(f"Column name {tag} not defined in {path}")

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
