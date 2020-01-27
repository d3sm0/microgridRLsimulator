from itertools import chain

import numpy as np
import pandas as pd


class Forecaster:

    def __init__(self, simulator, control_horizon, deviation_factor=None):
        """

        :param simulator: Instance of Simulator
        :param control_horizon: The number of forecast steps (includes the current step)
        :param deviation_factor: the std factor used for the noisy forecast
        """
        self.date_range = simulator.database.time_to_idx
        self.start_date_index = simulator.env_step  # maybe +1
        self.database = simulator.database
        self.grid = simulator.grid
        self.control_horizon = control_horizon  # min(control_horizon, len(
        # self.date_range) - 1 - self.start_date_index)  # -1 because the end date is not part of the problem
        date_range = simulator.database.time_to_idx[self.start_date_index-1:self.control_horizon]
        #date_range = date
        #    pd.date_range(start=self.date_range[-1], periods=self.control_horizon, freq=self.date_range.freq)
        self.forecast_date_range = list(sorted(set(chain(self.date_range, date_range))))
        self.forecasted_PV_production = None
        self.forecasted_consumption = None
        self.deviation_factor = deviation_factor

    def _forecast(self, env_step, noise_fn):

        self.forecasted_PV_production = []
        self.forecasted_consumption = []

        for i in range(self.control_horizon):
            non_flexible_production = 0
            non_flexible_consumption = 0
            next_step = env_step + i
            for generator in self.grid.generators:
                if not generator.steerable:
                    time = next_step * self.grid.period_duration * 60
                    updated_capacity = generator.find_capacity(time)
                    # Assumption: the capacity update is not taken into account for optimization
                    scale = (updated_capacity / generator.initial_capacity)
                    next_production = self.database.get_columns(generator.name, self.forecast_date_range[next_step])
                    non_flexible_production += scale * next_production

            for load in self.grid.loads:
                next_load = self.database.get_columns(load.name, self.forecast_date_range[next_step])
                non_flexible_consumption += next_load

            non_flexible_production, non_flexible_consumption = noise_fn(non_flexible_production, non_flexible_consumption, i)
            self.forecasted_PV_production.append(non_flexible_production)
            self.forecasted_consumption.append(non_flexible_consumption)

    def exact_forecast(self, env_step):
        """
        Make an exact forecast of the future loads and PV production

        Return nothing, fill the forecast lists.
        """

        def noise_fn(x, y, *args):
            return x, y

        self._forecast(env_step, noise_fn=noise_fn)

    def noisy_forecast(self, env_step):
        """
        Make an noise increasing forecast of the future loads and PV production

        Return nothing, fill the forecast lists.
        """

        # This forecast has a variable noise with respect to the forecast step
        std_factor = np.linspace(.0, self.deviation_factor, num=self.control_horizon)

        def noise_fn(production, consumption, time_step):
            # increasing from 0 at current step to deviation at final step
            noise = np.random.normal(scale=std_factor[time_step] * production)
            production = min(0, production + noise)
            # std relative to the exact value => use of a deviation factor increasing from 0% to 20% at last step (can be modified)
            noise = np.random.normal(scale=std_factor[time_step] * consumption)
            consumption = min(0, consumption + noise)
            return production, consumption

        self._forecast(env_step, noise_fn)

    def get_forecast(self):
        return [self.forecasted_consumption, self.forecasted_PV_production]
