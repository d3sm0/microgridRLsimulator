# -*- coding: utf-8 -*-
from datetime import datetime


class GridState:
    def __init__(self, grid, date_time):
        """
        Representation of the state of the system in the simulator. The state includes the state
        of charge of storage devices plus information regarding past operation of the system.

        :param grid: A Grid object
        :param date_time: The time at which the system in this state
        """
        n_storages = len(grid.storages)
        n_generators = len(grid.generators)
        self.date_time = date_time
        # self.delta_h = self.compute_delta_h()
        # List of state of charge for all storage devices, initialized at half of their capacity
        self.n_cycles = [0 for s in grid.storages]
        self.capacities = [s.capacity for s in grid.storages]
        self.state_of_charge = [s.capacity / 2.0 for s in grid.storages]
        self.res_gen_capacities = [g.capacity for g in grid.generators if not g.steerable]

        self.cum_total_cost = 0.0  # EUR Cumulative total energy cost to date
        self.cumulative_cost = 0.
        self.fuel_cost = 0.0  # EUR
        self.curtailment_cost = 0.0  # EUR
        self.load_not_served_cost = 0.0  # EUR
        self.total_cost = 0.0  # EUR

        # Auxiliary info
        self.grid_import = 0.0  # kWh
        self.grid_export = 0.0  # kWh
        self.production = 0.0  # kW
        self.consumption = 0.0  # kW
        self.charge = [0.0] * n_storages  # kW
        self.discharge = [0.0] * n_storages  # kW
        self.generation = [0.0] * n_generators  # kW
        self.non_steerable_production = 0.  # kW
        self.non_steerable_consumption = 0.  # kW
