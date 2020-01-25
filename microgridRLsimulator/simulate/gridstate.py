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
        self.cumulative_cost = 0.
        self.delta_t = grid.delta_t
        n_storages = grid.n_storages
        n_generators = grid.n_generators  # len(grid.generators)
        self.date_time = date_time
        # self.delta_h = self.compute_delta_h()
        # List of state of charge for all storage devices, initialized at half of their capacity
        self.state_of_charge = [s.capacity / 2.0 for s in grid.storages]
        self.n_cycles = [0] * grid.n_storages
        self.capacities = [s.capacity for s in grid.storages]
        self.res_gen_capacities = [g.capacity for g in grid.generators if not g.steerable]
        self.cum_total_cost = 0.0  # EUR Cumulative total energy cost to date
        self.fuel_cost = 0.0  # EUR
        self.curtailment_cost = 0.0  # EUR
        self.load_not_served_cost = 0.0  # EUR
        self.total_cost = 0.0  # EUR
        self.curtailment_price = grid.curtailment_price
        self.load_shedding_price = grid.load_shedding_price

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

    def compute_capacity(self, next_soc, n_cycles, capacity):
        self.state_of_charge = next_soc
        self.n_cycles = n_cycles
        self.capacities = capacity

    def update_storage(self, actual_charge, actual_discharge):
        self.charge = actual_charge
        self.discharge = actual_discharge

    def update_production(self, generation, fuel_cost):
        # actual components dynamics
        self.generation = generation
        self.fuel_cost = fuel_cost
        # Deduce actual production and consumption based on the control actions taken and the
        self.production = self.non_steerable_production + sum(self.discharge) + sum(self.generation)
        self.consumption = self.non_steerable_consumption + sum(self.charge)

    def compute_cost(self):
        curtailment_cost = self.grid_export * self.curtailment_price
        load_not_served_cost = self.grid_import * self.load_shedding_price
        self.total_cost = load_not_served_cost + curtailment_cost + self.fuel_cost

        multiobj = {'total_cost': self.total_cost,
                    'load_shedding': load_not_served_cost,
                    'fuel_cost': self.fuel_cost,
                    'curtailment': curtailment_cost,
                    'storage_maintenance': sum(self.n_cycles)
                    }
        return multiobj

    def compute_delta_h(self):
        """
        :return the number of hours since the 1st January of the current year.
        """

        raise NotImplementedError
        first_jan = datetime(self.date_time.year, 1, 1, 00, 00, 00)
        delta_s = (self.date_time - first_jan).total_seconds()  # difference in seconds
        delta_h = divmod(delta_s, 3600)[0]  # difference in hours
        return delta_h

    def perform_balancing(self):
        # Perform the final balancing
        # NOTE: for now since the system is off grid we assume that:
        # a) Imports are equivalent to load shedding
        # b) exports are equivalent to production curtailment

        net_import = (self.consumption - self.production) * self.delta_t
        self.grid_import = net_import if net_import > 0. else 0
        self.grid_export = 0 if net_import > 0. else -net_import
