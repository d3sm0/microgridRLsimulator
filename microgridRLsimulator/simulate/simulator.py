# -*- coding: utf-8 -*-
import collections
import itertools
import json
import os
from datetime import datetime

import pandas as pd

from microgridRLsimulator.history import Database
from microgridRLsimulator.model.grid import Grid
from microgridRLsimulator.plot import Plotter
from microgridRLsimulator.simulate.forecaster import Forecaster
from microgridRLsimulator.simulate.gridaction import GridAction
from microgridRLsimulator.simulate.gridstate import GridState
from microgridRLsimulator.utils import positive, negative, decode_gridstates, CastList

"""
TODO:
- create transition dynmaics for a grid of 1 load, 1 generator
- extend it to multiple storage, grids, generators
- trash the rest
"""


class Simulator:
    this_dir, _ = os.path.split(__file__)
    package_dir = os.path.dirname(this_dir)

    def __init__(self, start_date, end_date, case, params=None):
        """
        :param start_date: datetime for the start of the simulation
        :param end_date: datetime for the end of the simulation
        :param case: case name (string)
        :param decision_horizon:
        """

        MICROGRID_CONFIG_FILE = os.path.join(self.package_dir, "data", case, f"{case}.json")
        MICROGRID_DATA_FILE = os.path.join(self.package_dir, "data", case, f"{case}_dataset.csv")

        self.RESULTS_FOLDER = f"results/results_{case}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
        self.RESULTS_FILE = f"{self.RESULTS_FOLDER}/{case}_out.json"

        with open(MICROGRID_CONFIG_FILE, 'rb') as jsonFile:
            self.data = json.load(jsonFile)

        if params is not None:
            self.data.update(params)

        generation = self.data['generators'][1]['min_stable_generation']
        import warnings
        if generation == 0.:
            warnings.warn("min stable generation is 0!")

        self.grid = Grid(self.data)
        self.objectives = self.data["objectives"]

        if type(start_date) is str:
            start_date = pd.to_datetime(start_date)
        if type(end_date) is str:
            end_date = pd.to_datetime(end_date)

        self.start_date = start_date
        self.end_date = end_date
        self.database = Database(MICROGRID_DATA_FILE, self.grid.get_non_flexible_device_names(),
                                 date_slice=(start_date, end_date),
                                 freq=self.grid.period_duration)

        self.case = case
        self.actions = {}
        # converting dates to datetime object

        # Period duration is in hours because it is used to perform calculations
        #self.date_range = pd.date_range(start=start_date, end=end_date,
        #                                freq=str(self.grid.period_duration * 60) + 'min')
        #self.database.time_to_idx == self.date_range
        self.high_level_actions = self._infer_high_level_actions()

        self.env_step = 0
        self.cumulative_cost = 0.
        self.grid_states = collections.deque(maxlen=2)
        self.state_features = {k: v for k, v in self.data["features"].items() if v is True}
        self.forecaster = Forecaster(simulator=self, control_horizon=self.data['forecast_steps'] + 1,
                                     deviation_factor=0.2)

    def reset(self):
        """
        Resets the state of the simulator and returns a state representation for the agent.

        :return: A state representation for the agent as a list
        """
        self.actions = {}
        self.env_step = 0
        self.cumulative_cost = 0.
        self.grid = Grid(self.data)  # refresh the grid (storage capacity, number of cycles, etc)
        # Initialize a gridstate
        reset_state = self._make_reset_state()
        self.grid_states.append(reset_state)
        return self._decode_state((reset_state,))

    def _make_reset_state(self):
        reset_state = GridState(self.grid, 0.)
        reset_state.cumulative_cost = self.cumulative_cost
        realized_non_flexible_production = 0.0
        for g in self.grid.generators:
            if not g.steerable:
                realized_non_flexible_production += self.database.get_columns(g.name, self.start_date)
        realized_non_flexible_consumption = 0.0
        for l in self.grid.loads:
            realized_non_flexible_consumption += self.database.get_columns(l.name, self.start_date)
        # Add in the state the information about renewable generation and demand
        reset_state.non_steerable_production = realized_non_flexible_production
        reset_state.non_steerable_consumption = realized_non_flexible_consumption
        return reset_state

    def step(self, actions):
        """
        Method that can be called by an agent to create a transition of the system.

        :param high_level_action: Action taken by an agent (translated later on into an implementable action)
        :return: a tuple (next_state, reward, termination_condition)
        """
        dt = self.database.time_to_idx[self.env_step]
        # Use the high level action provided and the current state to generate the low level actions for each component

        actions = self.gather_action(actions)

        # Record these actions in a json file
        self.actions[dt.strftime('%y/%m/%d_%H')] = actions.to_json()

        #  Update the step number and check the termination condition
        self.env_step += 1
        is_terminal = self.check_terminal()
        p_dt = self.database.time_to_idx[self.env_step]  # It's the next step

        # Construct an empty next state
        dt = (p_dt - self.start_date).seconds / (60 ** 2)
        next_grid_state = GridState(self.grid, dt)

        # Apply the control actions
        n_storages = len(self.grid.storages)

        actual_charge, actual_discharge = self.compute_next_state_storage(actions, n_storages, next_grid_state)

        actual_consumption, actual_generation_cost, actual_production = self.apply_control(actions, actual_charge,
                                                                                           actual_discharge, n_storages)

        actual_export, actual_import = self.perform_balancing(actual_consumption, actual_production)

        # Compute the final cost of operation as the sum of three terms:
        multiobj = self.compute_cost(actual_export, actual_generation_cost, actual_import, n_storages, next_grid_state)
        self.add_demand_epv(next_grid_state, p_dt)
        self.grid_states.append(next_grid_state)
        # Pass the information about the next state, cost of the previous control actions and termination condition
        # self.grid_states[-(self.data['backcast_steps'] + 1):]
        return self._decode_state((next_grid_state,)), -multiobj['total_cost'], is_terminal, {'costs': multiobj}

    def check_terminal(self):
        is_terminal = False
        if self.env_step == self.database.max_steps - 1:
            is_terminal = True
        return is_terminal

    def gather_action(self, actions):
        if not isinstance(actions, GridAction):
            if self.data['action_space'].lower() == "discrete":
                actions = self._construct_action(actions)
            else:
                actions = self._construct_action_from_list(actions)
        return actions

    def add_demand_epv(self, next_grid_state, p_dt):
        # Add in the next state the information about renewable generation and demand
        # Note: here production refers only to the non-steerable production
        realized_non_flexible_production = 0.0
        for g in self.grid.generators:
            if not g.steerable:
                time = self.env_step * self.grid.period_duration * 60  # time in min (in order to be able to update capacity all min)
                g.update_capacity(time)
                realized_non_flexible_production += self.database.get_columns(g.name, p_dt) * (
                        g.capacity / g.initial_capacity)
        next_grid_state.res_gen_capacities = [g.capacity for g in self.grid.generators if not g.steerable]
        realized_non_flexible_consumption = 0.0
        for l in self.grid.loads:
            realized_non_flexible_consumption += self.database.get_columns(l.name, p_dt)
        next_grid_state.non_steerable_production = realized_non_flexible_production
        next_grid_state.non_steerable_consumption = realized_non_flexible_consumption

    def compute_cost(self, actual_export, actual_generation_cost, actual_import, n_storages, next_grid_state):
        # a) fuel costs for the generation
        # b) curtailment cost for the excess of generation that had to be curtailed
        # c) load shedding cost for the excess of load that had to be shed in order to maintain balance in the grid
        # Note that we can unbundle the individual costs according to the objective optimized
        self.grid_states[-1].fuel_cost = sum(actual_generation_cost[g] for g in self.grid.generators)
        self.grid_states[-1].curtailment_cost = actual_export * self.grid.curtailment_price
        self.grid_states[-1].load_not_served_cost = actual_import * self.grid.load_shedding_price
        self.grid_states[-1].total_cost = self.grid_states[-1].load_not_served_cost + self.grid_states[
            -1].curtailment_cost + self.grid_states[-1].fuel_cost
        multiobj = {'total_cost': self.grid_states[-1].total_cost,
                    'load_shedding': actual_import,
                    'fuel_cost': self.grid_states[-1].fuel_cost,
                    'curtailment': actual_export,
                    'storage_maintenance': sum([self.grid.storages[b].n_cycles for b in range(n_storages)])
                    }
        self.cumulative_cost += self.grid_states[-1].total_cost
        next_grid_state.cum_total_cost = self.cumulative_cost
        return multiobj

    def perform_balancing(self, actual_consumption, actual_production):
        # Perform the final balancing
        actual_import = actual_export = 0
        net_import = actual_consumption - actual_production
        if positive(net_import):
            actual_import = net_import * self.grid.period_duration
        elif negative(net_import):
            actual_export = -net_import * self.grid.period_duration
        # NOTE: for now since the system is off grid we assume that:
        # a) Imports are equivalent to load shedding
        # b) exports are equivalent to production curtailment
        self.grid_states[-1].grid_import = actual_import
        self.grid_states[-1].grid_export = actual_export
        return actual_export, actual_import

    def apply_control(self, actions, actual_charge, actual_discharge, n_storages):
        # Apply the control actions for the steerable generators based on the generators dynamics
        actual_generation = {g: 0. for g in self.grid.generators}
        actual_generation_cost = {g: 0. for g in self.grid.generators}
        for g in self.grid.generators:
            if g.steerable:
                actual_generation[g], actual_generation_cost[g] = g.simulate_generator(
                    actions.conventional_generation[g.name], self.grid.period_duration)
        # Record the generation output to the current state
        self.grid_states[-1].generation = list(actual_generation.values())
        # Deduce actual production and consumption based on the control actions taken and the
        # actual components dynamics
        actual_production = self.grid_states[-1].non_steerable_production \
                            + sum(actual_discharge[b] for b in range(n_storages)) \
                            + sum(actual_generation[g] for g in self.grid.generators)
        actual_consumption = self.grid_states[-1].non_steerable_consumption \
                             + sum(actual_charge[b] for b in range(n_storages))
        # Store the total production and consumption
        self.grid_states[-1].production = actual_production
        self.grid_states[-1].consumption = actual_consumption
        return actual_consumption, actual_generation_cost, actual_production

    def compute_next_state_storage(self, actions, n_storages, next_grid_state):
        # Compute next state of storage devices based on the control actions
        # and the storage dynamics
        next_soc = [0.0] * n_storages
        actual_charge = [0.0] * n_storages
        actual_discharge = [0.0] * n_storages
        for b in range(n_storages):
            (next_soc[b], actual_charge[b], actual_discharge[b]) = self.grid.storages[b].simulate(
                self.grid_states[-1].state_of_charge[b], actions.charge[b], actions.discharge[b],
                self.grid.period_duration
            )
            # Store the computed capacity and level of storage to the next state
            next_grid_state.capacities[b] = self.grid.storages[b].capacity
            next_grid_state.n_cycles[b] = self.grid.storages[b].n_cycles
            next_grid_state.state_of_charge[b] = next_soc[b]
        # Record the control actions for the storage devices to the current state
        self.grid_states[-1].charge = actual_charge[:]
        self.grid_states[-1].discharge = actual_discharge[:]
        return actual_charge, actual_discharge

    def store_and_plot(self, folder=None, learning_results=None, agent_options=None):
        """
        Store and plot results.

        :param folder: The simulation folder name.
        :param learning_results: A list containing the results of the learning progress
        :return: Nothing.
        """

        if learning_results is None:
            learning_results = [0]
        results = collections.defaultdict(lambda: CastList())

        for d in self.grid_states:
            results['dates'].append("%s" % d.date_time)
            results['soc'].append(d.state_of_charge)
            results['capacity'].append(d.capacities)
            results['res_gen_capacity'].append(d.res_gen_capacities)
            results['charge'].append(d.charge)
            results['discharge'].append(d.discharge)
            results['generation'].append(d.generation)
            results['fuel_cost'].append(d.fuel_cost)
            results['curtailment_cost'].append(d.curtailment_cost)
            results['load_not_served_cost'].append(d.load_not_served_cost)
            results['energy_cost'].append(d.total_cost)
            results['production'].append(d.production)
            results['consumption'].append(d.consumption)
            results['non_steerable_production'].append(d.non_steerable_production)
            results['non_steerable_consumption'].append(d.non_steerable_consumption)
            results['grid_import'].append(d.grid_import)
            results['grid_export'].append(d.grid_export)
            results['cum_total_cost'].append(d.cum_total_cost)
        results['avg_rewards'].append(learning_results)

        if folder is not None:
            self.RESULTS_FOLDER = folder
            self.RESULTS_FILE = "%s/%s_out.json" % (self.RESULTS_FOLDER, self.case)

        if not os.path.isdir(self.RESULTS_FOLDER):
            os.makedirs(self.RESULTS_FOLDER)

        with open(self.RESULTS_FILE, 'w') as jsonFile:
            json.dump(results, jsonFile)

        with open(self.RESULTS_FOLDER + "/mg_config.json", 'w') as jsonFile:
            json.dump(self.data, jsonFile)

        if agent_options is not None:
            with open(self.RESULTS_FOLDER + "/agent_options.json", 'w') as jsonFile:
                json.dump(agent_options, jsonFile)

        plotter = Plotter(results, '%s/%s' % (self.RESULTS_FOLDER, self.case))
        plotter.plot_results()

    def _infer_high_level_actions(self):
        """
        Method that infers the full high-level action space by the number of controllable storage devices. Simultaneous
        charging of a battery and charging of another is ruled-out from the action set

        :return: list of possible tuples
        """

        # The available decisions for the storage device are charge (C), discharge (D) and idle (I)
        combinations_list = itertools.product('CDI', repeat=len(self.grid.storages))

        # Generate the total actions set but exclude simultaneous charging of one battery and discharging of another
        combos_exclude_simul = list(filter(lambda x: not ('D' in x and 'C' in x), combinations_list))

        return combos_exclude_simul

    def _decode_state(self, gridstates):
        """
        Method that transforms the grid state into a list that contains the important information for the decision
        making process.

        :param gridstates: a list of Gridstate object that contains the whole information about the micro-grid
        :return: list with default or selected  state features.
        """

        state_list = decode_gridstates(gridstates, self.state_features,
                                       self.data['backcast_steps'] + 1)  # +1 because of the current state
        if self.data['forecast_steps'] > 0:
            if self.data['forecast_type'] == "exact":
                self.forecaster.exact_forecast(env_step=self.env_step)
            elif self.data['forecast_type'] == "noisy":
                self.forecaster.noisy_forecast(env_step=self.env_step)
            state_list += self.forecaster.forecasted_consumption[1:]
            state_list += self.forecaster.forecasted_PV_production[1:]
        return state_list

    def _construct_action(self, high_level_action):
        """
        Maps the  high level action provided by the agent into an action implementable by the simulator.
        high_level_action : 0 --> charging bat 1 - charging bat 2 ...
        high_level_action : 1 --> charging bat 1 - idling bat 2 ...
        high_level_action : 2 --> idle
        ...
        """

        n_storages = len(self.grid_states[-1].state_of_charge)

        generation = {g.name: 0. for g in self.grid.generators if g.steerable}
        charge = [0. for b in range(n_storages)]
        discharge = [0. for b in range(n_storages)]
        consumption = self.grid_states[-1].non_steerable_consumption
        state_of_charge = self.grid_states[-1].state_of_charge
        non_flex_production = self.grid_states[-1].non_steerable_production

        d = self.grid.period_duration
        # Compute the residual generation :
        # a) if it is positive there is excess of energy
        # b) if it is negative there is deficit
        net_generation = non_flex_production - consumption
        if negative(net_generation):
            # check if genset has to be active, if it is the case: activation at the min stable capacity and update the net generation.
            total_possible_discharge = 0.0
            genset_total_capacity = 0.0
            genset_min_stable_total_capacity = 0.0
            storages_to_discharge = [i for i, x in enumerate(self.high_level_actions[high_level_action]) if x == "D"]
            for b in storages_to_discharge:
                storage = self.grid.storages[b]
                total_possible_discharge += min(state_of_charge[b] * storage.discharge_efficiency / d,
                                                storage.max_discharge_rate)

            for g in self.grid.generators:  # TODO sort generators
                if g.steerable:
                    if net_generation + total_possible_discharge + genset_total_capacity < 0:
                        genset_min_stable_total_capacity += g.min_stable_generation * g.capacity
                        genset_total_capacity += g.capacity
                        generation[g.name] = g.min_stable_generation * g.capacity

            net_generation += genset_min_stable_total_capacity  # net generation takes into account the min stable production
        if positive(net_generation):
            storages_to_charge = [i for i, x in enumerate(self.high_level_actions[high_level_action]) if x == "C"]
            # if there is excess, charge the storage devices that are in Charge mode by the controller
            for b in storages_to_charge:
                storage = self.grid.storages[b]
                soc = state_of_charge[b]
                empty_space = storage.capacity - soc
                charge[b] = min(empty_space / (d * storage.charge_efficiency), net_generation,
                                storage.max_charge_rate)  # net_generation is already in kW
                net_generation -= charge[b]
        elif negative(net_generation):
            storages_to_discharge = [i for i, x in enumerate(self.high_level_actions[high_level_action]) if x == "D"]
            # if there is deficit, discharge the storage devices that are in Discharge mode by the controller
            for b in storages_to_discharge:
                storage = self.grid.storages[b]
                soc = state_of_charge[b]
                discharge[b] = min(soc * storage.discharge_efficiency / d, -net_generation,
                                   storage.max_discharge_rate)  # discharge = soc/d is always changed to soc*effi/d in the simulation
                net_generation += discharge[b]

            for g in self.grid.generators:  # TODO sort generators
                if g.steerable:
                    additional_generation = min(-net_generation, g.capacity - generation[g.name])
                    generation[g.name] += additional_generation  # Update the production of generator g
                    net_generation += additional_generation  # Update the remaining power to handle
        return GridAction(generation, charge, discharge)

    def _construct_action_from_list(self, actions_list):
        """

        :param action_list: a continuous action list ([charge1, charge2, ..., discharge1, discharge2, ..., gen1, gen2,...])
        :return: A GridAction object.

        """
        n_storages = len(self.grid.storages)
        generation = {g.name: 0. for g in self.grid.generators if g.steerable}
        charge = actions_list[:n_storages]
        discharge = actions_list[n_storages:2 * n_storages]
        gen = actions_list[2 * n_storages:]

        i = 0
        for g in self.grid.generators:
            if g.steerable:
                generation[g.name] = gen[i]
                i += 1

        return GridAction(generation, charge, discharge)
