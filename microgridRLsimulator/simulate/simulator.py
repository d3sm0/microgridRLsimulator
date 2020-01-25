# -*- coding: utf-8 -*-
import collections
import itertools
import json
import os
import warnings
from datetime import datetime

import pandas as pd

from microgridRLsimulator.history import Database
from microgridRLsimulator.model.grid import Grid
from microgridRLsimulator.plot import Plotter
from microgridRLsimulator.simulate.gridaction import GridAction
from microgridRLsimulator.simulate.gridstate import GridState
from microgridRLsimulator.utils import positive, negative, decode_gridstates, CastList


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
            self.grid_params = json.load(jsonFile)

        if params is not None:
            self.grid_params.update(params)

        generation = self.grid_params['generators'][1]['min_stable_generation']

        if generation == 0.:
            warnings.warn("min stable generation is 0!")

        self.grid = Grid(self.grid_params)
        # self.objectives = self.data["objectives"]

        self.case = case
        self.database = Database(MICROGRID_DATA_FILE, self.grid.get_non_flexible_device_names())
        # self.actions = {}
        # converting dates to datetime object
        if type(start_date) is str:
            start_date = pd.to_datetime(start_date)
        self.start_date = start_date
        if type(end_date) is str:
            end_date = pd.to_datetime(end_date)
        self.end_date = end_date

        data_start_date = self.database.first_valid_index
        data_end_date = self.database.last_valid_index
        self.check_date(data_end_date, data_start_date)
        # Period duration is in hours because it is used to perform calculations

        self.date_range = pd.date_range(start=start_date, end=end_date,
                                        freq=str(self.grid.delta_t * 60) + 'min')
        self.high_level_actions = _infer_high_level_actions(self.grid.n_storages)

        self.env_step = 0
        self.cumulative_cost = 0.
        self.grid_states = collections.deque(maxlen=2)
        self.state_features = {k: v for k, v in self.grid_params["features"].items() if v is True}

        # self.forecaster = Forecaster(simulator=self, control_horizon=self.data['forecast_steps'] + 1,
        # deviation_factor=0.2)

    def check_date(self, data_end_date, data_start_date):
        assert (self.start_date < self.end_date), "The end date is before the start date."
        assert (data_start_date <= self.start_date < data_end_date), "Invalid start date."
        assert (data_start_date < self.end_date <= data_end_date), "Invalid end date."

    def reset(self):
        """
        Resets the state of the simulator and returns a state representation for the agent.

        :return: A state representation for the agent as a list
        """
        # self.actions = collections.
        self.env_step = 0
        self.cumulative_cost = 0.
        # self.grid = Grid(self.grid_params)  # refresh the grid (storage capacity, number of cycles, etc)
        # Initialize a gridstate
        reset_state = self._make_reset_state()
        self.grid_states.append(reset_state)
        return self._decode_state((reset_state,))

    def _make_reset_state(self):
        reset_state = GridState(self.grid, self.start_date)
        reset_state.non_steerable_production = self.grid.get_production(self.database, self.start_date)
        reset_state.non_steerable_consumption = self.grid.get_load(self.database, self.start_date)
        return reset_state

    def step(self, actions):
        """
        Method that can be called by an agent to create a transition of the system.

        :param high_level_action: Action taken by an agent (translated later on into an implementable action)
        :return: a tuple (next_state, reward, termination_condition)
        """
        # dt = self.date_range[self.env_step]
        # Use the high level action provided and the current state to generate the low level actions for each component

        actions = self.gather_actions(actions)

        # Record these actions in a json file
        # self.actions[dt.strftime('%y/%m/%d_%H')] = actions.to_json()

        #  Update the step number and check the termination condition
        is_terminal, p_dt = self.check_terminal_state()

        # Construct an empty next state
        next_grid_state = GridState(self.grid, p_dt)

        # Apply the control actions

        self.compute_next_state(actions, next_grid_state)

        self.update_production(actions)

        self.grid_states[-1].perform_balancing()

        multiobj = self.compute_cost()

        self.cumulative_cost += self.grid_states[-1].total_cost
        next_grid_state.cum_total_cost = self.cumulative_cost

        self.update_non_steerable_production(next_grid_state, p_dt)
        self.grid_states.append(next_grid_state)

        # Pass the information about the next state, cost of the previous control actions and termination condition
        # self.grid_states[-(self.data['backcast_steps'] + 1):]
        return self._decode_state((next_grid_state,)), -multiobj['total_cost'], is_terminal, {'costs': multiobj}

    def check_terminal_state(self):
        self.env_step += 1
        is_terminal = False
        if self.env_step == len(self.date_range) - 1:
            is_terminal = True
        p_dt = self.date_range[self.env_step]  # It's the next step
        return is_terminal, p_dt

    def gather_actions(self, actions):
        if not isinstance(actions, GridAction):
            if self.grid_params['action_space'].lower() == "discrete":
                actions = self._construct_action(actions)
            else:
                actions = self._construct_action_from_list(actions)
        return actions

    def update_non_steerable_production(self, next_grid_state, p_dt):
        # Add in the next state the information about renewable generation and demand
        # Note: here production refers only to the non-steerable production

        next_grid_state.res_gen_capacities = self.grid.update_capacity(self.env_step * self.grid.delta_t)
        next_grid_state.non_steerable_production = self.grid.get_production(self.database, p_dt)
        next_grid_state.non_steerable_consumption = self.grid.get_load(self.database, p_dt)

    def compute_cost(self):
        # Compute the final cost of operation as the sum of three terms:
        # a) fuel costs for the generation
        # b) curtailment cost for the excess of generation that had to be curtailed
        # c) load shedding cost for the excess of load that had to be shed in order to maintain balance in the grid
        # Note that we can unbundle the individual costs according to the objective optimized

        multiobj = self.grid_states[-1].compute_cost()

        return multiobj

    def update_production(self, actions):
        # Apply the control actions for the steerable generators based on the generators dynamics
        actual_generation = {g: 0. for g in self.grid.generators}
        actual_generation_cost = {g: 0. for g in self.grid.generators}
        for g in self.grid.generators:
            # Record the generation output to the current state
            if g.steerable is True:
                actual_generation[g], actual_generation_cost[g] = g.simulate_generator(
                    actions.conventional_generation[g.name],
                    self.grid.delta_t)

        self.grid_states[-1].update_production(list(actual_generation.values()), sum(actual_generation_cost.values()))

    def compute_next_state(self, action, next_grid_state):
        # Compute next state of storage devices based on the control actions
        # and the storage dynamics

        params = (self.grid_states[-1].state_of_charge, action.charge, action.discharge)

        next_soc, actual_charge, actual_discharge = self.grid.storages.simulate(params, delta_t=self.grid.delta_t)

        # Store the computed capacity and level of storage to the next state
        next_grid_state.compute_capacity(next_soc, self.grid.storages.n_cycles(), self.grid.storages.capacity())

        # Record the control actions for the storage devices to the current state
        self.grid_states[-1].update_storage(actual_charge, actual_discharge)

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
            json.dump(self.grid_params, jsonFile)

        if agent_options is not None:
            with open(self.RESULTS_FOLDER + "/agent_options.json", 'w') as jsonFile:
                json.dump(agent_options, jsonFile)

        plotter = Plotter(results, '%s/%s' % (self.RESULTS_FOLDER, self.case))
        plotter.plot_results()

    def _decode_state(self, gridstates):
        """
        Method that transforms the grid state into a list that contains the important information for the decision
        making process.

        :param gridstates: a list of Gridstate object that contains the whole information about the micro-grid
        :return: list with default or selected  state features.
        """

        state_list = decode_gridstates(gridstates, self.state_features)  # +1 because of the current state
        # if self.data['forecast_steps'] > 0:
        #    if self.data['forecast_type'] == "exact":
        #        self.forecaster.exact_forecast(env_step=self.env_step)
        #    elif self.data['forecast_type'] == "noisy":
        #        self.forecaster.noisy_forecast(env_step=self.env_step)
        #    state_list += self.forecaster.forecasted_consumption[1:]
        #    state_list += self.forecaster.forecasted_PV_production[1:]
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

        d = self.grid.delta_t
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

        return GridAction(charge, discharge, generation)

    def _construct_action_from_list(self, actions_list):
        """

        :param action_list: a continuous action list ([charge1, charge2, ..., discharge1, discharge2, ..., gen1, gen2,...])
        :return: A GridAction object.

        """

        charge = actions_list[:self.grid.n_storages]
        discharge = actions_list[self.grid.n_storages:2 * self.grid.n_storages]
        gen = actions_list[2 * self.grid.n_storages:]

        gen = {g.name: v for g, v in zip(self.grid.generators, gen)}

        return GridAction(charge, discharge, gen)


def _infer_high_level_actions(n_storages):
    """
    Method that infers the full high-level action space by the number of controllable storage devices. Simultaneous
    charging of a battery and charging of another is ruled-out from the action set

    :return: list of possible tuples
    """

    # The available decisions for the storage device are charge (C), discharge (D) and idle (I)
    combinations_list = itertools.product('CDI', repeat=n_storages)

    # Generate the total actions set but exclude simultaneous charging of one battery and discharging of another
    def function(x):
        return not ('D' in x and 'C' in x)

    combos_exclude_simul = list(filter(function, combinations_list))

    return combos_exclude_simul
