# -*- coding: utf-8 -*-
import itertools

import numpy as np

from microgridRLsimulator.utils import check_type


class GridAction:
    def __init__(self, generation, charge, discharge):
        # not sure if this should be  adict
        # assert isinstance(generation, dict) and isinstance(charge, float) and isinstance(discharge,float )
        self.generation = generation
        self.charge = charge
        self.discharge = discharge


def binned_action_space(action_space, n_bins=10):
    charge, generator = action_space
    discharge = charge
    charge_bins = np.linspace(0, charge, n_bins)
    discharge_bins = np.logspace(0, discharge, n_bins)
    gen_bins = np.linspace(0, generator, n_bins)
    charge_battery = list(itertools.product(charge_bins, gen_bins))

    charge = [GridAction({'gen': 0}, c.item(), 0) for c in charge_bins]
    discharge = [GridAction({'gen': 0}, 0, d.item()) for d in discharge_bins]
    charge_battery = [GridAction({'gen': g.item()}, c.item(), 0) for c, g in charge_battery]

    action_space = charge + discharge + charge_battery
    action_space_idx_to_dict = {idx: a for idx, a in enumerate(action_space)}
    return action_space_idx_to_dict


def _construct_action_from_cluster(action, cluster, action_bound):
    _action = cluster[action]
    _action = np.random.normal(_action['mean'], _action['std'])
    _action = np.clip(_action, *action_bound)
    charge, discharge, gen = _action
    gen = {'gen': gen}
    return GridAction(gen, charge, discharge)


def _construct_action_from_list(action, n_storages, action_bound):
    assert isinstance(action, np.ndarray) and np.all(action >= 0), f"Wrong action format {action}"
    action = np.clip(action, *action_bound)
    charge = action[:1].item()
    discharge = action[n_storages:2 * n_storages].item()
    gen = action[2 * n_storages:].item()
    gen = dict(gen=gen)
    return GridAction(gen, charge, discharge)


def _infer_high_level_actions(n_storages):
    """
    Method that infers the full high-level action space by the number of controllable storage devices. Simultaneous
    charging of a battery and charging of another is ruled-out from the action set

    :return: list of possible tuples
    """

    # The available decisions for the storage device are charge (C), discharge (D) and idle (I)
    combinations_list = itertools.product('CDI', repeat=n_storages)

    # Generate the total actions set but exclude simultaneous charging of one battery and discharging of another
    combos_exclude_simul = list(filter(lambda x: not ('D' in x and 'C' in x), combinations_list))

    return combos_exclude_simul


def _construct_charge_action(action, state, grid):
    discharge = 0
    gen = 0
    charge = 0
    action = action.item()
    net = state.epv - state.demand
    if action < 0:  # discharging
        action = abs(action)
        max_discharge = min(state.soc * grid.storage.discharge_efficiency / grid.dt, grid.storage.max_discharge_rate)
        discharge = min(abs(net), action, max_discharge)
    else:
        gen = max(0, min(action, grid.engine.capacity))
        total_energy = gen + net
        charge = max(0, total_energy)
        if charge > 0:
            ds = grid.storage.capacity - state.soc
            charge = min(ds / (grid.dt * grid.storage.charge_efficiency), charge, grid.storage.max_charge_rate)

    assert (gen >= 0 and charge >= 0 and discharge >= 0), f"{charge}, {discharge}, {gen}"

    return GridAction({'gen': gen}, charge, discharge)


def _construct_action(high_level_action, state, grid):
    n_storages = len(state.state_of_charge)
    high_level_actions = _infer_high_level_actions(n_storages)
    generation = {g.name: 0. for g in grid.generators if g.steerable}
    charge = [0. for b in range(n_storages)]
    discharge = [0. for b in range(n_storages)]

    consumption = state.non_steerable_consumption
    state_of_charge = state.state_of_charge
    non_flex_production = state.non_steerable_production

    d = 1
    # Compute the residual generation :
    # a) if it is positive there is excess of energy
    # b) if it is negative there is deficit
    net_generation = non_flex_production - consumption
    if net_generation < 0:
        # check if genset has to be active, if it is the case: activation at the min stable capacity and update the net generation.
        total_possible_discharge = 0.0
        genset_total_capacity = 0.0
        genset_min_stable_total_capacity = 0.0
        storages_to_discharge = [i for i, x in enumerate(high_level_actions[high_level_action]) if x == "D"]
        for b in storages_to_discharge:
            storage = grid.storages[b]
            total_possible_discharge += min(state_of_charge[b] * storage.discharge_efficiency / d,
                                            storage.max_discharge_rate)

        for g in grid.generators:  # TODO sort generators
            if net_generation + total_possible_discharge + genset_total_capacity < 0:
                genset_min_stable_total_capacity += g.min_stable_generation
                genset_total_capacity += g.capacity
                generation[g.name] = g.min_stable_generation

        # net generation takes into account the min stable production
        net_generation += genset_min_stable_total_capacity

    # ++======= #@+++ ++# +++
    if net_generation > 0:
        storages_to_charge = [i for i, x in enumerate(high_level_actions[high_level_action]) if x == "C"]
        # if there is excess, charge the storage devices that are in Charge mode by the controller
        for b in storages_to_charge:
            storage = grid.storages[b]
            soc = state_of_charge[b]
            empty_space = storage.capacity - soc
            charge[b] = min(empty_space / (d * storage.charge_efficiency), net_generation,
                            storage.max_charge_rate)  # net_generation is already in kW
            net_generation -= charge[b]
    elif net_generation < 0:
        storages_to_discharge = [i for i, x in enumerate(high_level_actions[high_level_action]) if x == "D"]
        # if there is deficit, discharge the storage devices that are in Discharge mode by the controller
        for b in storages_to_discharge:
            storage = grid.storages[b]
            soc = state_of_charge[b]
            # discharge = soc/d is always changed to soc*effi/d in the simulation
            discharge[b] = min(soc * storage.discharge_efficiency / d, -net_generation, storage.max_discharge_rate)
            net_generation += discharge[b]

        for g in grid.generators:  # TODO sort generators
            additional_generation = min(-net_generation, g.capacity - generation[g.name])
            generation[g.name] += additional_generation  # Update the production of generator g
            net_generation += additional_generation  # Update the remaining power to handle

    charge = [max(0, c) for c in charge]
    discharge = [max(0, c) for c in discharge]
    generation = {k: max(0, v) for k, v in generation.items()}
    return GridAction(generation, charge, discharge)


def _construct_action_(action, state, grid):
    action_list = ["C", "D", "I"]
    action = action_list[action]
    consumption = state.demand
    soc = state.soc
    production = state.epv
    generation = dict(gen=0)
    net_generation = production - consumption
    check_type(production)
    check_type(consumption)
    charge = 0
    discharge = 0
    if net_generation < 0 and action == "D":
        max_discharge = min(soc * grid.storage.discharge_efficiency / grid.dt, grid.storage.max_discharge_rate)
        if net_generation + max_discharge < 0:
            # wired
            net_generation += grid.engine.min_stable_generation
            generation['gen'] = grid.engine.min_stable_generation

    if net_generation > 0:
        if action == "C":
            ds = grid.storage.capacity - soc
            charge = min(ds / (grid.dt * grid.storage.charge_efficiency), net_generation,
                         grid.storage.max_charge_rate)
            net_generation -= charge
            assert charge >= 0
        # else:
        #     discharge = min(soc * grid.storage.discharge_efficiency / grid.dt, production,
        #                     grid.storage.max_discharge_rate)

        #     net_generation += discharge

    elif net_generation < 0:
        if action == "D":
            discharge = min(soc * grid.storage.discharge_efficiency / grid.dt, -net_generation)
            discharge = min(discharge, grid.storage.max_discharge_rate)
            net_generation += discharge
            assert discharge >= 0

        assert (grid.engine.capacity - generation['gen']) >= 0
        add = min(-net_generation, grid.engine.capacity - generation['gen'])
        generation['gen'] += add
        net_generation += add
        assert generation['gen'] >= 0

    assert charge >= 0 and discharge >= 0 and generation['gen'] >= 0
    action = GridAction(generation, charge, discharge)
    return action
