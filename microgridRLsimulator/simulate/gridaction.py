# -*- coding: utf-8 -*-

import numpy as np

from microgridRLsimulator.utils import type_checker, check_type

ACTION_LIST = ["C", "D", "I"]


class GridAction:
    def __init__(self, generation, charge, discharge):
        self.generation = generation
        self.charge = charge
        self.discharge = discharge


def construct_action_from_list(action, n_storages, action_bound):
    assert isinstance(action, np.ndarray) and np.all(action >= 0), f"Wrong action format {action}"
    action = np.clip(action, *action_bound)
    charge = action[:1].item()
    discharge = action[n_storages:2 * n_storages].item()
    gen = action[2 * n_storages:].item()
    gen = dict(gen=gen)
    return GridAction(gen, charge, discharge)


def construct_continuous_action(action, state, grid):
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

    check_type([gen, charge, discharge])

    return GridAction(gen, charge, discharge)


def construct_discrete_action(action, state, grid):
    action = ACTION_LIST[action]
    consumption = state.demand
    soc = state.soc
    production = state.epv
    generation = 0.
    net_generation = production - consumption
    check_type((production, consumption, net_generation))
    charge = 0.
    discharge = 0.
    if net_generation < 0 and action == "D":
        max_discharge = min(soc * grid.storage.discharge_efficiency / grid.dt, grid.storage.max_discharge_rate)
        if net_generation + max_discharge < 0:
            # wired
            net_generation += grid.engine.min_stable_generation
            generation = grid.engine.min_stable_generation

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

        assert (grid.engine.capacity - generation) >= 0
        add = min(-net_generation, grid.engine.capacity - generation)
        generation += add
        net_generation += add
        assert generation >= 0

    assert charge >= 0 and discharge >= 0 and generation >= 0
    #check_type([generation, charge, discharge])
    action = GridAction(generation, charge, discharge)

    return action
