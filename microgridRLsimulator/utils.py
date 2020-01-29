from datetime import timedelta, datetime

import numpy as np


def datetime_range(start, end, delta):
    current = start
    if not isinstance(delta, timedelta):
        delta = timedelta(**delta)
    while current < end:
        yield current
        current += delta


# TOL_IS_ZERO = 2.5 * 2e-2
TOL_IS_ZERO = 1e-5  # A higher tolerance leads to large difference between the optimization problem and simulation


def negative(value, tol=TOL_IS_ZERO):
    """
    Check if a value is negative with respect to a tolerance.
    :param value: Value
    :param tol: Tolerance.
    :return: Boolean.
    """
    return value < -tol


def positive(value, tol=TOL_IS_ZERO):
    """
    Check if a value is positive with respect to a tolerance.
    :param value: Value
    :param tol: Tolerance.
    :return: Boolean.
    """
    return value > tol


def decode_gridstates(gridstates, features, n_sequences):
    """

    :param features: A features dict
    :param n_sequences: the number of state sequences (backcast)
    :return a list of the state values
    """
    values = list()
    state_alone_size = len(values)
    for gridstate in gridstates:
        values = _decode_gridstate(gridstate, values, features)
    n_missing_values = state_alone_size * (n_sequences - len(gridstates))
    values = n_missing_values * [.0] + values
    return values


def _decode_gridstate(gridstate, values, features=None):
    for attr in sorted(features):
        x = getattr(gridstate, attr)
        if isinstance(x, list):
            values += x
        else:
            values.append(x)
    return values


def time_string_for_storing_results(name, case):
    """

    :param case: the case name
    :return a string used for file or folder names
    """
    return name + f"_{case}_{datetime.now().strftime('%Y-%m-%d_%H%M')}"


class CastList(list):
    # TODO this is not elegant but works
    def append(self, x):
        if isinstance(x, list):
            x = list(map(float, x))
        if isinstance(x, np.float32) or isinstance(x, np.float64):
            x = float(x)
        super(CastList, self).append(x)


def gather_action_dimension(simulator):
    actions_upper_limits = list()
    # TODO use the period duration (action in kW)

    actions_upper_limits += [b.capacity * b.max_charge_rate / 100. for b in simulator.grid.storages]
    actions_upper_limits += [b.capacity * b.max_discharge_rate / 100. for b in simulator.grid.storages]
    actions_upper_limits += [g.capacity for g in simulator.grid.generators if g.steerable]
    actions_upper_limits = np.array(actions_upper_limits, np.float32)
    actions_lower_limits = np.zeros_like(actions_upper_limits)
    # For now, the action in continuous mode is a GridAction =/= from what is expected from gym framework
    # (array of actions value as the action space)
    #  TODO: make appropriate changes
    return actions_lower_limits, actions_upper_limits


def gather_space_dimension(simulator):
    # Observation space
    state_upper_limits = list()
    for attr, val in sorted(simulator.env_config["features"].items()):
        if val:
            if attr == "non_steerable_production" or attr == "res_gen_capacities":
                production = [sum(g.capacity for g in simulator.grid.generators if not g.steerable)]
                state_upper_limits += production
            elif attr == "non_steerable_consumption":
                consumption = [sum(l.capacity for l in simulator.grid.loads)]
                state_upper_limits += consumption
            elif attr == "n_cycles":
                # High number instead of Inf (if you want to be able to sample from it)
                state_upper_limits += [np.Inf for b in simulator.grid.storages]
            elif attr == "delta_h":
                state_upper_limits += [np.Inf]
            elif attr == "state_of_charge" or attr == "capacities":
                state_upper_limits += [b.capacity for b in simulator.grid.storages]

    forecast_upper_limits = []
    if simulator.env_config["forecast_steps"]:
        forecast_upper_limits = [sum(l.capacity for l in simulator.grid.loads),
                                 sum(g.capacity for g in simulator.grid.generators if not g.steerable)] * \
                                simulator.data["forecast_steps"]

    n_steps = simulator.env_config["backcast_steps"] + 1
    upper_limits = np.array(n_steps * state_upper_limits + forecast_upper_limits, dtype=np.float32)
    lower_limits = np.zeros_like(upper_limits)

    return lower_limits, upper_limits


import time


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, type, value, traceback):
        self.end = time.clock()
        print(f"{self.name}:{self.end - self.start}")


import os

pkg_dir = os.path.dirname(__file__)
MICROGRID_CONFIG_FILE = lambda case: os.path.join(pkg_dir, "data", case, f"{case}.json")
MICROGRID_DATA_FILE = lambda case: os.path.join(pkg_dir, "data", case, f"{case}_dataset.csv")
RESULT_PATH = lambda case: os.path.join(pkg_dir, "data", case, f"{case}.json")


def check_type(x):
    import numpy as np
    assert isinstance(x, np.float32)