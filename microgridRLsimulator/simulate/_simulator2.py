import json
import os
import pickle as pkl

import numpy as np

from microgridRLsimulator.history.database import load_db
from microgridRLsimulator.model._generator import Diesel, EPV
from microgridRLsimulator.model._storage import DCAStorage
from microgridRLsimulator.plot import render
from microgridRLsimulator.simulate.gridaction import GridAction
from microgridRLsimulator.simulate.gridaction import _construct_action_, _construct_action_from_list, \
    binned_action_space, _construct_charge_action
from microgridRLsimulator.utils import MICROGRID_CONFIG_FILE


def load_config(fname):
    with open(fname, 'rb') as jsonFile:
        data = json.load(jsonFile)
    return data


def check_generation(generation):
    if generation == 0.:
        import warnings;
        warnings.warn("Min generation is 0.")


def _decode_state(state):
    return np.array([state.demand, state.epv, state.soc], dtype=np.float32)


def _peform_balancing(production, consumption):
    net = consumption - production
    _import = max(0., net)
    _export = max(0., -net)
    assert net < 0 and _export < 0 or _import >= 0, f"net:{net} or export:{_export} are >0"
    return _import, _export


def check_json(value):
    try:
        json.dumps(value)
        return True
    except TypeError:
        return False


import collections


def _list_to_dict(info):
    infos = collections.defaultdict(lambda: [])
    for _info in info:
        for k, value in _info.items():
            value = value if isinstance(value, str) else float(value)
            assert check_json(value)
            infos[k].append(value)
    return dict(infos)


_cost_keys = [
    "cumulative_cost",
    "total_cost",
    "load_shedding",
    "fuel_cost",
    "curtailment",
    "storage_maintenance"
]
_state_keys = ["soc",
               "demand",
               "epv",
               "storage_capacity",
               "epv_capacity"]
_transition_keys = [

    "production",
    "consumption",
    "charge",
    "discharge",
    "import",
    "export"
]
_keys = _cost_keys + _state_keys + _transition_keys


class Grid:
    def __init__(self, start_date, end_date, params, case):
        self.n_storages = 1
        self.dt = params['period_duration'] // 60
        self.db = load_db(start_date, end_date, case=case, freq=self.dt)
        self.storage = DCAStorage(params['storages'][0])
        self.epv = EPV(params['generators'][0])
        self.engine = Diesel(params['generators'][1])

        #self._action_space_dict = binned_action_space((self.storage.max_discharge(), self.engine.capacity))

    def get_production(self, time):
        _time = time * self.dt * 60
        self.epv.update_capacity(_time)
        epv = self.db.get('EPV', time)
        return np.float32(epv * self.epv.usage())

    def get_consumption(self, time):
        return self.db.get('C1', time)

    def get_grid_status(self):
        return {"epv": {"capacity": self.epv.capacity},
                "storage": {"capacity": self.storage.capacity,
                            "n_cycles": self.storage.n_cycles}
                }

    def set_grid_state(self, state):
        for k, v in state.items():
            attr = getattr(self, k)
            for k, v in v.items():
                setattr(attr, k, v)

    def reset(self):
        self.epv.reset()
        self.engine.reset()
        self.storage.reset()

    def gather_action_space(self):

        high = np.array([self.storage.max_charge()], np.float32)
        low = np.array([-self.storage.max_discharge()], np.float32)

        # high = np.array([self.storage.max_charge(), self.engine.capacity], np.float32)
        # low = np.array([self.storage.max_discharge(), 0.], np.float32)
        return low, high

    def gather_observation_space(self):
        high = np.array([20., self.epv.capacity, self.storage.capacity], dtype=np.float32)
        low = np.zeros_like(high)
        return low, high


class GridState:
    def __init__(self, soc=0, epv=0, demand=0, time_step=0, grid_status=None):
        self.soc = soc
        self.epv = epv
        self.demand = demand
        self.time_step = time_step
        self.grid_status = grid_status

    def as_numpy(self):
        return _decode_state(self)


class Simulator:
    def __init__(self, start_date, end_date, case='elespino', params=None):
        env_config = load_config(MICROGRID_CONFIG_FILE(case))

        if params is not None:
            env_config['generators'][1]['min_stable_generation'] = params['min_stable_generation']
            env_config['storages'][0]['prob_failure'] = params['prob_failure']
            env_config.update(params)

        self.env_config = env_config
        check_generation(env_config['generators'][1]['min_stable_generation'])

        self.grid = Grid(start_date, end_date, env_config, case=case)

        path = os.path.dirname(__file__)
        path = os.path.join(path, 'clusters.pkl')
        with open(path, 'rb') as p:
            data = pkl.load(p)

        self._action_cluster = data
        # self._action_list = sorted(list(self._action_cluster.keys()))
        #self._action_list = list(self.grid._action_space_dict.keys())  # list(range(3))
        self._action_list = list(range(3))

        self.env_step = 0
        self.cost = 0
        self.grid_state = None
        self.infos = []

        print(f"Init simulator {case}:{str(self.grid.db.start_date), str(self.grid.db.end_date)})")
        print(f"\tMax steps:{self.grid.db.max_steps}")

    def sample(self):

        step = np.random.randint(0, self.grid.db.max_steps - 1)
        epv = self.grid.get_production(step)
        demand = self.grid.get_consumption(step)
        soc = np.random.uniform(0., self.grid.storage.capacity)
        state = GridState(epv=epv,
                          demand=demand,
                          soc=soc,
                          grid_status=self.grid.get_grid_status(),
                          time_step=step)
        self.set_state(state)
        return state.as_numpy()

    def reset(self):
        self.grid.reset()

        self.infos = []
        self.env_step = 0
        self.cost = 0
        self.start = 0

        epv, demand = self._update_demand_epv()
        state = GridState(
            epv=epv,
            demand=demand,
            grid_status=self.grid.get_grid_status(),
            soc=self.grid.storage.capacity / 2.
        )
        self.grid_state = state
        return state.as_numpy()

    def _update_demand_epv(self):
        epv = self.grid.get_production(self.env_step)
        demand = self.grid.get_consumption(self.env_step)
        assert isinstance(epv, np.float32) and isinstance(demand, np.float32)
        return epv, demand

    def step(self, action):
        next_state, reward, done, info = self._step(action)
        return next_state, reward, done, info

    def set_state(self, grid_state):
        self.grid_state = grid_state
        self.env_step = self.grid_state.time_step
        self.grid.set_grid_state(grid_state.grid_status)

    def _step(self, action):
        self.env_step += 1
        action = self._gather_action(action)

        soc_tp1, charge, discharge = self._compute_next_storage(action)
        production, consumption, generation_cost, generation = self._apply_control(action, charge, discharge)
        _import, _export = _peform_balancing(production, consumption)
        multi_obj = self._compute_cost(generation_cost, _export, _import)

        epv, demand = self._update_demand_epv()

        self.grid_state = GridState(soc=soc_tp1, epv=epv, demand=demand, time_step=self.env_step,
                                    grid_status=self.grid.get_grid_status())

        action = (action.charge, action.discharge, action.generation['gen'])

        self.cost += multi_obj['total_cost']

        info = self._make_info(_export, _import, charge, consumption, discharge, production, generation, action)
        info.update(multi_obj)

        self.infos.append(info)
        reward = -multi_obj['total_cost']
        assert isinstance(reward, float)
        return self.grid_state.as_numpy(),reward , self._is_terminal(), info

    def plot(self, path):
        infos = _list_to_dict(self.infos)
        render.store_and_plot(infos, self.env_config, output_path=path)

    def _make_info(self, _export, _import, charge, consumption, discharge, production, generation, action):
        dt = self.grid.db.time_to_idx[self.env_step]
        dt = dt.strftime('%Y-%m-%d %H:%M:%S')
        info = {
            "dates": dt,
            "soc": self.grid_state.soc,
            "capacity": self.grid.storage.capacity,
            "charge": charge,
            "discharge": discharge,
            "non_steerable_consumption": self.grid_state.demand,
            "non_steerable_production": self.grid_state.epv,
            "res_gen_capacity": self.grid.epv.capacity,
            "cumulative_cost": self.cost,
            "generation": generation,
            "production": production,
            "consumption": consumption,
            "grid_import": _import,
            "grid_export": _export,
        }
        return info

    def _is_terminal(self):
        return self.env_step == self.grid.db.max_steps - 1

    def _compute_next_storage(self, action):
        soc_tp1, charge, discharge = self.grid.storage.simulate(self.grid_state.soc,
                                                                action.charge,
                                                                action.discharge,
                                                                self.grid.dt)
        return soc_tp1, charge, discharge

    def _apply_control(self, action, charge, discharge):
        generation, generation_cost = self.grid.engine.simulate(action.generation['gen'], self.grid.dt)
        production = self.grid_state.epv + discharge + generation
        consumption = self.grid_state.demand + charge
        return production, consumption, generation_cost, generation

    def _compute_cost(self, generation_cost, _export, _import):
        curtailment = _export * self.env_config['curtailment_price']
        load_shedding = _import * self.env_config['load_shedding_price']

        _total_cost = load_shedding + curtailment + generation_cost
        multiobj = {'total_cost': _total_cost,
                    'load_shedding': load_shedding,
                    'fuel_cost': generation_cost,
                    'curtailment': curtailment,
                    'storage_maintenance': self.grid.storage.n_cycles
                    }
        return multiobj

    def _gather_action(self, action):
        action_bound = self.grid.gather_action_space()
        if not isinstance(action, GridAction):
            if self.env_config['action_space'].lower() == "discrete":

                #action = self.grid._action_space_dict[action]
                #
                action = _construct_action_(action, self.grid_state, self.grid)
                # action = _construct_action_from_cluster(action, self._action_cluster, action_bound)
            else:
               # action = _construct_action_from_list(action, self.grid.n_storages, action_bound)
                action = _construct_charge_action(action, self.grid_state, self.grid)
        return action
