"""
This file defines a class MicrogridEnv that wraps the Simulator in this package, so that it follows the
OpenAI gym (https://github.com/openai/gym) format.

"""

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from microgridRLsimulator.simulate.simulator import Simulator
from microgridRLsimulator.simulate.gridaction import GridAction
import copy


class MicrogridEnv(gym.Env):

    def __init__(self, start_date, end_date, case, purpose="Train", params=None):
        """
        :param start_date: datetime for the start of the simulation
        :param end_date: datetime for the end of the simulation
        :param case: case name (string)
        """

        self.simulator = Simulator(start_date, end_date, case, params=params)

        self.action_space = make_action_space(self.simulator)
        self.observation_space = make_observation_space(self.simulator)

        self.state = None
        self.np_random = None

        self.purpose = purpose
        self.seed()

    def sample(self):
        return self.simulator.sample()

    def set_state(self, state):
        self.simulator.set_state(state)

    def get_state(self):
        return copy.deepcopy(self.simulator.grid_state)

    def seed(self, seed=None):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    def reset(self):
        self.state = self.simulator.reset()
        return self._observation(self.state)

    def render(self, path):
        print(f"Plots store in {path}")
        self.simulator.plot(path)

    def step(self, action):
        """
        Step function, as in gym.
        May also accept a state as input (useful for MCTS, for instance).
        """

        assert self.action_space.contains(action) or isinstance(action, GridAction)
        self.state, reward, done, info = self.simulator.step(action)
        return self._observation(self.state), reward, done, info

    @staticmethod
    def _observation(state):
        return np.array(state, np.float32)


def make_action_space(simulator):
    if simulator.env_config['action_space'].lower() == "discrete":
        return spaces.Discrete(simulator.grid.gather_action_space())
    lower, upper = simulator.grid.gather_action_space()
    action_space = spaces.Box(lower, upper, dtype=np.float32)
    return action_space


def make_observation_space(simulator):
    lower, upper = simulator.grid.gather_observation_space()
    observation_space = spaces.Box(lower, upper, dtype=np.float32)
    return observation_space
