"""
This file defines a class MicrogridEnv that wraps the Simulator in this package, so that it follows the
OpenAI gym (https://github.com/openai/gym) format.

"""

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from microgridRLsimulator.simulate import Simulator
from microgridRLsimulator.simulate.gridaction import GridAction
from microgridRLsimulator.utils import gather_action_dimension, gather_space_dimension


class MicrogridEnv(gym.Env):

    def __init__(self, start_date, end_date, data_file, purpose="Train", params=None):
        """
        :param start_date: datetime for the start of the simulation
        :param end_date: datetime for the end of the simulation
        :param case: case name (string)
        """

        self.simulator = Simulator(start_date, end_date, data_file, params=params)

        self.action_space = make_action_space(simulator=self.simulator)
        self.observation_space = make_observation_space(simulator=self.simulator)

        self.state = None
        self.np_random = None

        self.purpose = purpose
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.simulator.reset()
        return self._observation(self.state)

    def step(self, action):
        """
        Step function, as in gym.
        May also accept a state as input (useful for MCTS, for instance).
        """

        assert self.action_space.contains(action) or isinstance(action, GridAction)
        self.state, reward, done = self.simulator.step(action)
        return self._observation(self.state), reward, done, {}

    @staticmethod
    def _observation(state):
        return np.array(state, np.float32)


def make_action_space(simulator):
    if simulator.data['action_space'].lower() == "discrete":
        return spaces.Discrete(len(simulator.high_level_actions))
    lower, upper = gather_action_dimension(simulator)
    action_space = spaces.Box(lower, upper, dtype=np.float32)
    return action_space


def make_observation_space(simulator):
    lower, upper = gather_space_dimension(simulator)
    observation_space = spaces.Box(lower, upper, dtype=np.float32)
    return observation_space
