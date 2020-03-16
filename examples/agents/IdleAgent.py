# -*- coding: utf-8 -*-

from copy import deepcopy

from microgridRLsimulator.utils import time_string_for_storing_results
from .agent import Agent


class IdleAgent(Agent):

    def __init__(self, env, n_test_episodes=1):
        super().__init__(env)
        self.n_test_episodes = n_test_episodes

    @staticmethod
    def name():
        return "Idle"

    def train_agent(self):
        pass  # Nothing to train the Idle agents with

    def simulate_agent(self, agent_options=None, **kwargs):
        for i in range(1, self.n_test_episodes + 1):
            state = self.env.reset()
            cumulative_reward = 0.0
            done = False

            while not done:
                # Take always the last action in the action space - Idle always
                action = len(self.env.simulator.high_level_actions) - 1
                next_state, reward, done, info = self.env.step(actions=action)
                # reward = self.reward_function(reward_info)
                cumulative_reward += reward
                state = deepcopy(next_state)
            print('i am in episode: %d and the reward is: %d.' % (i, cumulative_reward))
            self.env.simulator.store_and_plot(
                folder="results/" + self.name() + "/" + self.env.simulator.case + "/" + time_string_for_storing_results(
                    self.name() + "_" + self.env.purpose + "_from_" + self.env.simulator.start_date.strftime(
                        "%m-%d-%Y") + "_to_" + self.env.simulator.end_date.strftime("%m-%d-%Y"),
                    self.env.simulator.case) + "_" + str(i), agent_options=agent_options)

    def reward_function(self, reward_info):
        """
        Method that transforms the reward infos into a reward value with the help of a reward function tuned by the user.

        :param reward_info: dictionary that contains reward information relative to the chosen objectives 
        (total_cost, fuel_cost, load_shedding, curtailment, storage_maintenance).
        :return: reward value from a tuned reward function.
        """
        reward = - reward_info["total_cost"]
        return reward


agent_type = IdleAgent
