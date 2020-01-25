import numpy as np

from microgridRLsimulator.model.device import Device


class Generator(Device):

    def __init__(self, params):
        """

        :param name: Cf. parent class
        :param params: dictionary of params, must include a capacity value , a steerable flag, and a min_stable_generation value
        """
        super(Generator, self).__init__(params['name'])

        self.steerable = False
        self.initial_capacity = params["capacity"]
        operating_point = params["operating_point"]
        self.capacity = self.initial_capacity
        self.step = lambda time: (operating_point[1] - 1) * time / (24 * 60 * operating_point[0])

    def update_capacity(self, time):
        """

        Decrease the generator capacity over time using a linear function.

        :return: Nothing, updates the generator capacity.
        """
        self.capacity = self.compute_capacity(time)

    def compute_capacity(self, time):
        """

        Calculate the generator capacity at the next step.

        :return: The capacity without updating it. Useful for lookahead.
        """
        capacity = self.initial_capacity * (1 + self.step(time))
        return capacity


class Engine(Device):
    def __init__(self, params):
        super(Engine, self).__init__(params['name'])

        # In oder to determine the efficiency we need two operating points
        # It is considered that the fuel curve is linear (HOMER)
        self.steerable = True
        self.diesel_price = params["diesel_price"]

        self.capacity = params["capacity"]
        self.min_stable_generation = params["min_stable_generation"]
        min_stable_generation = self.capacity * self.min_stable_generation

        self.compute_production = lambda production: min(max(min_stable_generation, production), self.capacity)

        operating_point_1 = params["operating_point_1"]
        operating_point_2 = params["operating_point_2"]

        self.slope = (operating_point_1[1] - operating_point_2[1]) / (operating_point_1[0] - operating_point_2[0])
        self.intercept = operating_point_2[1] - operating_point_2[0] * self.slope

    def simulate_generator(self, production, simulation_resolution):
        production = self.compute_production(production) if production > 0. else 0.
        v_dot_diesel = self.intercept * np.sign(production) + self.slope * production
        diesel_consumption_l = v_dot_diesel * simulation_resolution
        total_cost = diesel_consumption_l * self.diesel_price

        return production, total_cost
