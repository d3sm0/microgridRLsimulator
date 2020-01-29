import numpy as np


class Generator:
    device_type = 'generator'

    def __init__(self, params):
        self.capacity = self.initial_capacity = params['capacity']
        self.name = params['name']
        self.steerable = False

    def simulate(self, production, dt):
        pass

    def get_capacity(self, time):
        pass

    def update_capacity(self, time):
        pass

    def reset(self):
        self.capacity = self.initial_capacity


class Diesel(Generator):
    device_type = 'diesel'

    def __init__(self, params):
        super(Diesel, self).__init__(params)

        self.steerable = True
        self.diesel_price = params['diesel_price']  # €/l

        operating_point_1 = params['operating_point_1']
        operating_point_2 = params['operating_point_2']

        self.min_stable_generation = params["min_stable_generation"] * self.capacity

        self.slope = (operating_point_1[1] - operating_point_2[1]) / (operating_point_1[0] - operating_point_2[0])
        self.intercept = operating_point_2[1] - operating_point_2[0] * self.slope

        self.compute_production = lambda production: min(max(self.min_stable_generation, production), self.capacity)

    def simulate(self, production, dt):
        assert production >= 0
        production = self.compute_production(production) if production > 0. else 0.
        v_dot_diesel = self.intercept * np.sign(production) + self.slope * production
        diesel_consumption_l = v_dot_diesel * dt
        total_cost = diesel_consumption_l * self.diesel_price
        return production, total_cost


class EPV(Generator):
    device_type = 'epv'

    def __init__(self, params):
        super(EPV, self).__init__(params)
        operating_point = params['operating_point']
        self.steerable = False
        self.step = lambda time: (operating_point[1] - 1) * time / (24 * 60 * operating_point[0])

    def get_capacity(self, time):
        capacity = self.initial_capacity * (1 + self.step(time))
        return capacity

    def update_capacity(self, time):
        self.capacity = self.get_capacity(time)

    def simulate(self, production, dt):
        raise NotImplementedError

    def usage(self):
        return self.capacity / self.initial_capacity

