import numpy as np


class Storage:
    storage_type = "Storage"

    def __init__(self, params):

        self.name = params['name']
        self.n_cycles = 0
        self.capacity = self.initial_capacity = params['capacity']
        self.charge_efficiency = params['charge_efficiency']
        self.discharge_efficiency = params['discharge_efficiency']

    def _simulate(self, soc, charge, discharge, dt):
        assert charge >= 0 and discharge >= 0 and soc >= 0

        soc = min(soc, self.capacity)

        if charge > 0:
            self.update_cycle(charge * self.charge_efficiency, dt)
            soc_tp1 = soc + charge * dt * self.charge_efficiency
            soc_tp1 = min(self.capacity, soc_tp1)
            charge = (soc_tp1 - soc) / (self.charge_efficiency * dt)
        elif discharge > 0:
            self.update_cycle(discharge * self.discharge_efficiency, dt)
            soc_tp1 = soc - discharge * dt / self.discharge_efficiency
            soc_tp1 = max(0, soc_tp1)
            discharge = (soc - soc_tp1) * (self.discharge_efficiency / dt)
        else:
            soc_tp1 = soc

        assert charge >= 0 and discharge >= 0 and soc_tp1 >= 0, "something is wrong here"
        return soc_tp1, charge, discharge

    def update_cycle(self, throughput, dt):
        raise NotImplementedError("Should be implemented by downstream class")

    def update_capacity(self):
        pass

    def simulate(self, soc, charge, discharge, dt):
        charge, discharge = _power(charge, discharge)
        self.update_capacity()
        soc_tp1, charge, discharge = self._simulate(soc, charge, discharge, dt)
        return soc_tp1, charge, discharge

    def reset(self):
        self.capacity = self.initial_capacity
        self.n_cycles = 0


def _power(charge, discharge):
    assert charge >= 0 and discharge >= 0
    net = charge - discharge
    charge = max(0., net)
    discharge = max(0., -net)
    return charge, discharge


class DCAStorage(Storage):
    storage_type = "DCAStorage"

    def __init__(self, params):
        super(DCAStorage, self).__init__(params)
        self.max_charge_rate = params['max_charge_rate']
        self.max_discharge_rate = params['max_discharge_rate']
        self.operating_point = params['operating_point']
        self.max_downtime = params['max_downtime']  # hour
        self.prob_failure = params['prob_failure']
        self.is_stochastic = bool(self.prob_failure)
        self.downtime = 0

    def update_capacity(self):
        if self.capacity > 0:
            self.capacity = self.initial_capacity + (
                    self.initial_capacity * (self.operating_point[1] - 1) / self.operating_point[0]) * self.n_cycles
        else:
            self.downtime += 1

    def simulate(self, soc, charge, discharge, dt):
        if self.is_stochastic:
            self.power_off()
        next_state = super(DCAStorage, self).simulate(soc, charge, discharge, dt)
        self.repair()
        return next_state

    def update_cycle(self, throughput, dt):
        if self.capacity > 0:
            self.n_cycles += throughput * dt / (2 * self.capacity)

    def repair(self):
        if self.downtime == self.max_downtime and self.capacity == 0:
            self.reset()
            self.prob_failure = 0.

    def power_off(self):
        if self.prob_failure > np.random.uniform() and self.capacity > 0:
            self.capacity = 0
            self.downtime = 0
        elif self.capacity > 0:
            self.prob_failure = min(self.prob_failure + 1 / self.capacity, 1.)

    def max_charge(self):
        return self.capacity * self.max_charge_rate / 100.

    def max_discharge(self):
        return self.capacity * self.max_discharge_rate / 100.
