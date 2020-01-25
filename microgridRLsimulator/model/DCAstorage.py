from microgridRLsimulator.model.storage import Storage
from microgridRLsimulator.utils import positive


class DCAStorage(Storage):
    def __init__(self, params):
        """
        :param name: Cf. parent class
        :param params: dictionary of params, must include a capacity value , a max_charge_rate value,
        a max_discharge_rate value, a charge_efficiency value, a discharge_efficiency value and an operating point.
        """

        super().__init__(params)
        # The number of cycle is 0 at the beginning -> we already have one operating point
        self.operating_point = params["operating_point"]
        self.initial_capacity = params["capacity"]

    def update_capacity(self):
        """

        Decrease the storage capacity using a linear function.

        :return: Nothing, updates the capacity.
        """
        self.capacity = (self.initial_capacity * (self.operating_point[1] - 1) / self.operating_point[
            0]) * self.n_cycles + self.initial_capacity

    def simulate(self, initial_soc, charge_action, discharge_action, deltat):
        """

        :param initial_soc: initial state of charge of the battery [kWh]
        :param charge_action: Charge action from a controller [kW]
        :param discharge_action: Discharge action from a controller [kW]
        :param deltat: Period duration [h]
        :return: the next state of charge, the actual charge and the actual discharge.
        """

        next_soc = initial_soc
        actual_charge, actual_discharge = Storage.actual_power(charge_action, discharge_action)

        if positive(actual_charge):
            # Check if efficiency has to be considered or not
            self.update_cycles(throughput=actual_charge * self.charge_efficiency, deltat=deltat)
            self.update_capacity()
            planned_evolution = initial_soc + actual_charge * deltat * self.charge_efficiency
            next_soc = min(self.capacity, planned_evolution)  # Check if the next soc is lower than the new capacity
            actual_charge = (next_soc - initial_soc) / (self.charge_efficiency * deltat)

        elif positive(actual_discharge):
            planned_evolution = initial_soc - actual_discharge * deltat / self.discharge_efficiency
            next_soc = max(0, planned_evolution)
            actual_discharge = (initial_soc - next_soc) * self.discharge_efficiency / deltat
            # same for discharge
            self.update_cycles(throughput=actual_discharge / self.discharge_efficiency, deltat=deltat)
            self.update_capacity()

        return next_soc, actual_charge, actual_discharge


def get_storage(storage_type):
    return DCAStorage if "dca" in storage_type.lower() else Storage


class Storages:
    def __init__(self, storages):
        self.storages = []
        for storage_params in storages:
            storage = get_storage(storage_params["type"])(storage_params)
            self.storages.append(storage)

    def append(self, storage):
        self.storages.append(storage)

    def simulate(self, args, delta_t):

        res = []
        for idx, (params) in enumerate(zip(*args)):
            out = self.storages[idx].simulate(*params, deltat=delta_t)
            res.append(out)

        return list(map(lambda x: list(x), zip(*res)))


    def n_cycles(self):
        return [storage.n_cycles for storage in self.storages]

    def capacity(self):
        return [storage.n_cycles for storage in self.storages]

    def __getitem__(self, item):
        return self.storages[item]

    def __iter__(self):
        for storage in self.storages:
            yield storage

    def __len__(self):
        return len(self.storages)
