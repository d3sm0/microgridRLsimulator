from microgridRLsimulator.model.device import Device


class Storage(Device):
    def __init__(self, params):
        """

        :param name: Cf. parent class
        :param params: dictionary of params, must include a capacity value , a max_charge_rate value,
        a max_discharge_rate value, a charge_efficiency value and a discharge_efficiency value.
        """
        assert 'name' in params.keys()
        super(Storage, self).__init__(params['name'])

        self.capacity = None
        self.max_charge_rate = None
        self.max_discharge_rate = None
        self.charge_efficiency = 1.0
        self.discharge_efficiency = 1.0
        self.n_cycles = 0
        for k in params.keys():
            if k in self.__dict__.keys():
                self.__setattr__(k, params[k])

        assert (self.capacity is not None)
        assert (self.max_charge_rate is not None)
        assert (self.max_discharge_rate is not None)

    def update_cycles(self, throughput, deltat):
        """

        Update the storage number of cycles.

        :param throughput: power charged or discharged, in absolute value [kW]
        :param deltat: period duration [h]
        :return: Nothing, updates number of cycles.
        """
        self.n_cycles += throughput * deltat / (2 * self.capacity)

    def simulate(self, initial_soc, charge_action, discharge_action, deltat):
        """

        :param initial_soc: initial state of charge of the battery [kWh]
        :param charge_action: Charge action from a controller [kW]
        :param discharge_action: Discharge action from a controller [kW]
        :param deltat : Period duration [h]
        :return: the next state of charge, the actual charge and the actual discharge.
        """

        next_soc = initial_soc
        actual_charge, actual_discharge = self.actual_power(charge_action, discharge_action)
        # TODO check action is an energy: action is a power
        if actual_charge > 0:
            planned_evolution = initial_soc + actual_charge * deltat * self.charge_efficiency
            next_soc = min(self.capacity, planned_evolution)  # period duration used to modify it in an energy
            actual_charge = (next_soc - initial_soc) / (self.charge_efficiency * deltat)

        elif actual_discharge < 0:
            planned_evolution = initial_soc - actual_discharge * deltat / self.discharge_efficiency  # same for discharge
            next_soc = max(0, planned_evolution)
            actual_discharge = (initial_soc - next_soc) * self.discharge_efficiency / deltat

        return next_soc, actual_charge, actual_discharge

    @staticmethod
    def actual_power(charge_action, discharge_action):
        """

            :param charge_action: Charge action from a controller
            :param discharge_action: Discharge action from a controller
            :return: the actual charge and the actual discharge.
            """

        # Take care of potential simultaneous charge and discharge.
        if discharge_action < 0:
            return charge_action, discharge_action
        net = charge_action - discharge_action
        action = abs(net)
        return (action, 0) if net > 0 else (0, action)


