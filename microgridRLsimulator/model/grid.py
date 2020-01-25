from microgridRLsimulator.model.DCAstorage import DCAStorage, Storages
from microgridRLsimulator.model.generator import Generator, Engine
from microgridRLsimulator.model.load import Load
from microgridRLsimulator.model.storage import Storage


class Grid:
    STORAGE_TYPES = dict(DCAStorage=DCAStorage, Storage=Storage)

    def __init__(self, data):
        """
        A microgridRLsimulator is represented by its devices which are either loads, generators or storage
        devices, and additional information such as prices.
        The period duration of the simulation is also stored at this level, although
        it is more part of the configuration of the simulation.

        :param data: A json type dictionary containing a description of the microgridRLsimulator.
        """
        self.loads = []
        for load in data["loads"]:
            self.loads.append(Load(**load))

        self.generators = []
        #self.steerable_generators = []
        for g in data["generators"]:
            if g['steerable'] is False:
                self.generators.append(Generator(g))
            else:
                self.generators.append(Engine(g))
                #self.steerable_generators.append(Engine(g))

        self.storages = Storages(data["storages"])

        self.delta_t = data["period_duration"] / 60  # minutes -> hours

        self.curtailment_price = data["curtailment_price"]
        self.load_shedding_price = data["load_shedding_price"]

    def get_production(self, database, time, update_capacity=False):
        realized_non_flexible_production = 0.0
        for epv in self.generators:
            # assert epv.steerable is False
            if not epv.steerable:
                realized_non_flexible_production += database.get_columns(epv.name, time) * (
                            epv.capacity / epv.initial_capacity)
        return realized_non_flexible_production

    def update_capacity(self, time):
        total_capacity = []
        time = time * 60
        for epv in self.generators:
            if not epv.steerable:
                epv.update_capacity(time)
                total_capacity.append(epv.capacity)
        return total_capacity

    def get_load(self, database, time):

        realized_non_flexible_consumption = 0.0
        for l in self.loads:
            realized_non_flexible_consumption += database.get_columns(l.name, time)
        return realized_non_flexible_consumption


    @property
    def n_storages(self):
        return len(self.storages)

    @property
    def n_generators(self):
        return len(self.generators)

    def get_non_flexible_device_names(self):
        """

        :return: The list of names of all non-flexible loads and generators for which there must be an entry in the data history
        """
        names = [d.name for d in self.loads]
        for d in self.generators:
            if not d.steerable:
                names.append(d.name)

        return names
