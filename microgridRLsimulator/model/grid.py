from microgridRLsimulator.model.DCAstorage import DCAStorage
from microgridRLsimulator.model.generator import Generator
from microgridRLsimulator.model.load import Load
from microgridRLsimulator.model.storage import Storage


class Grid:
    def __init__(self, data):
        """
        A microgridRLsimulator is represented by its devices which are either loads, generators or storage
        devices, and additional information such as prices.
        The period duration of the simulation is also stored at this level, although
        it is more part of the configuration of the simulation.

        :param data: A json type dictionary containing a description of the microgridRLsimulator.
        """
        self.loads = []
        for l in data["loads"]:
            self.loads.append(Load(l["name"], l["capacity"]))
        self.generators = []
        for g in data["generators"]:
            self.generators.append(Generator(g["name"], g))
        STORAGE_TYPES = {Storage.type(): Storage, DCAStorage.type(): DCAStorage}

        self.storages = []
        for s in data["storages"]:
            self.storages.append(STORAGE_TYPES[s["type"]](s["name"], s))

        self.period_duration = data["period_duration"] / 60  # minutes -> hours

        self.curtailment_price = data["curtailment_price"]
        self.load_shedding_price = data["load_shedding_price"]

    def get_non_flexible_device_names(self):
        """

        :return: The list of names of all non-flexible loads and generators for which there must be an entry in the data history
        """
        names = [d.name for d in self.loads]
        for d in self.generators:
            if not d.steerable:
                names.append(d.name)

        return names
