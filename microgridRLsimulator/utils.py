import collections
import json
from datetime import datetime
import time
import os

pkg_dir = os.path.dirname(__file__)
MICROGRID_CONFIG_FILE = lambda case: os.path.join(pkg_dir, "data", case, f"{case}.json")
MICROGRID_DATA_FILE = lambda case: os.path.join(pkg_dir, "data", case, f"{case}_dataset.csv")
RESULT_PATH = lambda case: os.path.join(pkg_dir, "data", case, f"{case}.json")


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, type, value, traceback):
        self.end = time.clock()
        print(f"{self.name}:{self.end - self.start}")


def type_checker(fn):
    def _wrap(*args, **kwargs):
        out = fn(*args, **kwargs)
        return list(map(out, check_type))

    return _wrap


def check_type(args):
    import numpy as np
    for a in args:
        assert isinstance(a, float) or isinstance(a, int), f"Found instance {type(a)}"


def time_string_for_storing_results(name, case):
    """

    :param case: the case name
    :return a string used for file or folder names
    """
    return name + f"_{case}_{datetime.now().strftime('%Y-%m-%d_%H%M')}"


def check_json(value):
    try:
        json.dumps(value)
        return True
    except TypeError:
        return False


def _list_to_dict(info):
    infos = collections.defaultdict(lambda: [])
    for _info in info:
        for k, value in _info.items():
            value = value if isinstance(value, str) else float(value)
            assert check_json(value)
            infos[k].append(value)
    return dict(infos)


def load_config(fname):
    with open(fname, 'rb') as jsonFile:
        data = json.load(jsonFile)
    return data
