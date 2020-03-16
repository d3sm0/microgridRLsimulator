from microgridRLsimulator.model.DCAstorage import DCAStorage
from microgridRLsimulator.simulate.simulator import _DCAStorage

storage_params = {
    "capacity": 120.0,
    "charge_efficiency": 0.75,
    "discharge_efficiency": 0.75,
}
dca_params = {
    "max_charge_rate": 97.0,
    "max_discharge_rate": 97.0,
    "operating_point": [3000, 0.7]
}


def _test_storage():
    _dca = _DCAStorage(max_charge_rate=dca_params['max_charge_rate'],
                       max_discharge_rate=dca_params['max_discharge_rate'],
                       operating_point=dca_params['operating_point'],
                       storage_params=storage_params)

    dca = DCAStorage(name='dca', params={**storage_params, **dca_params})

    a = _dca.simulate(10, 2, 1, 1)
    b = dca.simulate(10, 2, 1, 1)
    assert a == b
    a = _dca.simulate(10, 2, -1, 1)
    b = dca.simulate(10, 2, -1, 1)
    assert a == b

    a = _dca.simulate(10, 2, 3, 1)
    b = dca.simulate(10, 2, 3, 1)
    assert a == b

if __name__ == '__main__':
    _test_storage()
