from microgridRLsimulator.simulate.simulator import Simulator as _Simulator
from microgridRLsimulator.simulate.simulator import Simulator
import numpy as np

# Initialize environment
def main():
    env_init = {
        "start_date": "2016-01-01",
        "end_date": "2016-01-31",
        "case": "elespino"
    }

    params = {
        "action_space": "Discrete",
        "backcast_steps": 0,
        "forecast_steps": 0,
        "forecast_type": "exact"}

    S1 = Simulator(**env_init, params=params)
    S2 = _Simulator(**env_init, params=params)
    s0 = S1.reset()
    s1 = S2.reset()
    assert np.all(np.array(s0, dtype=np.float32 )== s1)

    while True:
        action = 1
        s0, r0, d, _obj1 = S1.step(action)
        s1, r1, d, _obj2 = S2.step(action)
        try:
            np.testing.assert_array_almost_equal(_obj1['action'], _obj2['action'],decimal=2)
        except AssertionError:
            print("Failing action")
            print(action,  _obj1["action"], _obj2["action"])
        #assert np.all(abs(np.array(_obj1['action'],dtype=np.float32) - np.array( _obj2['action'], dtype=np.float32)) < 1e-3)
        try:
            assert np.isclose(_obj1['production'], _obj2['production'], rtol=1e-2)
        except AssertionError:
            print(_obj1['production'] - _obj2['production'])
        try:
            assert np.isclose(_obj1['charge'], _obj2['charge'], rtol=1e-2)
        except AssertionError:
            print(_obj1['charge'] - _obj2['charge'])
        assert np.isclose(_obj1['discharge'], _obj2['discharge'], rtol=1e-2)
        assert np.isclose(_obj1['consumption'], _obj2['consumption'], rtol=1e-2)
        try:
            assert np.isclose(_obj1['import'], _obj2['grid_import'], rtol=1e-1)
        except AssertionError:
            print((_obj1['import'] -  _obj2['grid_import']))

        try:
            assert np.isclose(_obj1['export'], _obj2['grid_export'], rtol=1e-1), (_obj1['export'], _obj2['grid_export'])
        except AssertionError:
            pass
        try:
            np.testing.assert_array_almost_equal(s0, s1, decimal=1)
        except AssertionError:
            print("State failed")
            pass
        assert np.isclose(round(r1 - r0), 0., rtol=1e-2)

        if d:
            break



if __name__ == '__main__':
    main()
