
def _test_generator():
    epv_params = {
        "name": "EPV",
        "capacity": 240.0,
        "steerable": False,
        "operating_point": [365, 0.97]
    }

    params = {
        "name": "Diesel_1",
        "capacity": 58.0,
        "steerable": True,
        "min_stable_generation": 0.00,
        "diesel_price": 1,
        "operating_point_1": [75, 22.5],
        "operating_point_2": [25, 10.5]
    }

    from microgridRLsimulator.model.generator import Generator
    t = 10
    _epv = EPV(epv_params)
    generator = Generator(epv_params)
    _epv.update_capacity(t * 1 * 60)
    generator.update_capacity(t * 1 * 60)
    assert _epv.capacity == generator.capacity
    diesel = Diesel(params)

    generator = Generator(params)
    out = diesel.simulate(10, 1)
    out1 = generator.simulate_generator(10, 1)
    assert out == out1


if __name__ == '__main__':
    _test_generator()
