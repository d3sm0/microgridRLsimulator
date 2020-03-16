from microgridRLsimulator.gym_wrapper.microgrid_env import MicrogridEnv


def make_benchmark():
    base_key = lambda m: f"MicroGrid-{m}-v0"
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    task = {}
    fmt = lambda m: (f"2016-{m:02}-01", f"2016-{m:02}-28")

    for month_idx, month in enumerate(months):
        task[base_key(month)] = fmt(month_idx + 1)

    task.update( {
        "MicroGrid-May-2018-v0":("2018-05-01", "2016-05-28"),
        "MicroGrid-Jul-2018-v0":("2018-07-01", "2016-07-28"),
        "MicroGrid-Aug-2018-v0":("2018-08-01", "2016-08-28")
     })

    return task


_benchmark = make_benchmark()


def parse_id(env_id):
    try:
        dates = _benchmark[env_id]
    except KeyError:
        raise KeyError(f"Benchmark not in benchmark list {_benchmark.keys()}")
    return dates


def _test_benchmarks():
    for env_id in _benchmark.keys():
        _test_benchmark(env_id)


def _test_benchmark(env_id):
    start_date, end_date = parse_id(env_id)

    env = MicrogridEnv(
        start_date=start_date,
        end_date=end_date,
        case='elespino')

    env.reset()
    while True:
        _, _, done, _ = env.step(env.action_space.sample())
        if done:
            break


if __name__ == "__main__":
    _test_benchmarks()
