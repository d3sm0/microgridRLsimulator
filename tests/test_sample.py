
from microgridRLsimulator.gym_wrapper import MicrogridEnv


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

    import time
    env = MicrogridEnv(**env_init, params=params)
    for _ in range(10):
        s = env.sample()
        # print(s)

    env.reset()
    grid_state = env.get_state()
    _ = env.sample()
    env.set_state(grid_state)
    assert env.simulator.grid_state == grid_state
    assert env.simulator.env_step == grid_state.time_step
    assert env.simulator.grid.epv.capacity == grid_state.grid_status['epv']['capacity']


if __name__ == '__main__':
    main()
