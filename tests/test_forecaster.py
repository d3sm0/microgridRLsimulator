

if __name__ == '__main__':
    env_init = {
        "start_date": "2016-01-01",
        "end_date": "2016-01-31",
        "data_file": "elespino"
    }

    params = {
        "action_space": "Discrete",
        "backcast_steps": 0,
        "forecast_steps": 0,
        "forecast_type": "exact"}

    from microgridRLsimulator.gym_wrapper import MicrogridEnv

    env = MicrogridEnv(**env_init, params=params)

    fs = _Forecaster(env.simulator.grid)

    fs.exact_forecast(720, 10)
    fs.exact_forecast(0, 10)
    fs.noisy_forecast(0, 10)
