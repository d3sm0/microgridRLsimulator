"""
This demo shows how to create an interact with an environment.
"""

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
        "forecast_type": "exact",
        "min_stable_generation": 0.,
        "prob_failure": 0.9
    }

    import time
    env = MicrogridEnv(**env_init, params=params)
    sum_reward = 0
    T = 50
    dt = []
    start = time.perf_counter()
    state = env.reset()
    for tt in range(T):
        # print("state: ", state)
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        state = next_state
        sum_reward += reward
        if done:
            dt.append(time.perf_counter() - start)
            print(sum_reward)
            break
    # Store and plot
    env.render("plots/")


if __name__ == '__main__':
    main()
