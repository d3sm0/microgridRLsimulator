"""
This demo shows how to create an interact with an environment.
"""

from microgridRLsimulator.gym_wrapper import MicrogridEnv


# Initialize environment
def main():
    env_init = {
        "start_date": "2015-01-01",
        "end_date": "2015-01-31",
        "data_file": "case1"
    }

    params = {
        "action_space": "Discrete",
        "backcast_steps": 0,
        "forecast_steps": 0,
        "forecast_type": "exact"}

    env = MicrogridEnv(**env_init, params=params)

    # Compute cumulative reward of a random policy
    sum_reward = 0
    T = 1000
    state = env.reset()
    for tt in range(T):
        print("state: ", state)
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        state = next_state

        sum_reward += reward
        if done:
            break

    # Store and plot
    # env.simulator.store_and_plot()


if __name__ == '__main__':
    main()
