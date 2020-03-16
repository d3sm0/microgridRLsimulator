import json
import os

from .plot_results import Plotter

def store_and_plot(metrics, env_config, policy=None, output_path="plots/"):
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
#    try:
    with open(os.path.join(output_path, f"{env_config['case']}_out.json"), 'w') as jsonFile:
        json.dump(metrics, jsonFile)
#    except TypeError:
#        print("metrics not jsonable")

    try:
        with open(os.path.join(output_path, "env_config.json"), 'w') as jsonFile:
            json.dump(env_config, jsonFile)
    except TypeError:
        print("config not jsonable")

    if policy is not None:
        try:
            with open(os.path.join(output_path, "agent_options.json"), 'w') as jsonFile:
                json.dump(policy, jsonFile)
        except TypeError:
            print("policy not jsoanble")

    plotter = Plotter(metrics, os.path.join(output_path, env_config['case']))
    plotter.plot_results()
