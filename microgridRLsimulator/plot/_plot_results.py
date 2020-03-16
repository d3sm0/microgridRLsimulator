## editing file
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
# import mpld3
import numpy as np

plt.style.use('bmh')

# print(plt.style.available)
FONT_SIZE = 12


class Plotter:
    def __init__(self, results, case, savefigs=True):
        """

        :param results: A json type dictionary containing results
        :param case: Name of the case, as a string
        """
        self.results = results
        self.case = case
        self.savefigs = savefigs

        self.dates = [datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in results["dates"]]

    def get_ticks(self, start, end):
        xstep = int(max(1, (end - start) / 24))
        dates = self.dates[start:end]
        xtick_labels = dates[::xstep]
        xticks = range(0, len(dates), xstep)
        return xticks, xtick_labels

    def plot_results(self, from_date=None, to_date=None, case=None):

        if from_date is None:
            from_date = self.dates[0]
        if to_date is None:
            to_date = self.dates[-1]

        if from_date < self.dates[0]:
            raise ValueError('From date cannot be before %s' % self.dates[0])
        if to_date > self.dates[-1]:
            raise ValueError('To date cannot be after %s' % self.dates[-1])
        if to_date < from_date:
            raise ValueError('To date cannot be after from date')

        # Index of from_date in self.dates
        start = self.dates.index(from_date)
        end = self.dates.index(to_date) + 1  # It's range(start,end), so add 1 to take end into account
        figures = [self.plot_batteries(start, end),
                   self.plot_costs(start, end),
                   self.plot_flows(start, end),
                   self.plot_power_mix(start, end),
                   self.plot_res(start, end)]

        if case is not None:
            for fig in figures:
                fig.suptitle(case)

        # if self.savefigs:
        #     with open('%s_resutls.html' % self.case, 'w') as f:
        #         f.write("".join([mpld3.fig_to_html(fig) for fig in figures]))

        if self.results['avg_rewards']:
            self.plot_learning_progress()

    def plot_res(self, start, end):

        xticks, xticks_labels = self.get_ticks(start, end)

        for g in range(len(self.results["res_gen_capacity"][0])):
            res_gen = [x for x in self.results["non_steerable_production"][start:end]]
            capacity = [x[g] for x in self.results["res_gen_capacity"][start:end]]

            fig = plt.figure(figsize=(16, 9))

            ax1 = plt.subplot(1, 1, 1)
            ax1.set_ylabel('kW', fontsize=FONT_SIZE)
            ax1.plot(capacity, label="Capacity")
            ax1.plot(res_gen, label="RES production", drawstyle="steps-post")
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xticks_labels)
            ax1.legend(fontsize=FONT_SIZE)
            fig.autofmt_xdate()
            if self.savefigs:
                fig.savefig('%s_res_gen.pdf' % self.case)
                ax1.set_xticklabels(list(range(len(res_gen))))

            return fig

    def plot_batteries(self, start, end):

        xticks, xticks_labels = self.get_ticks(start, end)

        for b in range(len(self.results["soc"][0])):
            soc = [x[b] for x in self.results["soc"][start:end]]
            capacity = [x[b] for x in self.results["capacity"][start:end]]
            charge = [-x[b] for x in self.results["charge"][start:end]]
            discharge = [x[b] for x in self.results["discharge"][start:end]]

            fig = plt.figure(figsize=(16, 9))

            ax1 = plt.subplot(2, 1, 1)
            ax1.set_ylabel('kWh', fontsize=FONT_SIZE)
            ax1.plot(capacity, label="Capacity")
            ax1.plot(soc, 'k', label="State of charge")
            ax1.set_xticks(xticks)
            ax1.set_xticklabels(xticks_labels)
            ax1.legend(fontsize=FONT_SIZE)

            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            ax2.set_ylabel('kW', fontsize=FONT_SIZE)
            ax2.plot(discharge, label="Discharge", drawstyle='steps-post')
            ax2.plot(charge, label="Charge", drawstyle='steps-post')
            ax2.set_ylim([min(charge) * 1.1 - 1e-3, max(discharge) * 1.1 + 1e-3])
            ax2.legend(fontsize=FONT_SIZE)
            # ax2.axhline(y=0, color='k', lw=0.5)

            fig.autofmt_xdate()
            if self.savefigs:
                fig.savefig('%s_battery_%d_soc.pdf' % (self.case, b))

                ax1.set_xticklabels(list(range(len(discharge))))
                ax2.set_xticklabels(list(range(len(discharge))))

            return fig

    def plot_costs(self, start, end):

        xticks, xticks_labels = self.get_ticks(start, end)

        cum_total_cost = self.results["cum_total_cost"][start:end]
        energy_cost = self.results["energy_cost"][start:end]
        fuel_cost = np.array(self.results["fuel_cost"][start:end])
        curtailment_cost = np.array(self.results["curtailment_cost"][start:end])
        load_not_served_cost = np.array(self.results["load_not_served_cost"][start:end])
        cum_fuel_cost = np.pad(np.cumsum(fuel_cost), (1, 0), 'constant', constant_values=0)[
                        :-1]  # Cumulative sum translated to 1 in order to align with cum total cost
        cum_curtailment_cost = np.pad(np.cumsum(curtailment_cost), (1, 0), 'constant', constant_values=0)[:-1]
        cum_load_not_served_cost = np.pad(np.cumsum(load_not_served_cost), (1, 0), 'constant', constant_values=0)[:-1]
        fig = plt.figure(figsize=(16, 9))

        ax1 = plt.subplot(2, 1, 1)
        ax1.set_ylabel('EUR', fontsize=FONT_SIZE)
        ax1.plot(cum_total_cost, 'k', label="Cumulative total cost")
        ax1.fill_between(range(len(energy_cost)), cum_fuel_cost, label="Fuel cost")
        ax1.fill_between(range(len(energy_cost)), cum_fuel_cost, cum_fuel_cost + cum_curtailment_cost,
                         label="Curtailment cost")
        ax1.fill_between(range(len(energy_cost)), cum_fuel_cost + cum_curtailment_cost,
                         cum_fuel_cost + cum_curtailment_cost + cum_load_not_served_cost, label="Lost load cost")
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticks_labels)
        ax1.legend(fontsize=FONT_SIZE)

        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.set_ylabel('EUR', fontsize=FONT_SIZE)
        ax2.plot(energy_cost, label="Total cost", drawstyle='steps-post', color="k")
        ax2.fill_between(range(len(energy_cost)), fuel_cost, label="Fuel cost", step="post")
        ax2.fill_between(range(len(energy_cost)), fuel_cost, fuel_cost + curtailment_cost, label="Curtailment cost",
                         step="post")
        ax2.fill_between(range(len(energy_cost)), fuel_cost + curtailment_cost,
                         fuel_cost + curtailment_cost + load_not_served_cost, label="Lost load cost", step="post")
        ax2.set_ylim([min(energy_cost) * 1.1 - 1e-3, max(energy_cost) * 1.1 + 1e-3])
        ax2.legend(fontsize=FONT_SIZE)
        # ax2.axhline(y=0, color='k', lw=0.5)

        fig.autofmt_xdate()
        if self.savefigs:
            fig.savefig('%s_costs.pdf' % self.case)

            ax1.set_xticklabels(list(range(len(energy_cost))))
            ax2.set_xticklabels(list(range(len(energy_cost))))
        return fig

    def plot_flows(self, start, end):

        xticks, xticks_labels = self.get_ticks(start, end)

        exports = np.array(self.results["grid_export"][start:end])
        imports = np.array(self.results["grid_import"][start:end])
        net_export = exports - imports

        productions = self.results["production"][start:end]
        consumptions = [-x for x in self.results["consumption"]][start:end]

        fig = plt.figure(figsize=(16, 9))

        ax1 = plt.subplot(2, 1, 1)
        ax1.set_ylabel('kWh', fontsize=FONT_SIZE)
        ax1.plot(net_export, 'k', label="Net export to grid", drawstyle='steps-post')
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticks_labels)
        ax1.legend(fontsize=FONT_SIZE)
        # ax1.axhline(y=0, color='k', lw=0.5)

        ax2 = plt.subplot(2, 1, 2, )
        ax2.set_ylabel('kW', fontsize=FONT_SIZE)
        ax2.plot(productions, label="Production", drawstyle='steps-post')
        ax2.plot(consumptions, label="Consumption", drawstyle='steps-post')
        ax2.set_ylim([min(consumptions) * 1.1, max(productions) * 1.1])
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xticks_labels)
        ax2.legend(fontsize=FONT_SIZE)
        # ax2.axhline(y=0, color='k', lw=0.5)

        fig.autofmt_xdate()
        if self.savefigs:
            fig.savefig('%s_flows.pdf' % self.case)
            ax1.set_xticklabels(list(range(len(productions))))
            ax2.set_xticklabels(list(range(len(productions))))

        return fig

    def plot_power_mix(self, start, end):

        xticks, xticks_labels = self.get_ticks(start, end)

        exports = np.array(self.results["grid_export"][start:end])
        imports = np.array(self.results["grid_import"][start:end])

        productions = np.array(self.results["production"][start:end])
        non_steerable_productions = np.array(self.results["non_steerable_production"][start:end])

        consumptions = np.array([x for x in self.results["consumption"]][start:end])
        non_steerable_consumptions = np.array(self.results["non_steerable_consumption"][start:end])

        charge = np.array([sum(x) for x in self.results["charge"][start:end]])
        discharge = np.array([sum(x) for x in self.results["discharge"][start:end]])

        generation = np.array([sum(x) for x in self.results["generation"][start:end]])

        fig = plt.figure(figsize=(16, 9))

        ax1 = plt.subplot(2, 1, 1)
        ax1.set_ylabel('kWh', fontsize=FONT_SIZE)

        ax1.fill_between(range(len(generation)), generation, label="Genset", step='post')
        ax1.fill_between(range(len(generation)), generation, generation + discharge, label="Discharges", step='post')
        ax1.fill_between(range(len(generation)), generation + discharge,
                         generation + discharge + non_steerable_productions, label="RES production", step='post')
        ax1.fill_between(range(len(generation)), generation + discharge + non_steerable_productions,
                         generation + discharge + non_steerable_productions + imports, label="Load Shedding",
                         step='post')

        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xticks_labels)
        ax1.legend(fontsize=FONT_SIZE)
        # ax1.axhline(y=0, color='k', lw=0.5)

        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.set_ylabel('kW', fontsize=FONT_SIZE)

        ax2.fill_between(range(len(charge)), non_steerable_consumptions, label="Load", step='post')
        ax2.fill_between(range(len(charge)), non_steerable_consumptions, charge + non_steerable_consumptions,
                         label="Charges", step='post')
        ax2.fill_between(range(len(charge)), charge + non_steerable_consumptions,
                         charge + non_steerable_consumptions + exports, label="Curtailment", step='post')

        ax2.set_ylim([0, max(max(consumptions + exports), max(productions + imports)) * 1.1])
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xticks_labels)
        ax2.legend(fontsize=FONT_SIZE)
        # ax2.axhline(y=0, color='k', lw=0.5)

        fig.autofmt_xdate()
        if self.savefigs:
            fig.savefig('%s_gen_mix.pdf' % self.case)

            ax1.set_xticklabels(list(range(len(charge))))
            ax2.set_xticklabels(list(range(len(charge))))

        return fig
        # y = mpld3.fig_to_html(fig)
        # with open('%s_gen_mix.html'% self.case, 'w') as f:
        #     f.write(y)

    def plot_learning_progress(self):

        fig = plt.figure(figsize=(16, 9))
        plt.title('Learning progress')
        plt.plot(range(len(self.results['avg_rewards'])), self.results['avg_rewards'])
        if self.savefigs:
            fig.savefig('%s_learning_progress.pdf' % self.case)


if __name__ == "__main__":

    arguments = sys.argv[1:]

    if arguments[0] == "all":
        folders = os.listdir("results")
    else:
        folders = arguments

    for folder in folders:
        path = "results/"+folder+"/"
        for file in [jsonfile for jsonfile in os.listdir("results/"+folder) if jsonfile.endswith('.json')]:

            with open(path+file, "rb") as json_results:
                results = json.load(json_results)

                plotter = Plotter(results, 'results/%s' % file.split("_")[0], savefigs=False)
                plotter.plot_results(case=folder)
    plt.show()
