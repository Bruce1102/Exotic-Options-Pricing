import numpy as np
class MonteCarlo:
    def __init__(self, option, n_sim:int = 1_000):
        self.option = option
        self.n_sim  = n_sim

        self.simulation_results = np.empty((n_sim, option.get_params()['n']))
        self.payoff = np.empty(n_sim)

    def simulate(self):
        for i in range(0, self.n_sim):
            self.option.simulate()
            self.simulation_results[i] = self.option.get_simulated_prices()
            self.payoff[i] = self.option.get_payoff(spot = self.simulation_results[i][-1])

    def get_simulation_results(self):
        return self.simulation_results

    def get_simulation_payoff(self):
        return self.payoff

    def get_fair_value(self):
        params = self.option.get_params()

        discounted_payoffs = np.exp(-params['rate'] * params['tau']) * self.payoff
        return np.mean(discounted_payoffs)