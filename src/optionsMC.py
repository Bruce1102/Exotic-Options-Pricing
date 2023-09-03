from src.stochastic_process_simulation import StochasticProcessSimulation
import numpy as np

class Option:
    def __init__(self, underlying: StochasticProcessSimulation, strike:float, rate:float, tau:float, call_put:str = 'call'):
        """
        Initialize the Option.

        Parameters:
        - underlying : Stochastic process of underlying asset.
        - strike     : Strike price of option
        - rate       : Risk free rate.
        - tau        : Option's time to maturity.
        - call_put   : Option type, call or put
        """

        self.underlying = underlying
        self.strike = strike
        self.rate = rate
        self.tau = tau
        self.call_put = call_put

        self.hist_prices = np.zeros(underlying.n_steps)
    
    def get_params(self):
        params = {'initial_price': self.underlying.initial_price,
                  'rate': self.rate,
                  'tau' : self.tau,
                  'strike': self.strike,
                  'n': self.underlying.n_steps}
        return params

    def simulate(self):
        self.underlying.simulate()
        self.hist_prices = self.underlying.get_simulation()

    def get_simulated_prices(self):
        return self.hist_prices

    def get_payoff(self, spot):
        if self.call_put == 'call':
            return max(spot - self.strike, 0)
        else:
            return max(self.strike - spot, 0)


class Asian(Option):
    def __init__(self, underlying: StochasticProcessSimulation, strike:float, rate:float, 
                 tau:float, call_put:str='call', average_type:str='arithmetic', 
                 float_fixed:str='floating', fixed:int = 0):

        super().__init__(underlying, strike, rate, tau, call_put)

        self.average_type = average_type
        self.floating_fixed = float_fixed
        self.fixed = fixed #fixed average rate


    def _compute_average(self):
        if self.average_type == 'arithmetic': 
            return self.hist_prices.mean()
        else: 
            return np.prod(self.hist_prices) / len(self.hist_prices)

    def get_payoff(self, spot):
        average = self._compute_average()
        value = (average, self.fixed)

        if self.floating_fixed == 'floating':
            value = (spot, average)

        if self.call_put == 'call':
            return max(value[0] - value[1], 0)
        else:
            return max(value[1] - value[0], 0)
            
class LookBack(Option):
    def __init__(self, underlying: StochasticProcessSimulation, strike:float, rate:float, 
                 tau:float, call_put:str='call', min_max:str='max'):

        super().__init__(underlying, strike, rate, tau, call_put)

        self.min_max = min_max

    def get_payoff(self, spot):
        compute_value = max(self.hist_prices)

        if self.floating_fixed == 'floating':
            value = (spot, average)

        if self.call_put == 'call':
            return max(value[0] - value[1], 0)
        else:
            return max(value[1] - value[0], 0)
