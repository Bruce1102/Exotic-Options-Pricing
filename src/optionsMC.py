from src.stochastic_process_simulation import StochasticProcessSimulation
import numpy as np

class Option:
    """Base class for standard vanila european options."""
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
        if call_put not in ['call', 'put']:
            raise ValueError("Invalid call/put type. Choose from 'call', 'put'")

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
    """Asian options class."""
    def __init__(self, underlying: StochasticProcessSimulation, strike:float, rate:float, 
                 tau:float, call_put:str='call', average_type:str='arithmetic', 
                 float_fixed:str='floating', fixed:int = 0):

        if average_type not in ['arithmetic', 'geometric']:
            raise ValueError("Invalid average type. Choose from 'arithmetic', 'geometric'.")

        if float_fixed not in ['floating', 'fixed']:
            raise ValueError("Invalid float/fixed type. Choose from 'floating', 'fixed'.")

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
    """LookBack option class."""
    def __init__(self, underlying: StochasticProcessSimulation, strike:float, rate:float, 
                 tau:float, call_put:str='call', min_max:str='max'):
        
        if min_max not in ['max', 'min']:
            raise ValueError("Invalid min/max type. Choose from 'min', 'max'.")

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


class Barrier(Option):
    """Barrier option class."""
    def __init__(self, underlying: StochasticProcessSimulation, strike:float, rate:float, 
                 tau:float, barrier:float, rebate:float = 0, knock = 'up-out', call_put = "call"):

        if knock not in ['up-out', 'down-out', 'up-in', 'down-in']:
            raise ValueError("Invalid knock type. Choose from 'up-out', 'down-out', 'up-in', 'down-in'.")

        super().__init__(underlying, strike, rate, tau, call_put)

        self.barrier = barrier
        self.rebate = rebate
        self.knock = knock
        self.call = call

    

    def get_payoff(self, spot: float) -> float:
        """Calculate the payoff for a barrier option."""
        if self.knock == 'up-out' and any(price > self.barrier for price in self.hist_prices):
            return self.rebate
        if self.knock == 'down-out' and any(price < self.barrier for price in self.hist_prices): 
            return self.rebate
        if self.knock == 'up-in' and not any(price > self.barrier for price in self.hist_prices):
            return 0
        if self.knock == 'down-in' and not any(price < self.barrier for price in self.hist_prices):
            return 0
        return max(spot - self.strike, 0) if self.put_call == 'call' else max(self.strike - spot, 0)
