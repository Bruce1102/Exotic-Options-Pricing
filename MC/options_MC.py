from MC.stochastic_process_simulation import *
import numpy as np

class Option:
    """Base class for standard vanila european options."""
    def __init__(self, underlying: StochasticProcessSimulation, strike:float, rate:float, tau:float, call_put: str='call'):
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

    def get_payoff(self, spot: float) -> float:
        if self.call_put == 'call':
            return max(spot - self.strike, 0)
        else:
            return max(self.strike - spot, 0)


class AsianOption(Option):
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
        self.fixed = fixed


    def _compute_average(self):
        if self.average_type == 'arithmetic': 
            return self.hist_prices.mean()
        else: 
            return np.prod(self.hist_prices) / len(self.hist_prices)

    def get_payoff(self, spot: float) -> float:
        average = self._compute_average()
        value = (average, self.fixed)

        if self.floating_fixed == 'floating':
            value = (spot, average)

        if self.call_put == 'call':
            return max(value[0] - value[1], 0)
        else:
            return max(value[1] - value[0], 0)
            
class LookbackOption(Option):
    """LookBack option class."""
    def __init__(self, underlying: StochasticProcessSimulation, strike:float, rate:float, 
                 tau:float, call_put:str='call'):

        super().__init__(underlying, None, rate, tau, call_put)

        self.min_max = (min if self.call_put == 'call' else max)


    def get_payoff(self, spot: float) -> float:
        min_max = self.min_max(self.hist_prices)

        return max(spot - min_max, 0) if self.call_put == 'call' else max(min_max - spot, 0)


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

        self.knocked_in = False
        self.knocked_out = False

    def _check_knocked(self):
        simulated_prices = underlying.get_simulation()
        for price in simulated_prices:
            if (self.knock == 'up-out' and price >= self.barrier) or (self.knock == 'down-out' and price <= self.barrier):
                self.knocked_out = True
                break
            if (self.knock == 'up-in' and price >= self.barrier) or (self.knock == 'down-in' and price <= self.barrier):
                self.knocked_out = True
                break

    def get_payoff(self, spot: float) -> float:
        """Calculate the payoff for a barrier option."""
        # Check for knocked in or out
        self._check_knocked(self)
        
        # if it has been knocked in or not knocked out
        if self.knocked_in or (self.knock[-3:] == 'out' and not self.knocked_out):
            return max(spot - self.strike, 0) if self.put_call == 'call' else max(self.strike - spot, 0)
        else:
            return self.rebate


class American(Option):
    """Base class for standard vanila european options."""
    def __init__(self, underlying: StochasticProcessSimulation, strike:float, rate:float, tau:float, dividend: float, call_put:str = 'call'):
        super().__init__(underlying, strike, rate, tau, dividend, call_put)
    
    def early_exercise_value(self, spot:float, time: float) -> float:
        """ Calculate the value of the option if it's exercised early."""
        discounted_payoff = np.exp(-self.rate * (self.tau - time)) * self.get_payoff(spot)
        return discounted_payoff

    def get_optimal_payoff(self, spot: float, time: float) -> float:
        """Determine the optimal payoff between exercising now or continuing to hold the option."""
        # Calculate the value if the option is exercised now
        early_exercise_val = self.early_exercise_value(spot, time)

        # Calculate the expected value if the option is held (this is a placeholder and should be replaced with a proper model)
        continuation_val = self.get_payoff(spot)  # This is a simplification; in practice, you'd use a model like binomial tree or finite difference method.

        # Return the maximum of the two values
        return max(early_exercise_val, continuation_val)