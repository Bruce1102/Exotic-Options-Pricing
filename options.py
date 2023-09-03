import numpy as np
import math

class OptionsBlackScholes:
    def __init__(self, strike: float, rate: float, 
                 tau: float, sigma: float, bounds: tuple, 
                 put_call: str):

        self.strike   = strike
        self.rate     = rate
        self.tau      = tau
        self.sigma    = sigma
        self.put_call = put_call
        self.x_low    = bounds[0]
        self.x_up     = bounds[1]

    def payoff(self, spot: float):
        if self.put_call == 'put':
            return max(self.strike - spot, 0)
        else:
            return max(spot - self.strike, 0)
    
    def get_coefficients(self, t: float, spot: float): 
        """Return coefficients of Black Scholes formula"""
        coeff = {'a': - 0.5 * (spot*self.sigma(t, spot))**2,
                 'b': - self.rate * spot,
                 'c': self.rate,
                 'd': 0}

        return coeff

    def adjust_grid(self, grid: np.array, dx: int):
        return grid

    def bound_cond_tup(self,  spot):
        if self.put_call = 'call':
            return max(spot - self.strike, 0)
        else:
            return max(self.strike - spot, 0)

    def bound_cond_x_low(self, t):
        if self.put_call == 'call':
            return 0.0
        else:
            return self.strike * np.exp(-self.rate * (self.tau - t))
    
    def bound_cond_x_up(self, t):
        if self.put_call == 'call':
            return self.strike * np.exp(-self.rate * (self.tau - t))
        else:
            return 0.0


class Barrier(OptionsBlackScholes):
    def __init__(self, rate, sigma, strike, barrier, t_up, x_low, x_up, knock = 'up-out', put_call = "call"):
        """ Constructor of a up-and-out European Call option
        Parameters:
            rate   : underlying asset's risk free interest rate
            sigma  : lambda function of underlying asset's standard deviation
            strike : opption's strike price
            barrier: level of the barrier
            t_up   : option's time to expriry
            x_low  : lower bound for the option's spot price
            x_up   : upper bound for the option's spot price
        """

        self.rate = rate
        self.sigma = sigma
        self.strike = strike
        self.barrier = barrier
        self.t_up = t_up
        self.x_low = x_low
        self.x_up = x_up

        self.knock = knock
        self.put_call = put_call

    def _is_zero(self, spot):
        """Compute upper boundary conditions for time"""
        if (self.knock == 'up-out' and spot >= self.barrier) or (self.knock == 'down-out' and spot <= self.barrier):
            return True
        elif (self.knock == 'up-in' and spot < self.barrier) or (self.knock == 'down-in' and spot > self.barrier):
            return True
        else: 
            return False
    
    # Returning self boundary conditions:
    def bound_cond_tup(self,  spot):
        if self._is_zero(spot):
            return 0

        if self.put_call == 'call':
            return max(spot - self.strike, 0)
        else:
            return max(self.strike - spot, 0)

    def bound_cond_x_low(self, t):
        if self.put_call == 'call':
            return 0
        else:
            return self.strike * np.exp(-self.rate * (self.tau - t))
    
    def bound_cond_x_up(self, t):
        if self.knock == 'up-out':
            return 0.0
        elif self.knock == 'down-out':
            return self.strike * np.exp(-self.rate * (self.tau - t))
        elif self.knock == 'up-in':
            return self.x_up - self.strike * np.exp(-self.rate * (self.tau - t))
        elif self.knock == 'down-in':
            return 0.0