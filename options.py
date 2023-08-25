import math
import numpy as np


class OptionsBlackScholes:
    def __init__(self, strike: float, tau: float, put_call: str):
        self.strike = strike
        self.tau    = tau

    def payoff(self, spot:float):
        pass

    
class European(OptionsBlackScholes):
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


    def a(self, t: float, spot: float): 
        """Compute coefficient for second derivative"""
        return - 0.5 * (spot*self.sigma)**2
    def b(self, t: float, spot: float): 
        """Compute coefficient for first derivative"""
        return - self.rate * spot
    def c(self, t: float, spot: float): 
        """Compute coefficient for value"""
        return self.rate
    def d(self, t: float, spot: float):
        """Compute coefficient for intercept"""
        return 0

    def boundary_condition_tau(self, spot: float):
        """Compute upper boundary conditions for time"""
        if self.put_call == 'put':
            return max(self.strike - spot, 0)
        else:
            return max(spot - self.strike, 0)

    def boundary_condition_x_low(self, t: float):
        """Boundary condition for the lower bound of the spot price.""" 
        if self.put_call == 'put':
            # For a put option, as spot approaches 0, the option value approaches
            # the present value of the strike price
            return self.strike * np.exp(-self.rate * (self.tau - t))
        else:
            # For call option, as spot approaches to 0, the value will be 0
            return 00

    def boundary_condition_x_up(self, t: float):
        """Boundary condition for the upper bound of the spot price.""" 
        if self.put_call == 'put':
            # For put option, as spot goes to infinity, the option approahes to 0
            return 0.0
        else:
            # For call option, as spot approaches infinity, the option value approaches 
            # spot minus the present value of the strike price
            return self.x_up - self.strike * np.exp(-self.rate * (self.tau - t))



class Barrier(OptionsBlackScholes):
    def __init__(self, strike: float, rate: float, 
                 tau: float, sigma: float, barrier: float, bounds: tuple, 
                 knock: str, put_call: str):

        # Check if knock is 'in' or 'out'
        # Check if put_call is 'put' or 'call'

        self.strike   = strike
        self.rate     = rate
        self.tau      = tau
        self.sigma    = sigma
        self.x_low    = bounds[0]
        self.x_up     = bounds[1]
        self.barrier  = barrier
        self.put_call = put_call
        self.knock    = knock

    def _is_zero(self, spot):
        """Compute upper boundary conditions for time"""
        if self.knock == 'up-out' and spot >= self.barrier:
            return True
        elif self.knock == 'down-out' and spot <= self.barrier:
            return True
        elif self.knock == 'up-in' and spot < self.barrier:
            return True
        elif self.knock == 'down-in' and spot > self.barrier:
            return True
        else: 
            return False

    def payoff(self, spot: float):
        if self._is_zero(spot):
            return 0
        elif self.put_call == 'put':
            return max(self.strike - spot, 0)
        else:
            return max(spot - self.strike, 0)


    def a(self, t: float, spot: float): 
        """Compute coefficient for second derivative"""
        return - 0.5 * (spot*self.sigma)**2

    def b(self, t: float, spot: float): 
        """Compute coefficient for first derivative"""
        return - self.rate * spot

    def c(self, t: float, spot: float): 
        """Compute coefficient for value"""
        return self.rate

    def d(self, t: float, spot: float):
        return 0

    def boundary_condition_tau(self, spot: float):
        """Compute upper boundary conditions for time"""
        if self._is_zero(spot):
            return 0
        elif self.put_call == 'put':
            return max(self.strike - spot, 0)
        else:
            return max(spot - self.strike, 0)

    def boundary_condition_x_low(self, t: float):
        """Boundary condition for the lower bound of the spot price.""" 
        if self.put_call == 'put':
            return self.strike * np.exp(-self.rate * (self.tau - t))
        else:
            return 0.0

    def boundary_condition_x_up(self, t: float):
        """Boundary condition for the upper bound of the spot price.""" 
        if self.knock == 'up-out':
            return 0.0
        elif self.knock == 'down-out':
            return self.strike * np.exp(-self.rate * (self.tau - t))
        elif self.knock == 'up-in':
            return self.x_up - self.strike * np.exp(-self.rate * (self.tau - t))
        elif self.knock == 'down-in':
            return 0.0