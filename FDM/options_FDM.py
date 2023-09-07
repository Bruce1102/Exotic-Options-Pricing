import numpy as np
import math

class OptionsBlackScholes:
    """Class containing data and methods to describe the Black-Scholes PDE for European options"""

    def __init__(self, strike: float, tau: float, rate: float, sigma, dividend: float, bounds: (int, int), call_put: str = 'call'):
        """ Constructor
        : param rate   : underlying asset's risk free interest rate
        : param sigma  : underlying asset's standard deviation
        : param strike : opption's strike price
        : param tau   : option's time to expriry
        : param x_low  : lower bound for the option's spot price
        : param x_up   : upper bound for the option's spot price
        """
        self.strike = strike
        self.tau = tau
        self.rate = rate
        self.sigma = sigma
        self.dividend = dividend
        self.x_low = bounds[0]
        self.x_up = bounds[1]
        self.call_put = call_put
        self.adjust = False

    def coeff_a(self, t: float, x: float) -> float:
        """Return Black Schole's coefficient 'a'."""
        return - (x * self.sigma(t, x)) ** 2 / 2

    def coeff_b(self, t: float, x: float) -> float:
        """Return Black Schole's coefficient 'b'."""
        return (self.dividend - self.rate) * x

    def coeff_c(self, t: float, x: float) -> float:
        """Return Black Schole's coefficient 'c'."""
        return self.rate

    def coeff_d(self, t: float, x: float) -> float:
        """Return Black Schole's coefficient 'd'."""
        return 0

    def boundary_condition_tau(self,  spot):
        """Compute upper boundary conditions for time"""
        return max(spot - self.strike, 0) if self.call_put == 'call' else max(self.strike - spot, 0)

    def boundary_condition_x_low(self, t):
        """Compute boundary conditions for x_low"""
        return 0.0 if self.call_put == 'call' else math.exp(-self.rate * (self.tau - t)) * self.strike

    def boundary_condition_x_up(self, t):
        """Compute boundary conditions for x_up"""
        return math.exp(-self.rate * (self.tau - t)) * self.strike if  self.call_put == 'call' else 0.0

    def adjust_point(self, grid_row, dx, tau_remaining):
        return grid


class AmericanOptionsBlackScholes(OptionsBlackScholes):
    """Class containing data and methods to describe the Black-Scholes PDE for American options"""
    def __init__(self, strike: float, tau: float, rate: float, sigma, dividend: float, bounds: (int, int), call_put: str = 'call'):
        super().__init__(strike, tau, rate, sigma, dividend, bounds, call_put)
        self.adjust = True

    def boundary_condition_x_low(self, t):
        """Compute boundary conditions for x_low"""
        return 0.0 if self.call_put == 'call' else math.exp(-self.rate * (self.tau - t)) * (self.strike - self.x_low)
    
    def boundary_condition_x_up(self, t):
        """Compute boundary conditions for x_up"""
        return self.x_up - math.exp(-self.rate * (self.tau - t)) * self.strike if self.call_put == 'call' else 0.0

    def adjust_point(self, option_value: float, x: float) -> float:
        """Check for early exercise for American options and adjust the option value if necessary."""
        intrinsic_value = max(self.strike - x, 0) if self.call_put == 'put' else max(x - self.strike, 0)
        return max(option_value, intrinsic_value)



class KnockOutOptionsBlackScholes(OptionsBlackScholes):
    def __init__(self, strike: float, tau: float, rate: float, sigma, dividend: float, barrier:float, knock:str, bounds: (int, int), call_put: str = 'call', rebate: float = 0):
        super().__init__(strike, tau, rate, sigma, dividend, bounds, call_put)
        self.barrier = barrier
        self.knock = knock
        self.rebate = rebate
        self.adjust = True

    def _check_knocked(self, spot: float) -> float:
        """Check if given the spot price, the option is knocked in or out"""
        if (self.knock == 'up' and spot >= self.barrier) or (self.knock == 'down' and spot <= self.barrier):
            return True    

    def boundary_condition_tau(self, spot):
        """Return rebate if it has been knocked out, else return regular boundary tau conditions"""
        return self.rebate if self._check_knocked(spot) else super().boundary_condition_tau(spot)

    def boundary_condition_x_low(self, tau):
        """Return rebate if it has been knocked out, else return regular boundary x_low conditions"""
        return self.rebate if self.knock == 'down' else super().boundary_condition_x_low(tau)

    def boundary_condition_x_up(self, tau):
        """Return rebate if it has been knocked out, else return regular boundary x_up conditions"""
        return self.rebate if self.knock == 'up' else super().boundary_condition_x_up(tau)

    def adjust_point(self, option_value: float, x: float) -> float:
        """Return rebate if knocked out, else return original value"""
        return self.rebate if self._check_knocked(x) else option_value
            

        
