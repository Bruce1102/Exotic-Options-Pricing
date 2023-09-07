import numpy as np
import math
from scipy.stats import ncx2
from scipy.optimize import minimize
import statsmodels.api as sm

class StochasticProcessSimulation:
    def __init__(self, initial_price: float=100, mu: float=0.05, sigma: float=0.1, tau: float=1, 
                 dividend: float=0, n_steps:int = 1_000):
        """ Initialize the StochasticProcessSimulation.

        Parameters:
        - initial_price : Initial asset price.
        - mu            : Drift term or mean term
        - sigma         : diffusion term or volatility of the asset.
        - lambda_jump   : Intensity of the jump.
        - tau           : duration of simulation (in years)
        - n_steps       : number of time step to simulate.
        """
        self.initial_price     = initial_price
        self.discrete_dividend = dividend * tau / n_steps
        self.mu                = mu - self.discrete_dividend
        self.sigma             = sigma
        self.tau               = tau
        self.dt                = tau / n_steps
        self.n_steps           = n_steps
        self.simulation        = np.zeros(n_steps) # Initialising price simulation
        self.params            = [self.mu, self.sigma]
        self.bounds            = [(None, None), (0, 1)]

    def negative_log_likelihood(self, params, data, dt):
        """Negative log-likelihood computation specific to GBM."""
        mu, sigma = params
        n = len(data) - 1  # Number of log-returns
        
        log_returns = np.log(data[1:] / data[:-1])# Compute the log-returns
        
        # Compute the log-likelihood
        log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma**2 * dt) - 0.5 * np.sum((log_returns - mu*dt)**2 / (sigma**2 * dt))
        
        return -log_likelihood

    def set_params(self, params):
        """Setting given parameters to itself"""
        self.mu = params[0]
        self.sigma = params[1]

    def fit(self, data, dt):
        """Universal fit method."""
        self.initial_price = data[0]

        result = minimize(self.negative_log_likelihood, self.params, args=(data, dt), bounds=self.bounds, method='Nelder-Mead')
        
        if result.success:
            self.set_params(result.x)
            return self, result.x
        else:
            raise ValueError("MLE optimization did not converge.")


    def _drift(self, price:float) -> float:
        """Compute the drift term for the given price."""
        return self.mu * price * self.dt

    def _diffusion(self, price:float) -> float:
        """Compute the diffusion term for the given price."""
        return self.sigma * price * np.sqrt(self.dt) * np.random.normal()

    def drift_diffusion(self, price:float) -> float:
        """Compute the combined drift and diffusion for the given price."""
        return self._drift(price) + self._diffusion(price)

    def simulate(self):
        """Simulate the price dynamics using the drift-diffusion model."""
        self.simulation[0] = self.initial_price
        for i in range(1, self.n_steps):
            self.simulation[i] = self.simulation[i-1] + self.drift_diffusion(self.simulation[i-1])

    def get_simulation(self) -> np.array:
        """Return the simulated price dynamics."""
        return self.simulation


class VasicekProcess(StochasticProcessSimulation): 
    def __init__(self, initial_price:float=100, k: float=0.1, theta: float=0.05, sigma:float=0.1, 
                 tau:float=1, dividend: float=0, n_steps:int = 1_000):
        """ Initialize the Vasicek Process.

        Parameters:
        - initial_price : Initial asset price.
        - k             : Speed of mean reversion
        - theta         : Amplitude of mean reversion
        - mu            : Drift term or mean term
        - sigma         : diffusion term or volatility of the asset.
        - lambda_jump   : Intensity of the jump.
        - tau           : duration of simulation (in years)
        - n_steps       : number of time step to simulate.
        """
        super().__init__(initial_price, 0, sigma, tau, dividend, n_steps)
        self.k      = k
        self.theta  = theta
        self.params = [self.k, self.theta, self.sigma]
        self.bounds = [(None, None), (None, None), (0, 1)]
        

    def set_params(self, params):
        """Setting given parameters to itself"""
        self.k = params[0]
        self.theta = params[1]
        self.sigma = params[2]

    def negative_log_likelihood(self, params, data, dt):
        """Negative log-likelihood computation specific to Vasicek."""
        k, theta, sigma = params
        n = len(data) - 1  # Number of log-returns
        
        mu = data[:-1] + k * (theta - data[:-1]) * dt

        log_returns = np.log(data[1:] / data[:-1])# Compute the log-returns
        
        # Compute the log-likelihood
        log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma**2 * dt) - 0.5 * np.sum((log_returns - mu*dt)**2 / (sigma**2 * dt))
        
        return -log_likelihood
        

    def _drift(self, price:float) -> float:
        """Override the drift term for Vasicek Process."""
        return self.k * (self.theta - price) * self.dt

    


class CIRProcess(VasicekProcess): 
    def __init__(self, initial_price:float=100, k: float=0.1, theta: float=0.05, sigma:float=0.1, 
                 tau: float=1, dividend: float=0, n_steps:int = 1_000):
        """ Initialize the Vasicek Process.

        Parameters:
        - initial_price : Initial asset price.
        - k             : Speed of mean reversion
        - theta         : Amplitude of mean reversion
        - sigma         : diffusion term or volatility of the asset.
        - lambda_jump   : Intensity of the jump.
        - tau           : duration of simulation (in years)
        - n_steps       : number of time step to simulate.
        """
        super().__init__(initial_price, k, theta, sigma, tau, n_steps)

    def negative_log_likelihood(self, params, data, dt):
        k = params[0]
        theta = params[1]
        sigma = params[2]

        n = len(data)
        likelihoods = []
        for t in range(1, n):
            mu = k * (theta - data[t-1]) * dt
            sigma_t = sigma * np.sqrt(data[t-1] * dt)
            likelihood = (1 / (sigma_t * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((data[t] - data[t-1] - mu) / sigma_t)**2)
            likelihoods.append(np.log(likelihood))
        return -sum(likelihoods)

    def fit(self, data, dt):
        """Universal fit method."""
        self.initial_price = data[0]

        # Regularization term
        lambda_reg = 1e-5

        def regularized_nll(params, data, dt):
            k, theta, sigma = params
            nll = self.negative_log_likelihood(params, data, dt)
            return nll + lambda_reg * sigma**2

        result = minimize(regularized_nll, self.params, args=(data, dt), bounds=self.bounds, method='Nelder-Mead')

        if result.success:
            self.set_params(result.x)
            return self, result.x
        else:
            raise ValueError("MLE optimization did not converge.")


    def _diffusion(self, price:float) -> float:
        """Override the diffusion term for CIR Process."""
        return self.sigma * np.sqrt(price) * np.sqrt(self.dt) * np.random.normal()



class StochasticVolatility(StochasticProcessSimulation):
    def __init__(self, initial_price:float, volatility_process:CIRProcess, mu: float=0.01, tau:float=1, n_steps:int = 1_000):
        """
        Initialize the Stochastic Volatility model.

        Parameters:
        - initial_price      : Initial asset price.
        - mu                 : Drift term for the asset.
        - volatility_process : CIR Process modeling the stochastic volatility (Must be pre-fitted).
        - tau                : Duration of simulation (in years).
        - n_steps            : Number of time steps to simulate.
        """
        super().__init__(initial_price, mu, None, tau, n_steps) # Will use our own sigma process
        self.volatility_process = volatility_process

        volatility_process.simulate()
        self.volatilities = volatility_process.get_simulation()

        self.params = [self.mu]


    def negative_log_likelihood(self, params, data, dt):
        # Extract parameters. Now, only mu is the parameter to be estimated.
        mu = params[0]
        
        # Get the already simulated volatilities from the fitted volatility process
        volatilities = self.volatility_process.get_simulation()
        
        # Compute log-returns for stock prices
        log_returns = np.log(data[1:] / data[:-1])
        
        # Compute likelihood for stock prices given volatilities
        expected_log_returns = mu * dt
        log_likelihood = -0.5 * np.sum((log_returns - expected_log_returns)**2 / volatilities[:-1]) - 0.5 * len(log_returns) * np.log(2 * np.pi * dt)
        
        return -log_likelihood

    def set_params(self, params):
        """Setting given parameters to itself"""
        self.mu = mu

    def _volatility_diffusion(self, price, volatility):
        return price * np.sqrt(volatility) * np.sqrt(self.dt) * np.random.normal()

    def drift_diffusion(self, price:float, volatility:float) -> float:
        """Compute the combined drift and diffusion for the given price."""
        drift = self._drift(price)
        diffusion = self._volatility_diffusion(price, volatility)
        return drift + diffusion

    def simulate(self):
        """Simulate the price dynamics using the drift-diffusion model."""
        for i in range(1, self.n_steps):
            # Update stock price using the drift-diffusion model
            self.simulation[i] = self.simulation[i-1] + self.drift_diffusion(self.simulation[i-1], self.volatilities[i])



class Jump:
    def __init__(self, lambda_jump: float, mu_jump: float, sigma_jump: float, dt:float):
        """
        Initialize the Merton's Jump Process.

        Parameters:
        - lambda_jump : Expected number of jumps per year.
        - mu_jump     : Expected return of the jump.
        - sigma_jump  : Volatility of the jump.
        """
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
        self.dt = dt

        self.initial_params = [0.1, 0.05, 0.1]
        self.bounds = [(0, None), (0, None), (1e-5, 1)]

    def compute_jump(self, price:float) -> float:
        """
        Compute dJ_t over a given interval.

        Parameters:
        - interval_length: Length of the time interval.

        Returns:
        - dJ_t: Total jump over the interval.
        """
        # Determine the number of jumps over the interval
        num_jumps = np.random.poisson(self.lambda_jump * self.dt)
        
        # Determine the sizes of the jumps
        jump_sizes = np.random.normal(self.mu_jump, self.sigma_jump, num_jumps)
        
        # Compute dJ_t
        dJ_t = np.sum(np.exp(jump_sizes) - 1)
        
        return price * dJ_t

class StochasticJumpProcess:
    def __init__(self, stochastic_process, jump_process: Jump, initial_price: float, n_steps: int = 1_000):
        """Initialize the stochastic process with jumps."""
        self.stochastic_process = stochastic_process
        self.jump_process = jump_process
        self.dt = self.stochastic_process.dt
        self.n_steps = n_steps
        self.initial_price = initial_price
        self.simulation = np.zeros(n_steps)
        self.simulation[0] = self.initial_price

        if isinstance(self.stochastic_process, StochasticVolatility):
            self.stochastic_process.volatility_process.simulate()

    def negative_log_likelihood(self, params, data):
        # Split the parameters for the two processes
        params_stochastic = params[:len(self.stochastic_process.initial_params)]
        params_jump = params[len(self.stochastic_process.initial_params):]

        # Compute the expected jumps for the given parameters and data
        expected_jumps = [self.jump_process.compute_jump(price) for price in data[:-1]]

        # Adjust the data for the jumps to get the returns due to the underlying stochastic process
        adjusted_data = data[1:] - data[:-1] - expected_jumps
        adjusted_data = np.concatenate(([data[0]], adjusted_data))  # Add back the initial price

        # Compute the negative log-likelihoods for the two processes
        nll_stochastic = self.stochastic_process.negative_log_likelihood(params_stochastic, adjusted_data)
        nll_jump = self.jump_process.negative_log_likelihood(params_jump, data)

        # Return the combined negative log-likelihood
        return nll_stochastic + nll_jump

    def fit(self, data):
        """Fit the model to the given data using MLE."""
        
        # Define the combined negative log-likelihood function
        def combined_nll(params_combined):
            return self.negative_log_likelihood(params_combined, data)

        # Initial parameters: combining both stochastic process and jump process parameters
        initial_params_combined = self.stochastic_process.initial_params + self.jump_process.initial_params

        # Bounds: combining both stochastic process and jump process bounds
        bounds_combined = self.stochastic_process.bounds + self.jump_process.bounds

        # Optimize the combined negative log-likelihood
        result = minimize(combined_nll, initial_params_combined, bounds=bounds_combined)

        if result.success:
            # Update the parameters of the stochastic process and the jump process
            self.stochastic_process.set_params(result.x[:len(self.stochastic_process.initial_params)])
            self.jump_process.set_params(result.x[len(self.stochastic_process.initial_params):])
            return result.x
        else:
            raise ValueError("MLE optimization did not converge.")

    def simulate(self):
        """Simulate the price dynamics with jumps."""
        for i in range(1, self.n_steps):
            if isinstance(self.stochastic_process, StochasticVolatility):
                # If the process is StochasticVolatility, we need to pass the volatility as well
                drift_diffusion = self.stochastic_process.drift_diffusion(self.simulation[i-1], self.stochastic_process.volatilities[i-1])
            else:
                drift_diffusion = self.stochastic_process.drift_diffusion(self.simulation[i-1])

            jump = self.jump_process.compute_jump(self.simulation[i-1])

            self.simulation[i] = self.simulation[i-1] + drift_diffusion + jump

    def get_simulation(self) -> np.array:
        """Return the simulated price dynamics."""
        return self.simulation