import numpy as np
import math

class StochasticProcessSimulation:
    def __init__(self, initial_price:float, mu:float, sigma:float, tau:float, n_steps:int = 1_000):
        """ Initialize the StochasticProcessSimulation.

        Parameters:
        - initial_price : Initial asset price.
        - mu            : Drift term or mean term
        - sigma         : diffusion term or volatility of the asset.
        - lambda_jump   : Intensity of the jump.
        - tau           : duration of simulation (in years)
        - n_steps       : number of time step to simulate.
        """
        self.initial_price = initial_price
        self.mu = mu
        self.sigma = sigma
        self.tau = tau
        self.dt = tau / n_steps
        self.n_steps = n_steps
        self.simulation = np.zeros(n_steps) # Initialising price simulation
        self.simulation[0] = self.initial_price

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
        for i in range(1, self.n_steps):
            self.simulation[i] = self.simulation[i-1] + self.drift_diffusion(self.simulation[i-1])

    @property
    def get_simulation(self) -> np.array:
        """Return the simulated price dynamics."""
        return self.simulation


class VasicekProcess(StochasticProcessSimulation): 
    def __init__(self, initial_price:float, k: float, theta: float, mu:float, sigma:float, tau:float, n_steps:int = 1_000):
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
        super().__init__(initial_price, mu, sigma, tau, n_steps)
        self.k = k
        self.theta = theta

    def _drift(self, price:float) -> float:
        """Override the drift term for Vasicek Process."""
        return self.k * (self.theta - self.mu) * price * self.dt


class CIRProcess(VasicekProcess): 
    def __init__(self, initial_price:float, k: float, theta: float, mu:float, sigma:float, tau:float, n_steps:int = 1_000):
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
        super().__init__(initial_price, k, theta, mu, sigma, tau, n_steps)

    def _diffusion(self, price:float) -> float:
        """Override the diffusion term for CIR Process."""
        return self.sigma * np.sqrt(price) * np.sqrt(self.dt) * np.random.normal()

class StochasticVolatility(StochasticProcessSimulation):
    def __init__(self, initial_price:float, initial_volatility:float, mu:float, sigma:CIRProcess, tau:float, n_steps:int = 1_000):
        """
        Initialize the Stochastic Volatility model.

        Parameters:
        - initial_price      : Initial asset price.
        - initial_volatility : Initial volatility value.
        - mu                 : Drift term for the asset.
        - sigma              : CIR Process modeling the stochastic volatility.
        - tau                : Duration of simulation (in years).
        - n_steps            : Number of time steps to simulate.
        """
        super().__init__(initial_price, mu, None, tau, n_steps) # Will use our own sigma process
        self.initial_volatility = initial_volatility
        self.sigma = sigma
        self.volatilities = np.zeros(n_steps)
        self.volatilities[0] = initial_volatility

    def _volatility_diffusion(self, volatility:float):
        """Compute v_t for volatility through the CIR Process."""
        return self.sigma.drift_diffusion(volatility)

    def drift_diffusion(self, price:float, volatility:float) -> float:
        """Compute the combined drift and diffusion for the given price."""
        return self._drift(price) + price * np.sqrt(self._diffusion(price, volatility))

    def simulate(self):
        """Simulate the price dynamics using the drift-diffusion model."""
        for i in range(1, self.n_steps):
            # Update stock price using the drift-diffusion model
            self.simulation[i] = self.simulation[i-1] + self.drift_diffusion(self.simulation[i-1], self.volatilities[i-1])
            # Update volatility using the full dynamics of the CIR process
            self.volatilities[i] = self._volatility_diffusion(self.volatilities[i-1])


class Jump:
    def __init__(self, lambda_jump: float, mu_jump: float, sigma_jump: float):
        """
        Initialize the Kou's Jump Process.

        Parameters:
        - lambda_jump : Expected number of jumps per year.
        - mu_jump     : Expected return of the jump.
        - sigma_jump  : Volatility of the jump.
        """
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump

    def jump_sizes(self, interval_length: float) -> np.array:
        """Compute the sizes of the jumps over the given interval."""
        num_jumps = np.random.poisson(self.lambda_jump * interval_length)
        return np.random.normal(self.mu_jump, self.sigma_jump, num_jumps)

    def apply_jumps(self, price: float, interval_length: float) -> float:
        """Apply the jumps to the given price over the interval."""
        jumps = self.jump_sizes(interval_length)
        for jump in jumps:
            price *= np.exp(jump)
        return price


class KouJump(Jump):
    def __init__(self, lambda_jump: float, p: float, lambda_positive: float, lambda_negative: float):
        """
        Initialize the Kou's Jump Process.

        Parameters:
        - lambda_jump    : Intensity of the jump.
        - p              : Probability of positive jump.
        - lambda_positive: Parameter for positive exponential distribution.
        - lambda_negative: Parameter for negative exponential distribution.
        """
        super().__init__(lambda_jump, None)  # No size_jump for Kou's model in the parent class
        self.p = p
        self.lambda_positive = lambda_positive
        self.lambda_negative = lambda_negative

    def jump_sizes(self, interval_length: float) -> np.array:
        """Compute the sizes of the jumps over the given interval using Kou's model."""
        num_jumps = np.random.poisson(self.lambda_jump * interval_length)
        jump_sizes = np.zeros(num_jumps)
        
        for i in range(num_jumps):
            if np.random.uniform() < self.p:
                jump_sizes[i] = np.random.exponential(scale=1/self.lambda_positive)
            else:
                jump_sizes[i] = -np.random.exponential(scale=1/self.lambda_negative)
        
        return jump_sizes


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

    def drift_diffusion_jump(self, price: float) -> float:
        """Compute the combined drift, diffusion, and jump for the given price."""
        price_with_drift_diffusion = self.stochastic_process.drift_diffusion(price)
        return self.jump_process.apply_jumps(price_with_drift_diffusion, self.dt)

    def simulate(self):
        """Simulate the price dynamics with jumps."""
        for i in range(1, self.n_steps):
            self.simulation[i] = self.drift_diffusion_jump(self.simulation[i-1])

    def get_simulation(self) -> np.array:
        """Return the simulated price dynamics."""
        return self.simulation