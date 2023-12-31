<h1>Exotic Options Pricing Project</h1>
This project focuses on the pricing of exotic options using both Monte Carlo and Finite Difference Methods. The exotic options covered in this project include Asian, American, Lookback, and Barrier options.



<h2>Exotic Options:</h2>

Exotic options are financial derivatives that offer more complexity than standard options in terms of their payoff conditions and structures. Below is a list of features of exotic options


**1. American Options (Early Exercise):** 

Unlike standard European options, the holder of this option has the privilage of exercising the option at any point before maturity as they wish. For American type options, the path dependency is weak, meaning the partial differential equation to be solved has no more independent variables than a European contract

**2. Barrier Options (Weak Path Dependency):** 

These options become activated or deactivated when the price of the underlying asset crosses a certain barrier level. Depending on the specifics, they can be classified as knock-in or knock-out. This is considered as an exotic option due to the property of weak path dependency - The final payoff is influenced by the path taken by the underlying asset price. 

**3. Asian Options (Strong Path Dependency):** 

These options have their payoff determined by the average price of the underlying asset over a certain period, rather than just the final price. This averaging can either be arithmetic or geometric. Strong path dependence in financial options refers to a characteristic where the final payoff of the option is highly sensitive and contingent on the specific historical price path taken by the underlying asset. In other words, the cumulative price movements of the asset over time significantly influence the option's ultimate value.

**3. Lookback Options (Strong Path Dependency):** 

These options allow the holder to "look back" over time to determine the payoff. The payoff of a lookback option depends on the optimal value the underlying asset reached during the life of the option. Similar to Asian options, lookback options are a form of strong path dependency as the payoff of this option is solely dependent on the path taken by the underlying asset.



<h2> Pricing methods </h2>

As these exotic options have different properties, different options will be priced with different methods. American and knock-out barrier options will be priced with Finite Different Methods as they have a very similar PDE to the standard European option, the only difference would be boundary conditions and additional function to modify each point on the mesh grid. A regular Black Scholes PDE has two variables - value of underlying asset and time. Since Asian, Lookback, and knock-in barrier are strongly path dependent, they require an additional variable representing the historical price; using FDM to solve this PDE will require us to generate a 3D grid. To tackle this problem these options will be priced with a Monte-Carlo method.



<h2>Finite Difference Method (FDM):</h2>

Finite Difference Methods are numerical techniques used to approximate solutions to differential equations by discretizing them. In the context of options pricing, FDM is used to solve the Black-Scholes partial differential equation (PDE). The main idea behind FDM is to replace the derivatives in the differential equation with differences at discrete points on a grid or mesh.

**Grid/Mesh:**

The grid or mesh is a set of discrete points where the solution is approximated. In the context of options pricing:

One axis (usually the x-axis) represents the underlying asset's price. This axis spans from a minimum to a maximum possible asset price. The range is chosen to be wide enough to ensure that the option's price is effectively zero at the maximum price and the minimum price.

The other axis (usually the t-axis) represents time, starting from the current time and ending at the option's expiration time.

The grid's intersections represent discrete points where the option's price is estimated. The more points on the grid, the more accurate the approximation, but it also requires more computational resources.

<h3> FDM Schemes: </h3>
Each FDM scheme approximates the derivatives in the Black-Scholes PDE differently. The Black-Scholes PDE can be defined in the following equation:

$$\frac{\partial V}{\partial t} = a(t, x) \frac{\partial^2 V}{\partial x^2} + b_(t, x) \frac{\partial V}{\partial x} + c(t, x) V(t, x)  + d(t,x)$$

$$a(t, x) = \frac{1}{2} \sigma^2 $$

$$b(t, x) = -r x $$

$$c(t, x) = r - dividend $$

$$c(t, x) = 0 $$



**1. Explicit Scheme:** 

The explicit scheme uses the option prices at the current time level to estimate the option prices at the next time level. The equation for the explicit scheme is:

$$V_{i-1, j} = A_{i, j} V_{i, j-1} + B{i, j} V_{i, j} + C_{i, j} V_{i, j+1} + D_{i, j}$$

$$A_{i, j} = \frac{\Delta t}{\Delta x} (\frac{b_{i,j}}{2} - \frac{a_{i,j}}{\Delta x}) $$

$$B_{i, j} = 1 - \Delta t c_{i,j} + \frac{2 \Delta t a_{i, j}}{\Delta x^2}$$

$$C(t, x) = - \frac{\Delta t}{\Delta x} (\frac{b_{i,j}}{2} - \frac{a_{i,j}}{\Delta x})$$

$$D(t, x) = - \Delta t d_{i, j} $$

Where $V_{i,j}$ represents the option price at $i^{th}$ time level and $j^{th}$ asset price level.

**2. Implicit Scheme:** 

Unlike the explicit scheme, this method uses future values in its calculations, making it unconditionally stable. However, it often requires solving a system of linear equations, which can be computationally intensive.

$$V_{i+1, j} = A_{i, j} V_{i, j+1} + B{i, j} V_{i, j} + C_{i, j} V_{i, j-1} + D_{i, j}$$

$$A_{i,j} = - \frac{\delta t}{\delta x} (\frac{b_{i,j}}{2} + \frac{a_{i,j}}{\delta x^2})$$

$$B_{i,j} = 1 + \delta t c_{i,j} - \frac{2 \delta t a_{i,j}}{\delta x^2}$$

$$C_{i,j} = \frac{\delta t}{\delta x} (\frac{b_{i,j}}{2} + \frac{a_{i,j}}{\delta x^2})$$

$$D_{i,j} = \delta t d_{i,j}$$

**3. Crank-Nicolson Scheme:** 

A blend of explicit and implicit schemes, it offers a balance between accuracy and stability.

$$V_{i-j, j} = \frac{1}{2}(\text{Explicit Scheme + Implicit Scheme})$$

This method provides a more accurate approximation than either the explicit or implicit schemes alone, especially for problems where the solution exhibits rapid changes.

<h2>Monte Carlo Method:</h2>

In this code, the Monte-Carlo pricing method includes 3 parts, stochastic process modelling, options initialisation and finally running the Monte-Carlo simulation. Stochastic process modelling is responsible for modelling the underlying asset's price dynamics. Options initialisation is for creating an options contract based on the underlying asset's stochastic process. The Monte-Carlo simulation will be responsible for simulating the random paths and computing the fair value of the option contract today.

<h3>Stochastic Process Modeling:</h3>

The asset price is modeled using various stochastic processes. The project incorporates an object-oriented approach to represent these processes. The processes used include:

**1. Geometric Brownian Motion (GBM):**

$$d S_t = \mu S_t dt + \sigma S_t dW_t$$
- GBM is a continuous-time stochastic process where the logarithm of the randomly varying quantity follows a Brownian motion. It's widely used in finance for modeling stock prices.


**2. Vasicek Model:**
$$d r_t = k (\theta - r_t) dt + \sigma dW_t$$
- The Vasicek model describes the evolution of interest rates. It's a mean-reverting model where rates tend to drift towards a long-term mean.

**3. Cox-Ingersoll-Ross (CIR) Model**
$$d r_t = k (\theta - r_t) dt + \sigma \sqrt{r_t } dW_t$$
- CIR is an extension of the Vasicek model, ensuring that interest rates remain non-negative.

**4. Heston Model:**
$$d S_t = \mu S_t dt + \sqrt({v_t} S_t d W_{t1}$$
$$d v_t = k (\theta - v_t) dt + \xi \sqrt{v_t } dW_{t2}$$
- The Heston model describes the evolution of stock prices and their volatility. It's known for capturing volatility smiles and skews in the market.

**5. Merton's Jump Diffusion:**

Jump processes are incorporated into stochastic models to account for sudden and significant changes in the price of the underlying asset. These jumps can be due to various unexpected events, such as political upheavals, major economic announcements, or other macro events that can cause abrupt market movements. By introducing a jump component, the model becomes better equipped to capture the real-world discontinuities in asset prices. Below is an example of a Merton's jump process added into a geometric brownian motion.

$$d S_t = \mu S_t dt + \sigma S_t dW_t + S_t J dq_t$$

$$dq_t = 
\begin{cases} 
1 & \text{with probability } \lambda dt \\
0 & \text{with probability } 1 - \lambda dt 
\end{cases}$$

Where:
- $J$: Jump size, between 0 to 1
- $dq_t$: Compound Poisson process


<h3>Option Classes:</h3> 
For each type of exotic option, a dedicated class is defined. These classes provide functionalities to simulate the asset's behavior and compute its payoff.

- **1. Asian Option:**

Simulate price of underlying asset, compute the average price of the path taken. Compare with the spot price or the provided strike price (depending whether it is a floating or fixed Asian option)

- **2. Barrier Option:**

The 'BarrierOption' class contains an boolean attribute 'knocked_in' or 'knocked_out'. The purpose of this attribute is to indicate whether the option has surpassed the barrier hence causing the option to become valueless.


- **3. Lookback Option:**

The 'LookbackOption' class contains a attribute 'hist_low' and 'hist_high' which represents the lowest and highest price in the simulated prices. These values will be used to determine the payoff of the option.


<h3>Monte Carlo Simulation:</h3>

The Monte Carlo method is a powerful computational technique used in a wide range of fields, including finance, physics, engineering, and more. It provides a versatile approach to solving problems that involve randomness, uncertainty, or complex systems.

The Monte Carlo simulation will be ran by following these steps:

**1. Random Sampling:** Monte Carlo simulations generate a large number of random price paths for the underlying asset based on the chosen stochastic process. These price paths simulate the potential future trajectories of the asset's price.

**2. Option Payoff Calculation:** For each generated price path, the payoff of the exotic option is computed. This step involves applying the specific conditions and terms of the exotic option to determine its value at the end of each simulation.

**3. Statistical Analysis:** After running a significant number of simulations, statistical analysis is performed on the calculated payoffs. This analysis provides insights into the distribution of possible option values, including measures like the mean (expected value) and standard deviation (volatility).

**4. Discounting:** To obtain the present value of the option, the calculated payoffs are discounted back to the present time using the risk-free rate. This step considers the time value of money.

**5. Option Price Estimation:** The final step involves aggregating the present values of the option payoffs to estimate the option's fair market price. This Monte Carlo estimate reflects the expected value of the exotic option under the assumed stochastic process.
