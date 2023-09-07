import numpy as np
from scipy.linalg import solve_banded
import math
from FDM.options_FDM import *

class PDESolver:
    """ Abstract class to solve Black-Scholes PDE
    """

    def __init__(self, pde, imax, jmax):
        """ Constructor
        : param pde : PDE to be solved
        : param imax : last value of the first variable's discretisation
        : param jmax : last value of the second variable's discretisation
        """
        # Initialising self variables
        self.pde  = pde
        self.imax = imax
        self.jmax = jmax
        
        # Initialising dt and dx 
        self.dt = pde.tau / imax
        self.dx = (pde.x_up - pde.x_low) / jmax

        # Initialising grid
        self.grid = np.empty((imax + 1, jmax + 1), dtype = float)

    def check_stability(self, time, spot):
        """Checking if the ratio of dt dx is stable for explicit scheme"""
        sigma_value = self.pde.sigma(time, spot)
        max_price = self.pde.x_up

        max_dt = (self.dx**2) / (2 * sigma_value**2 * max_price**2)

        if self.dt > max_dt:
            print(f"Current dt of {self.dt} is unstable. Reduce it to below {max_dt}.")
        else:
            print(f"Current dt of {self.dt} is stable.") 

    def t(self, i):
        """ Return the descretised value of t at index i """
        return self.dt * i
    
    def x(self, j):
        """ Return the descretised value of x at index j """
        return self.dx * j

    # Helper umbrella function to get coefficients
    def coeff_a(self, i, j):
        """Umbrella function for Black Scholes coefficient 'a'."""
        return self.pde.coeff_a(self.t(i), self.x(j))

    def coeff_b(self, i, j):
        """Umbrella function for Black Scholes coefficient 'b'."""
        return self.pde.coeff_b(self.t(i), self.x(j))

    def coeff_c(self, i, j):
        """Umbrella function for Black Scholes coefficient 'c'."""
        return self.pde.coeff_c(self.t(i), self.x(j))

    def coeff_d(self, i, j):
        """Umbrella function for Black Scholes coefficient 'd'."""
        return self.pde.coeff_d(self.t(i), self.x(j))

    def boundary_condition_tau(self, j):
        """Umbrella function for boundary condition tau (time to maturity)."""
        return self.pde.boundary_condition_tau(self.x(j))

    def boundary_condition_x_low(self, i):
        """Umbrella function for boundary condition x low."""
        return self.pde.boundary_condition_x_low(self.t(i))
    
    def boundary_condition_x_up(self, i):
        """Umbrella function for boundary condition x up."""
        return self.pde.boundary_condition_x_up(self.t(i))

    def interpolate(self, t, x):
        """ Get interpolated solution value at given time and space
        : param t : point in time
        : param x : point in space
        return    : interpolated solution value
        """
        # Compute closest discrete point (i, j), using floor division "//"
        i = int(t // self.dt) 
        j = int((x - self.pde.x_low) // self.dx)

        # Compute vertical and horizontal distance/weight from each node
        i_1 = (t - self.dt * i) / self.dt
        i_0 = 1 - i_1
        j_1 = (x + self.pde.x_low - self.dx * j) / self.dx
        j_0 = 1 - j_1

        # Step 3: Returning the weighted average
        return (i_0 * j_0 * self.grid[i,j] 
                + i_1 * j_0 * self.grid[i+1, j]
                + i_0 * j_1 * self.grid[i, j+1] 
                + i_1 * j_1 * self.grid[i+1, j+1])

class ExplicitScheme(PDESolver):
    """ Black Scholes PDE solver using the explicit scheme
    """
    def __init__(self, pde, imax, jmax):
        super().__init__(pde, imax, jmax)

    def _A(self, i, j):
        """Coefficient 'A' for explicit scheme."""
        return (self.dt / self.dx) * ((self.pde.coeff_b(i, j) / 2) - (self.pde.coeff_a(i, j) / self.dx))
    
    def _B(self, i, j):
        """Coefficient 'B' for explicit scheme."""
        return 1 - self.dt * self.pde.coeff_c(i, j) + 2 * (self.dt * self.pde.coeff_a(i, j) / (self.dx ** 2))
    
    def _C(self, i, j):
        """Coefficient 'C' for explicit scheme."""
        return - (self.dt / self.dx) * ((self.pde.coeff_b(i, j) / 2) + (self.pde.coeff_a(i, j) / self.dx)) 
    
    def _D(self, i, j):
        """Coefficient 'D' for explicit scheme."""
        return - self.dt * self.pde.coeff_d(i, j)
    

    def solve_grid(self):
        # Compute all grid points for the last row
        self.grid[self.imax, :] = [self.boundary_condition_tau(j) for j in range(self.jmax + 1)]

        # Iterate for all i from self.imax to 0 inclusive and update grid points:
        for i in range(self.imax, 0, -1):
            # (i, 0) boundary conditions for x_low
            self.grid[i - 1, 0] = self.boundary_condition_x_low(i - 1)
            # (i, -1) boundary conditions for x_up
            self.grid[i - 1, -1] = self.boundary_condition_x_up(i - 1)

            # v_(i-1, j) formula
            for j in range(1, self.jmax):
                self.grid[i-1, j] = (self._A(i, j) * self.grid[i, j-1] 
                                    + self._B(i, j) * self.grid[i, j]
                                    + self._C(i, j) * self.grid[i, j+1]
                                    + self._D(i, j))
                
                # Check for early exercise for American options
                
                if self.pde.adjust:
                    x = self.x(j)
                    self.grid[i-1, j] = self.pde.adjust_point(self.grid[i-1, j], x)


from scipy import sparse

class ImplicitScheme(PDESolver):
    """ Black Scholes PDE solver using the implicit scheme
    """
    def __init__(self, pde, imax, jmax):
        super().__init__(pde, imax, jmax)

    # Functions for calculating coefficients
    """ Coefficient {*insert coefficient letter}_{i,j} for Implicit scheme
        : param i : index of x discretisation
        : param j : index of t discretisation
        """
    def A(self, i, j): return 0
    def B(self, i, j): return 1
    def C(self, i, j): return 0
    def D(self, i, j): return - self.dt * self.coeff_d(i-1, j)
    def E(self, i, j): return - (self.dt / self.dx) * ((self.coeff_b(i-1, j) / 2) - (self.coeff_a(i-1, j) / self.dx))
    def F(self, i, j): return 1 + self.dt * self.coeff_c(i-1, j) - (2 * self.dt * self.coeff_a(i-1, j)) / (self.dx ** 2)
    def G(self, i, j): return (self.dt / self.dx) * ((self.coeff_b(i-1, j) / 2) + (self.coeff_a(i-1, j) / self.dx))

    def get_W(self, i):
        """
        Compute the intermediate vector w_i used to compute the right-hand-side of 
        the linear system of equations in the implicit scheme.
        : param i : index of x discretisation
        return    : a numpy array of [w_1, ....., w_{jmax-1}] 
        """

        # Step 1: Initialise first row of elements
        W = [self.D(i,1) + self.A(i, 1) * self.boundary_condition_x_low(i) - self.E(i, 1)*self.boundary_condition_x_low(i-1)]

        # Step 2: add middle rows (D_{i, x}'s)
        W += [self.D(i, j) for j in range(2, self.jmax - 1)]

        # Step 3: add final row
        W += [self.D(i,self.jmax - 1) + self.C(i, self.jmax - 1) * self.boundary_condition_x_up(i) 
              - self.G(i, self.jmax - 1)*self.boundary_condition_x_up(self.jmax - 1)]

        return W
    
    def compute_vi(self, i):
        """
        Compute the v_{i-1} vector solving the inverse problem,
        Uses the precomputed values of self.grid[i, :] to compute the self.grid[i-1,:]
        : param i : i-th iteration
        return    : The v_[i-1] vector, nunmpy array of length self.jmax+1
        
        """
        # initialise A diagonal matrix:
        A_diag_left  = [self.A(i, j) for j in range(2, self.jmax)]
        A_diag_cent  = [self.B(i, j) for j in range(1, self.jmax)]
        A_diag_right = [self.C(i, j) for j in range(1, self.jmax-1)]
        A = sparse.diags([A_diag_left, A_diag_cent, A_diag_right], [-1, 0, 1])

        # initialise B diagonal matrix:
        B_diag_left  = [self.E(i, j) for j in range(2, self.jmax)]
        B_diag_cent  = [self.F(i, j) for j in range(1, self.jmax)]
        B_diag_right = [self.G(i, j) for j in range(1, self.jmax-1)]
        B = sparse.diags([B_diag_left, B_diag_cent, B_diag_right], [-1, 0, 1])

        # Computing the righ hand side of the equation (the matricies in the brackets)
        rhs = A @ self.grid[i, 1:-1 ] + self.get_W(i)
        return sparse.linalg.splu(B).solve(rhs)
    def solve_grid(self):
        """
        Iteratively solve the PDE for the entire grid with the "compute_vi" function
        """
        # Step 1: initialise the last row of the grid using boundary conditions on 't'
        self.grid[self.imax, :] = [self.boundary_condition_tau(j) for j in range(self.jmax + 1)]

        # Step 2: iteratively compute the next v_i
        for i in range(self.imax, 0, -1):
            # Step 2.1: Set elements on row 'i-1' on row edges using boundary conditions
            self.grid[i-1, 0] = self.boundary_condition_x_low(i - 1)
            self.grid[i-1, self.jmax] = self.boundary_condition_x_up(i-1)

            # Step 2.2: Set middle rows of column 'i-1' using "compute_vi" function
            self.grid[i-1, 1:-1] = self.compute_vi(i)

            if self.pde.adjust:
                for j in range(1, self.jmax):
                    x = self.x(j)
                    self.grid[i-1, j] = self.pde.adjust_point(self.grid[i-1, j], x)

class CrankNicolsonScheme(ImplicitScheme):
    """ Black Scholes PDE solver using the Crank Nicolson scheme"""
    def __init__(self, pde, imax, jmax):
        super().__init__(pde, imax, jmax)

    def A(self, i, j): 
        """ Coefficient A for Crank Nicolson Scheme"""
        explicit_a = (self.dt / self.dx) * ((self.pde.coeff_b(i, j) / 2) - (self.pde.coeff_a(i, j) / self.dx))
        implicit_a = 0
        return 0.5 * (explicit_a + implicit_a)
    def B(self, i, j):
        """ Coefficient B for Crank Nicolson Scheme"""
        explicit_b = 1 - self.dt * self.pde.coeff_c(i, j) + 2 * (self.dt * self.pde.coeff_a(i, j) / (self.dx ** 2))
        implicit_b = 1
        return 0.5 * (explicit_b + implicit_b)

    def C(self, i, j): 
        """ Coefficient C for Crank Nicolson Scheme"""
        explicit_c = - (self.dt / self.dx) * ((self.pde.coeff_b(i, j) / 2) + (self.pde.coeff_a(i, j) / self.dx)) 
        implicit_c = 0
        return 0.5 * (explicit_c + implicit_c)

    def D(self, i, j):
        """ Coefficient D for Crank Nicolson Scheme"""
        explicit_d = - self.dt * self.pde.coeff_d(i, j)
        implicit_d = - self.dt * self.coeff_d(i-1, j)
        return 0.5 * (explicit_d + implicit_d)

