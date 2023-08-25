class PDESolver:
    def __init__(self, pde, imax, jmax):
        """ Constructor
        Parameters:
            pde : PDE to be solved
            imax : last value of the first variable's discretisation
            jmax : last value of the second variable's discretisation
        """
        # Initialising self variables
        self.pde  = pde
        self.imax = imax
        self.jmax = jmax

        # Initialising dt and dx 
        self.dt = pde.t_up / imax
        self.dx = (pde.x_up - pde.x_low) / jmax

        # Initialising grid
        self.grid = np.empty((imax + 1, jmax + 1), dtype = float)

    def t(self, i):
        """ Return the descretised value of t at index i """
        return self.dt * i
    
    def x(self, j):
        """ Return the descretised value of x at index j """
        return self.dx * j

    # Helper umbrella function to get coefficients
    def a(self, i, j): return self.pde.a(self.t(i), self.x(j))
    def b(self, i, j): return self.pde.b(self.t(i), self.x(j))
    def c(self, i, j): return self.pde.c(self.t(i), self.x(j))
    def d(self, i, j): return self.pde.d(self.t(i), self.x(j))
    
    # Helper umbrella function to get boundary conditions
    def t_up(self, j): return self.pde.bound_cond_tup(self.x(j))
    def x_low(self, i): return self.pde.bound_cond_x_low(self.t(i))
    def x_up(self, i): return self.pde.bound_cond_x_up(self.t(i))

    def interpolate(self, t, x):
        """ Get interpolated solution value at given time and space
        Parameters:
            t : point in time  
            x : point in space
        Return
            interpolated solution value
        
        NOTE:
        To interpolate and find value at point p (t, x), we must first find the discrete points
        i and j. Afterwards calculate the vertical and horizontal distance between the points;
               vertical distance = (t - i*dt)
               horizontal distance = (x - j*dx)
        Compute weighted average afterwards.


        (i+1, j) --------------- (i+1, j+1)
               |            |    |
               | ---------- p -- |
               |            |    |
               |            |    |
               |            |    |
               |            |    |
           (i,j) --------------- (i, j+1)
        """
        # Step 1: calculating closest discrete point (i, j), using floor division "//"
        i = int(t // self.dt) 
        j = int((x - self.pde.x_low) // self.dx)

        # Step 2: calculating vertical and horizontal distance/weight from each node
        #         (Since we are calculating weighting rather than physical distance
        #          we will divide it by dt or dx)
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
    """ PDE solver using the explicit scheme"""
    def __init__(self, pde, imax, jmax):
        super().__init__(pde, imax, jmax)

    # Functions for calculating coefficients
    """ Coefficient A_{i,j} for explicit scheme """
    def A(self, i, j):
        """ Coefficient A_{i,j} for explicit scheme """
        return (self.dt / self.dx) * ((self.b(i, j) / 2) - (self.a(i, j) / self.dx))
    
    def B(self, i, j):
        """ Coefficient B_{i,j} for explicit scheme """
        return 1 - self.dt * self.c(i, j) + 2 * (self.dt * self.a(i, j) / (self.dx ** 2))
    
    def C(self, i, j):
        """ Coefficient C_{i,j} for explicit scheme """
        return - (self.dt / self.dx) * ((self.b(i, j) / 2) + (self.a(i, j) / self.dx)) 
    
    def D(self, i, j):
        """ Coefficient D_{i,j} for explicit scheme """
        return - self.dt * self.d(i, j)
    

    def solve_grid(self):
        """ Method for solving the mesh grid"""
        # 1. Compute all grid points for the last row
        self.grid[self.imax, :] = [self.t_up(j) for j in range(self.jmax + 1)]

        # 2. Iterate for all i from self.imax to 0 inclusive and update grid points:
        for i in range(self.imax, 0, -1):

            # 2.1 (i, 0) boundary conditions for x_low
            self.grid[i - 1, 0] = self.x_low(i - 1)
            # 2.2 (i, -1) boundary conditions for x_up
            self.grid[i - 1, -1] = self.x_up(i - 1)

            # 2.3 v_(i-1, j) formula
            #     Keep in mind, this formula is applied to every row EXCEPT for the
            #     top and bottom row which was already defined by boundary conditions
            self.grid[i-1, 1:self.jmax] = [(self.A(i, j) * self.grid[i, j-1] 
                                            + self.B(i, j) * self.grid[i, j]
                                            + self.C(i, j) * self.grid[i, j+1]
                                            + self.D(i, j)) for j in range(1, self.jmax)]