import numpy as np
from scipy.integrate import solve_bvp


def sturm_liouville_solver(p, pd, q, w, a, b, alpha, beta, N):
    """
    Solve the Sturm-Liouville problem using numerical methods.

    Parameters:
        p : callable
            Function defining the coefficient p(x) in the differential equation.
        pd : callable
            Function defining the coefficient dp(x)/dx in the differential equation.
        q : callable
            Function defining the coefficient q(x) in the differential equation.
        w : callable
            Function defining the coefficient w(x) in the differential equation.
        a : float
            Left boundary of the interval.
        b : float
            Right boundary of the interval.
        alpha : float
            Boundary condition at x=a.
        beta : float
            Boundary condition at x=b.
        N : int
            Number of grid points for numerical integration.

    Returns:
        sol : tuple
            A tuple containing the solution (x, u) where x is the grid points and u is the solution.
    """

    def fun(x, y, lmbda):
        return np.vstack((y[1], (-q(x) * y[0] + w(x) * y[0] * lmbda) / p(x)-(pd(x) * y[1])/ p(x)))

    def bc(ya, yb, lmbda):
        k = p[0]
        return np.array([ya[0], yb[0] , ya[1]-k])

    x = np.linspace(a, b, N)
    y_init = np.zeros((2, x.size))
    lmbda_guess = 1.0  # Initial guess for eigenvalue

    # Solve the boundary value problem for various initial guesses of eigenvalue
    for i in range(10):  # Iterate to refine the eigenvalue guess
        sol = solve_bvp(fun, bc, x, y_init, p=[lmbda_guess])
        if sol.success:
            break
        lmbda_guess *= 2.0  # Increase guess if not successful

    return sol.x, sol.y[0]


# Example usage:
def p(x):
    return 1
def q(x):
    return -np.pi ** 2

def pd(x):
    return 0
def w(x):
    return 1


a = 0
b = np.pi
alpha = 0
beta = 0
N = 100

x, u = sturm_liouville_solver(p, q, w, a, b, alpha, beta, N)
print("Eigenvalues:", x)
print("Eigenfunctions:", u)
a = 0
b = np.pi
alpha = 0
beta = 0
N = 100

x, u = sturm_liouville_solver(p, q, w, a, b, alpha, beta, N)
print("Eigenvalues:", x)
print("Eigenfunctions:", u)


