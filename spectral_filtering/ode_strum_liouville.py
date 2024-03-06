import numpy as np
from scipy.integrate import solve_bvp

class Ode_strum_liouville:
    def __init__(self, T):
        self.T = T
        self.t_start = 1 / T
        self._p = lambda x: x**2-x**4
        self._pd = lambda x: 2*x-4*x**3
        self._q = lambda x: -2*x**2
        self._w = lambda w: 1

    def _fun(self, x, y, lmbd):
        y1d = -2 * y[1] * (1 - 2 * x ** 2) / (x - x ** 3) + 2 * y[0] / (1 - x ** 2) + lmbd[0] * y[0] / (x ** 2 - x ** 4)
        # return np.vstack((y[1], (-q(x) * y[0] + w(x) * y[0] * lmbd) / p(x)-(pd(x) * y[1])/ p(x)))
        return np.vstack((y[1], y1d))

    def _bc(self, ya, yb, lmbd):
        k = lmbd[0]
        return np.array([ya[0], yb[0], ya[1] - k])
        # return np.array([ya[0], yb[0], ya[1]])

    def solve_ode_sl(self, lmbd):
        x = np.linspace(self.t_start, 1 - self.t_start, self.T)
        y = np.zeros((2, x.size))
        y[0, 0] = 1
        sol = solve_bvp(self._fun, self._bc, x, y, p=[lmbd])
        return sol

    def solve_ode_sl_lmbd_fixed(self, lmbd):
        x = np.linspace(self.t_start, 1 - self.t_start, self.T)
        y = np.zeros((2, x.size))
        y[0, 0] = 1

        def fun(x, y):
            y1d = -2 * y[1] * (1 - 2 * x ** 2) / (x - x ** 3) + 2 * y[0] / (1 - x ** 2) + lmbd * y[0] / (
                        x ** 2 - x ** 4)
            return np.vstack((y[1], y1d))

        def bc(ya, yb):
            return np.array([ya[0], yb[0]])

        sol = solve_bvp(fun, bc, x, y)
        return sol