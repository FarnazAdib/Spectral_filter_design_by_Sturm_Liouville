import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from plot_dir.plot_henkel import plot_ODE
# Fixed lmbd so it is not found.
lmbd = -97
def fun(x, y):
    y1d = -2*y[1]*(1-2*x**2)/(x-x**3) + 2*y[0] / (1-x**2) + lmbd*y[0]/(x**2-x**4)
    # return np.vstack((y[1], (-q(x) * y[0] + w(x) * y[0] * lmbd) / p(x)-(pd(x) * y[1])/ p(x)))
    return np.vstack((y[1], y1d))
def bc(ya, yb):

    return np.array([ya[0], yb[0]])

T = 1000
t_start = 1 / T

x = np.linspace(t_start, 1-t_start, T-1)
y = np.zeros((2, x.size))


y[0, 0] = 1.0
# y[0, 3] = -1
sol = solve_bvp(fun, bc, x, y)
my_plot = plot_ODE(sol, T, lmbd)