import numpy as np
from scipy.integrate import solve_bvp
from plot_dir.plot_hankel import plot_ODE


def p(x):
    return x**2-x**4


def pd(x):
    return 2*x-4*x**3


def q(x):
    return -2*x**2


def w(x):
    return 1


def fun(x, y, lmbd):
    return np.vstack((y[1], (-q(x) * y[0] + w(x) * y[0] * lmbd) / p(x)-(pd(x) * y[1])/ p(x)))


def bc(ya, yb, lmbd):
    k = lmbd[0]
    return np.array([ya[0], yb[0] , ya[1]])


T = 1000
t_start = 1 / T
x = np.linspace(t_start, 1-t_start, T-1)
y = np.zeros((2, x.size))
y[0, 0] = 1
# y[0, 3] = -1
sol = solve_bvp(fun, bc, x, y, p=[-97])
print(sol.p[0])

my_plot = plot_ODE(sol, T, sol.p[0])