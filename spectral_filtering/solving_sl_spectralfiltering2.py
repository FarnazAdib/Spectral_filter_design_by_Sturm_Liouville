import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

def p(x):
    return x**2-x**4

def pd(x):
    return 2*x-4*x**3

def q(x):
    return -2*x**2

def w(x):
    return 1
def fun(x, y, lmbd):
    y1d = -2*y[1]*(1-2*x**2)/(x-x**3) + 2*y[0] / (1-x**2) + lmbd[0]*y[0]/(x**2-x**4)
    # return np.vstack((y[1], (-q(x) * y[0] + w(x) * y[0] * lmbd) / p(x)-(pd(x) * y[1])/ p(x)))
    return np.vstack((y[1], y1d))
def bc(ya, yb, lmbd):
    k = lmbd[0]
    return np.array([ya[0], yb[0] , ya[1]-k])


x = np.linspace(0.01, 0.99, 99)
y = np.zeros((2, x.size))
y[0, 0] = 1
# y[0, 3] = -1
sol = solve_bvp(fun, bc, x, y, p=[0.003])
print(sol.p[0])


x_plot = np.linspace(0.01, 0.99, 99)
y_plot = sol.sol(x_plot)[0]
plt.plot(x_plot, y_plot)
plt.xlabel("x")
plt.ylabel("y")
plt.show()