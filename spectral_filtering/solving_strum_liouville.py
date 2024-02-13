import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

def p(x):
    return 1

def pd(x):
    return 0

def q(x):
    return 0

def w(x):
    return -1
def fun(x, y, lmbd):
    k = lmbd[0]
    return np.vstack((y[1], (-q(x) * y[0] + w(x) * y[0] * k**2) / p(x)-(pd(x) * y[1])/ p(x)))

def bc(ya, yb, lmbd):
    k = lmbd[0]
    return np.array([ya[0], yb[0] , ya[1]-k])


x = np.linspace(0, 1, 5)
y = np.zeros((2, x.size))
y[0, 1] = 1
y[0, 3] = -1
sol = solve_bvp(fun, bc, x, y, p=[6])
print(sol.p[0])


x_plot = np.linspace(0, 1, 100)
y_plot = sol.sol(x_plot)[0]
plt.plot(x_plot, y_plot)
plt.xlabel("x")
plt.ylabel("y")
plt.show()