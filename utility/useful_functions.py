import numpy as np
from sklearn.preprocessing import normalize
from spectral_filtering.ode_strum_liouville import Ode_strum_liouville

def norm_and_scale(y):
    y_max = np.max(y)
    y_min = np.min(y)
    if y_max > abs(y_min):
        y = normalize(y.reshape(-1, 1), axis=0, norm="l2")
    else:
        y = normalize(-y.reshape(-1, 1), axis=0, norm="l2")
    return y


def norm_diff(y, x):
    y = norm_and_scale(y)
    x = norm_and_scale(x)
    return np.linalg.norm(y-x)


def orth_hankel(my_hankel, N):
    for i in range(N):
        o_i = 0.0
        for j in range(N):
            if j != i:
                o_ij = np.abs(np.dot(my_hankel.phi[:, i], my_hankel.phi[:, j]))
                if o_ij > o_i:
                    o_i = o_ij
        print("maximum othogoanility index for",i ,"is", o_i)

def y_sl(lam_set, T):
    y_ode = np.zeros([T, lam_set.size])
    my_ode_solver = Ode_strum_liouville(T)
    t_start = 1.0 / T
    x_plot = np.linspace(t_start, 1 - t_start, T)
    for i in range(lam_set.size):
        my_ode_solution = my_ode_solver.solve_ode_sl(lam_set[i])
        y = my_ode_solution.sol(x_plot)[0]
        print("The unknown parameter:", my_ode_solution.p[0])
        y = normalize(y.reshape(-1, 1), axis=0, norm="l2")
        y_ode[:, i] = y[:, 0]
    return y_ode


def orth_sl(y_ode):
    Len = y_ode.shape[1]
    for i in range(Len):
        o_i = 0.0
        for j in range(Len):
            if j != i:
                o_ij = np.abs(np.dot(y_ode[:, i], y_ode[:, j]))
                # print("i is",i,"and j is",j, "The orthogonality index o_ij is", o_ij)
                if o_ij > o_i:
                    o_i = o_ij
        print("maximum othogoanility index for", i, "is", o_i)

