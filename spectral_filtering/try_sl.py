import jax.numpy as jnp
import numpy as np
from ode_strum_liouville import Ode_strum_liouville
from plot_dir.plot_hankel import plot_hankel_eigenvectors, plot_ODE, plot_ODE_Hankel
import matplotlib.pyplot as plt
import pickle5 as pickle

T = 1000
ind = jnp.array([0,1,2,5,7,8])
with open('Hankel_100', 'rb') as pr:
    my_hankel = pickle.load(pr)[0]

lmbd = -0.89
my_ode_solver = Ode_strum_liouville(T)
i = jnp.array([1])
my_ode_solution = my_ode_solver.solve_ode_sl(lmbd)
my_plot_ODE = plot_ODE(my_ode_solution, T, my_ode_solution.p[0], normalized=True)
print(my_ode_solution.p[0])
my_plot_Hankel = plot_hankel_eigenvectors(my_hankel.phi, i)
plt.show()
# sigma2lambda = np.array([[0, 1, 2],[-1.0,-0.9, -1.7, -3.0],[-0.044,-0.89, -1.73, -2.91]])


# i = 1
# my_plot_Hankel = plot_ODE_Hankel(my_ode_solution, T, my_ode_solution.p[0], my_hankel.phi[:,i], i)
# sigma2lambda = np.array([[1, 4, 6],[-3.0, -5.0, -8.0],[-3.48, -6.05, -9.38]])

# my_ode_solution = my_ode_solver.solve_ode_sl_lmbd_fixed(lmbd)
# my_plot_ODE = plot_ODE(my_ode_solution, T, lmbd, normalized=True)
# i = 3
# my_plot_Hankel = plot_ODE_Hankel(my_ode_solution, T, lmbd, my_hankel.phi[:, i], i)
# sigma2lambda = np.array([[0, 1, 2, 3],[-0.01, -3.48, -6.05, -9.38]])