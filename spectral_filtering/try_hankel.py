import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pickle
import copy
from scipy import signal
from hankel_matrix import Hankel_matrix
from ode_strum_liouville import Ode_strum_liouville
from plot_dir.plot_hankel import plot_hankel_eigenvectors, plot_ODE, plot_ODE_Hankel2, plot_Hankel_ODE
from utility.useful_functions import orth_hankel, orth_sl, y_sl


# Building the Hankel matrix and saving or loading the saved data
T = 1000
# my_hankel = Hankel_matrix(T)
# with open('Hankel_1000', 'wb') as pw:
#     pickle.dump([my_hankel], pw)
# print("End of program")
with open('Hankel_1000', 'rb') as pr:
    my_hankel = pickle.load(pr)[0]

# Plot Hankel eigen vectors
# ind = jnp.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
# my_plot_Hankel = plot_hankel_eigenvectors(my_hankel.phi, ind, normalized=False)
# plt.show()
# orth_hankel(my_hankel, 50)


# associating solutions to ode and Hankel
lam_set = np.array([-87])
y_ode = y_sl(lam_set, T)
my_plot_ODE = plot_ODE(y_ode, lam_set)
ind = jnp.array([24])
my_plot_Hankel_ODE = plot_Hankel_ODE(my_hankel.phi, ind, y_ode, lam_set)
plt.show()

# my_ode_solver = Ode_strum_liouville(T)
# my_ode_solution = my_ode_solver.solve_ode_sl(-87)
# print(my_ode_solution.p[0])
# my_plot_ODE = plot_ODE(my_ode_solution, T, my_ode_solution.p[0], normalized=False)
# i = 24
# my_plot_Hankel = plot_ODE_Hankel2(my_ode_solution, T, my_ode_solution.p[0], my_hankel.phi[:,i], i, normalized=True)
# plt.show()

# Plotting ODE solutions and checking orthogonality
# lam_set = np.array([-15, -25, -35, -40, -44, -50, -60, -65, -75, -80, -87])
# y_ode = y_sl(lam_set, T)
# my_plot_ODE = plot_ODE(y_ode, np.array([-15]))
# my_plot_ODE = plot_ODE(y_ode, np.array([-15, -25, -35, -40, -44, -50]))
# my_plot_ODE = plot_ODE(y_ode, np.array([-60, -65, -75, -80, -87]))
# plt.show()
# orth_sl(y_ode)



# Filtering eigenvectors
# my_hankel_filtered = copy.deepcopy(my_hankel)
# low_pass = signal.butter(N=10, Wn=0.05, btype='low', analog=False, output='sos', fs=1)
#
# for i in range(14,20):
#     my_hankel_filtered.phi = my_hankel_filtered.phi.at[:, i].set( signal.sosfilt(low_pass, my_hankel.phi[:, i]))
# orth_hankel(my_hankel_filtered, 50)

# plotting filtered data
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.plot(jnp.arange(T), my_hankel.phi[:, i])
# ax1.set_title('The true eigenvector')
# ax1.axis([0, T, -1, 1])
# ax2.plot(jnp.arange(T), filtered_phi)
# ax2.set_title('Filtered eigenvector')
# ax2.axis([0, T, -1, 1])
# ax2.set_xlabel('Time [seconds]')
# plt.tight_layout()
# plt.show()