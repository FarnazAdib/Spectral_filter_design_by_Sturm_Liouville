import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from utility.useful_functions import norm_and_scale, norm_diff
import matplotlib.colors as mcolors


def plot_hankel_eigenvectors(phi, ind, normalized=True):
    _colors, _label_fontsize, _legend_fontsize = set_properties()
    sns.set_theme()
    plt.figure()
    ax = plt.gca()
    T = phi.shape[0]
    if normalized:
        phi = normalize(phi, axis=0, norm="l2")
    for i in range(len(ind)):
        plt.plot(jnp.arange(T), phi[:, ind[i]], label=f'$\phi_{{{ind[i]+1}}}$', color=_colors[i])
        plt.axis([0, T - 1, -1, 1])
        plt.legend(fontsize=_legend_fontsize)
        # plt.title(title)
    return plt


def plot_ODE(y_ode, lmbd_set, normalized=True):
    _colors, _label_fontsize, _legend_fontsize = set_properties()
    sns.set_theme()
    plt.figure()
    ax = plt.gca()
    T = y_ode.shape[0]
    if normalized:
        y_ode = normalize(y_ode, axis=0, norm="l2")
    for i in range(len(lmbd_set)):
        plt.plot(jnp.arange(T), y_ode[:, i], label='$\zeta$('+str(round(-lmbd_set[i], 2))+')', color=_colors[i])
        ax.set(xlim=(0, T))
        # plt.axis([0, T - 1, -1, 1])
        plt.legend(fontsize=_legend_fontsize)

    return plt


def plot_Hankel_ODE(phi, ind, y_ode, lmbd_set, normalized=True):
    _colors, _label_fontsize, _legend_fontsize = set_properties()
    sns.set_theme()
    plt.figure()
    ax = plt.gca()
    T = phi.shape[0]
    if normalized:
        phi = normalize(phi, axis=0, norm="l2")
        y_ode = normalize(y_ode, axis=0, norm="l2")
    for i in range(len(ind)):
        plt.plot(jnp.arange(T), phi[:, ind[i]], label=f'$\phi_{{{ind[i] + 1}}}$', color=_colors[2*i])
        plt.plot(jnp.arange(T), y_ode[:, i], label='$\zeta$('+str(round(-lmbd_set[i], 2))+')', color=_colors[2*i+1])
        ax.set(xlim=(0, T))
        plt.legend(fontsize=_legend_fontsize)
    return plt


def plot_ODE_Hankel2(sol, T, lmbd, phi_i, ind_i, normalized=True):
    _colors, _label_fontsize, _legend_fontsize = set_properties()
    # plt.style.use('seaborn')
    sns.set_theme()
    fig = plt.figure()
    plt.figure()
    ax = plt.gca()
    t_start = 1.0 / T
    x_plot = np.linspace(t_start, 1-t_start, T)
    y_plot = sol.sol(x_plot)[0]
    if normalized:
        y_plot = norm_and_scale(y_plot)
        print("Norm of ode solution", np.linalg.norm(y_plot))
        phi_i = norm_and_scale(phi_i)
        print("Norm of eigenvector", np.linalg.norm(phi_i))
        # y_plot = normalize(y_plot.reshape(-1, 1), axis=0, norm="l2")
        # phi_i = normalize(phi_i.reshape(-1, 1), axis=0, norm="l2")
    plt.plot(jnp.arange(T), y_plot, label='$\phi_{ODE}$('+str(round(-lmbd, 2))+')', color=_colors[0])
    plt.plot(jnp.arange(T), phi_i, label=f'$\phi_{{{ind_i+1}}}$', color=_colors[1])
    ax.set(xlim=(0, T))
    # plt.axis([0, T - 1, -1, 1])
    # ax.set_xlabel("$N$", fontsize=_label_fontsize)

    plt.legend(fontsize=_legend_fontsize)
    # plt.show()
    print("Norm difference between ODE and eignvector is", norm_diff(y_plot,phi_i))
    return plt

def set_properties():
    _label_fontsize = 16
    _legend_fontsize = 16
    # _colors = ["gray", "orange", "teal", "aqua", "green", "red", 'black', "violet", "aqua" ]
    # _colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
    #            'tab:olive' 'tab:cyan']
    _colors = [mcolors.TABLEAU_COLORS['tab:blue'], mcolors.TABLEAU_COLORS['tab:orange'],
               mcolors.TABLEAU_COLORS['tab:green'], mcolors.TABLEAU_COLORS['tab:red'],
               mcolors.TABLEAU_COLORS['tab:purple'], mcolors.TABLEAU_COLORS['tab:brown'],
               mcolors.TABLEAU_COLORS['tab:pink'], mcolors.TABLEAU_COLORS['tab:gray'],
               mcolors.TABLEAU_COLORS['tab:olive'], mcolors.TABLEAU_COLORS['tab:cyan']]
    return _colors, _label_fontsize, _legend_fontsize
