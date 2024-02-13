import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns


def plot_henkel_eigenvectors(phi, ind):
    _colors, _label_fontsize, _legend_fontsize = set_properties()
    # plt.style.use('seaborn')
    sns.set_theme()
    plt.figure()
    ax = plt.gca()
    T = phi.shape[0]
    for i in range(len(ind)):
        plt.plot(jnp.arange(T), phi[:, ind[i]], label=f'$\phi_{{{ind[i]}}}$', color=_colors[i])
        plt.axis([0, T - 1, -1, 1])
        ax.set_xlabel("Time", fontsize=_label_fontsize)
        # ax.set_ylabel('$r_{:1}$'.format(i + 1)+'$_k$', fontsize=_label_fontsize)
        plt.legend(fontsize=_legend_fontsize)
        # plt.title(title)
        plt.show()
    return plt




def set_properties():
    _label_fontsize = 16
    _legend_fontsize = 16
    _colors = ["gray", "orange", "teal", "aqua", "green", "red", 'black', "violet", "aqua" ]

    return _colors, _label_fontsize, _legend_fontsize
