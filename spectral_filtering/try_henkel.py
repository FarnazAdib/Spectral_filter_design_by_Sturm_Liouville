import jax.numpy as jnp
from henkel_matrix import Henkel_matrix
from plot_dir.plot_henkel import plot_henkel_eigenvectors


T = 500
my_henkel = Henkel_matrix(T)
# print(my_henkel.Z)
print("The eigenvalues")
print(my_henkel.sigma)
print("The eigenvectors")
print(my_henkel.phi[:,0])

ind = jnp.array([0, 1, 3, 5, 8, 10])
plot_henkel_eigenvectors(my_henkel.phi, ind)