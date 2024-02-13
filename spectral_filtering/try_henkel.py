import jax.numpy as jnp
from henkel_matrix import Henkel_matrix
from plot_dir.plot_henkel import plot_henkel_eigenvectors


L = 100
my_henkel = Henkel_matrix(L)
# print(my_henkel.Z)
print("The eigenvalues")
print(my_henkel.sigma)
# print("The eigenvectors")
# print(my_henkel.phi)

ind = jnp.array([1])
plot_henkel_eigenvectors(my_henkel.phi, ind)