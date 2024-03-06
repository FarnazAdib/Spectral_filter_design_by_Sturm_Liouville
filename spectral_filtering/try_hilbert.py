import jax.numpy as jnp
from hankel_matrix import Hankel_matrix


T = 6
theta = -1.0
my_hankel = Hankel_matrix(T)
H = my_hankel.build_hilbert(theta)
eigenvalue_H, eigenvector_H = jnp.linalg.eigh(H)
idx = eigenvalue_H.argsort()[::-1]
eigenvalue_H = eigenvalue_H[idx]
eigenvector_H = eigenvector_H[:, idx]
print("Eigenvalues of Hilbert matrix:")
print(eigenvalue_H)

Tri_T = my_hankel.build_tridiagonal(theta)
eigenvalue_T, eigenvector_T = jnp.linalg.eigh(Tri_T)
idx = eigenvalue_T.argsort()[::-1]
eigenvalue_T = eigenvalue_T[idx]
eigenvector_T = eigenvector_T[:, idx]
print("Eigenvalues of the triangular matrix:")
print(eigenvalue_T)

for i in range(T):
    print("Eigenvector", i)
    print(eigenvector_H[:, i])
    print(eigenvector_T[:, i])
print("End of program")



# my_ode_solver = Ode_strum_liouville(T)
# my_ode_solution = my_ode_solver.solve_ode_sl(-8.0)
# print(my_ode_solution.p[0])
# my_plot_ODE = plot_ODE(my_ode_solution, T, my_ode_solution.p[0], normalized=False)
# i = 3
# my_plot_Hankel = plot_ODE_Hankel(my_ode_solution, T, my_ode_solution.p[0], my_hankel.phi[:,i], i)