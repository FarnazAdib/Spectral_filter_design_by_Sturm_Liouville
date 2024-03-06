import jax.numpy as jnp


class Hankel_matrix:
    def __init__(self, L: int):
        self.L = L
        self.Z = self._build_henkel()
        self.sigma, self.phi = self._find_eigens()


    def _build_henkel(self):
        Z = jnp.zeros([self.L, self.L])
        for i in range(self.L):
            for j in range(self.L):
                Z = Z.at[i, j].set(2/((i+j+2)**3-(i+j+2)))
        return Z

    def build_hilbert(self, theta):
        H = jnp.zeros([self.L, self.L])
        for i in range(self.L):
            for j in range(self.L):
                H = H.at[i, j].set(1 / (i + j+2 + theta))
        return H

    def build_tridiagonal(self, theta):
        T = jnp.zeros([self.L, self.L])
        for i in range(self.L):
            T = T.at[i, i].set(-2 * (self.L - i - 1) * (self.L + i + 1 + theta) * ((i + 1)**2 + theta * i - self.L))
        for i in range(self.L - 1):
            T = T.at[i, i + 1].set((i + 1) * (self.L - i - 1) * (theta + i + 2) * (theta + self.L + i + 2))
            T = T.at[i + 1, i].set((i + 1) * (self.L - i - 1) * (theta + i + 2) * (theta + self.L + i + 2))
        return T

    def _find_eigens(self):
        sigma_unorderd, phi_unorderd = jnp.linalg.eigh(self.Z)
        idx = sigma_unorderd.argsort()[::-1]
        sigma = sigma_unorderd[idx]
        phi = phi_unorderd[:, idx]
        return sigma, phi
