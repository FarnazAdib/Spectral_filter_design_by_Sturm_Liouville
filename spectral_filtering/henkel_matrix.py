import jax.numpy as jnp

class Henkel_matrix:
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

    def _find_eigens(self):
        sigma_unorderd, phi_unorderd = jnp.linalg.eigh(self.Z)
        idx = sigma_unorderd.argsort()[::-1]
        sigma = sigma_unorderd[idx]
        phi = phi_unorderd[:, idx]
        return sigma, phi
