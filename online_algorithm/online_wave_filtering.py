import jax.numpy as jnp
import jax
from jax.numpy.linalg import inv
import numpy as np
from jax import grad
from jax import jit


from spectral_filtering.henkel_matrix import Henkel_matrix

class Online_wawe_filtering:
    def __init__(self, T:int, m:int, p:int, k=10, eta=0.01, R_M = 1.0):
        '''

        :param T: time horizon
        :param k: number of filters
        :param eta: learning rate
        :param R_M: radius parameter
        '''
        self.T = T
        my_henkel = Henkel_matrix(T)
        self.sigma = my_henkel.sigma[:k]
        self.phi = my_henkel.phi[:, :k]
        self.k = k
        self.k_prime = m * k + 2 * m + p
        self.M = jnp.zeros([p, self.k_prime])
        self.u_history = jnp.zeros((self.T, m, 1))
        self.X_tilde = jnp.zeros([self.k_prime, 1])
        self.m = m


    def run_wave_filtering(self):
        # The main loop

        for t in range(self.T):
            # for each filter. We have self.k filters
            for j in range(self.k):
                # build \sigma_j^{0.25} \sum_{i=1:T} \phi(j)(i)u(t-i)
                sum_j = 0.0
                idu = j * self.m + jnp.arange(self.m)
                for i in range(self.T):
                    sum_j = sum_j + self.sigma[j]**(0.25)*self.phi[i, j] * self.u_history[t-i, :]
                self.X_tilde = self.X_tilde.at[idu].set(sum_j)
            #adding u_{t-1} u(t), y_{t-1} to y



