import numpy as np
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

# Define the parameters
N = 100  # Number of grid points
L = 5.0  # Length of the domain
x = np.linspace(0.01,0.99, N)  # Discretized domain

# Define the differential operator matrix
h = x[1] - x[0]  # Grid spacing
D = diags([-1, 0, 1], [-1, 0, 1], shape=(N, N)) / (2 * h)  # First derivative matrix
A = D.dot((1 - x**2) * x**2 * D.dot(D)) - 2 * x**2 * np.eye(N)  # Differential operator matrix

# Solve the eigenvalue problem
eigenvalues, eigenfunctions = eigsh(A, k=5, which='LM')  # Solve for the 5 smallest eigenvalues

# Sort the eigenvalues and eigenfunctions
idx = eigenvalues.argsort()
eigenvalues = eigenvalues[idx]
eigenfunctions = eigenfunctions[:, idx]

# Display the results
for i in range(5):
    print(f"Eigenvalue {i+1}: {eigenvalues[i]}")
