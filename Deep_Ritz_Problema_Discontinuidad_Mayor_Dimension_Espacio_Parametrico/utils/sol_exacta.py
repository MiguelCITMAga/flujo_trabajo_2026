import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def solve_poisson_fd_two_materials(
    N, k1, k2, beta=0.5, sigma = (0.8, 0.8), domain=(0.0, 1.0)
):
    """
    Resuelve:
    -div(k(x) grad u) = f
    con k discontinuo en x=0.5
    Dirichlet homogéneo

    N : puntos por dirección
    k1, k2 : coeficientes de material
    mu : centro de la fuente (mu1, mu2)
    """

    a, b = domain
    h = (b - a) / (N - 1)

    x = np.linspace(a, b, N)
    y = np.linspace(a, b, N)
    X, Y = np.meshgrid(x, y, indexing="ij")

    def k_func(x):
        return k1 if x < beta else k2

    def f_func(x, y):
        return np.exp(-2 * ((x - sigma[0])**2 + (y - sigma[1])**2))

    n = N * N
    A = sp.lil_matrix((n, n))
    b_vec = np.zeros(n)

    def idx(i, j):
        return i * N + j

    for i in range(N):
        for j in range(N):
            p = idx(i, j)

            # frontera
            if i == 0 or i == N-1 or j == 0 or j == N-1:
                A[p, p] = 1.0
                b_vec[p] = 0.0
                continue

            x_ij = x[i]

            # coeficientes en caras (promedio armónico)
            k_e = 2 * k_func(x[i]) * k_func(x[i+1]) / (k_func(x[i]) + k_func(x[i+1]))
            k_w = 2 * k_func(x[i]) * k_func(x[i-1]) / (k_func(x[i]) + k_func(x[i-1]))
            k_n = k_func(x[i])
            k_s = k_func(x[i])

            A[p, p] = (k_e + k_w + k_n + k_s) / h**2
            A[p, idx(i+1, j)] = -k_e / h**2
            A[p, idx(i-1, j)] = -k_w / h**2
            A[p, idx(i, j+1)] = -k_n / h**2
            A[p, idx(i, j-1)] = -k_s / h**2

            b_vec[p] = f_func(X[i, j], Y[i, j])

    u_flat = spla.spsolve(A.tocsr(), b_vec)
    u = u_flat.reshape((N, N))

    return X, Y, u
