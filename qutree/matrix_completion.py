from numpy.linalg import svd
import numpy as np

def low_rank_complete(X, rank, max_iter=50):
    X_filled = X.copy()
    nan_mask = np.isnan(X)
    # Initialize missing values with column means
    col_means = np.nanmean(X_filled, axis=0)
    inds = np.where(nan_mask)
    X_filled[inds] = np.take(col_means, inds[1])
    for _ in range(max_iter):
        # SVD and rank truncation
        U, S, Vt = svd(X_filled, full_matrices=False)
        S[rank:] = 0
        X_new = (U * S) @ Vt
        # Restore observed entries
        X_filled[~nan_mask] = X[~nan_mask]
        X_filled[nan_mask] = X_new[nan_mask]
    return X_filled


def simple_ALS(X, rank=2, n_iter=10, lam=0.1):
    m, n = X.shape
    U = np.random.rand(m, rank)
    V = np.random.rand(n, rank)

    mask = ~np.isnan(X)

    for _ in range(n_iter):
        for i in range(m):
            Vi = V[mask[i, :]]
            Xi = X[i, mask[i, :]]
            U[i] = np.linalg.solve(Vi.T @ Vi + lam * np.eye(rank), Vi.T @ Xi)

        for j in range(n):
            Uj = U[mask[:, j]]
            Xj = X[mask[:, j], j]
            V[j] = np.linalg.solve(Uj.T @ Uj + lam * np.eye(rank), Uj.T @ Xj)

    return U @ V.T
