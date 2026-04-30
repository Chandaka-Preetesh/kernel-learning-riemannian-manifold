"""
bregman.py
----------
Bregman-projection kernel learning.

Goal (from the paper):
    min  D(K, K0)          — stay close to the initial kernel
    s.t. K_ii + K_jj - 2 K_ij  =  d_geo(i,j)^2   for all (i,j)

The algorithm iteratively corrects each pairwise constraint:

    dist   = K_ii + K_jj - 2 K_ij          (current kernel distance)
    target = d_geo(i,j)^2                   (desired geodesic distance²)
    δ      = (target - dist) / 4

    K_ii  +=  δ
    K_jj  +=  δ
    K_ij  -=  δ
    K_ji  -=  δ

After all iterations we project K onto the positive-semidefinite cone to keep
it a valid kernel matrix.
"""

import numpy as np


def bregman_kernel_learning(
    K0: np.ndarray,
    D_geo: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
    verbose: bool = False,
) -> np.ndarray:
    """
    Parameters
    ----------
    K0 : np.ndarray  shape (N, N)
        Initial kernel matrix (from kernel_learning.initial_kernel).
    D_geo : np.ndarray  shape (N, N)
        Geodesic distance matrix.
    max_iter : int
        Maximum number of outer iterations.
    tol : float
        Convergence tolerance on the mean absolute constraint violation.
    verbose : bool
        Print progress every 10 iterations.

    Returns
    -------
    K : np.ndarray  shape (N, N)
        Optimised kernel matrix preserving geodesic distances.
    """
    K = K0.copy()
    N = K.shape[0]
    n_pairs = N * (N - 1) // 2

    if verbose:
        print(f"    Bregman optimisation: N={N}, pairs={n_pairs}, max_iter={max_iter}, tol={tol}")

    for it in range(max_iter):
        max_violation = 0.0
        sum_violation = 0.0
        sum_sq_violation = 0.0

        for i in range(N):
            for j in range(i + 1, N):
                # Current kernel-induced distance²
                dist = K[i, i] + K[j, j] - 2 * K[i, j]

                # Target geodesic distance²
                target = D_geo[i, j] ** 2

                # Constraint error
                error = target - dist

                # Update (δ = error / 4  keeps the constraint balanced)
                delta = error / 4.0

                K[i, i] += delta
                K[j, j] += delta
                K[i, j] -= delta
                K[j, i] -= delta

                abs_error = abs(error)
                max_violation = max(max_violation, abs_error)
                sum_violation += abs_error
                sum_sq_violation += error ** 2

        mean_violation = sum_violation / n_pairs
        rmse_violation = (sum_sq_violation / n_pairs) ** 0.5

        # [GRAPH DATA] Log every 5th iteration for convergence curve
        if verbose and (it % 5 == 0 or it == max_iter - 1):
            print(f"    BREGMAN_ITER_{it:04d}_MAX_VIOLATION  = {max_violation:.6e}")
            print(f"    BREGMAN_ITER_{it:04d}_MEAN_VIOLATION = {mean_violation:.6e}")
            print(f"    BREGMAN_ITER_{it:04d}_RMSE_VIOLATION = {rmse_violation:.6e}")

        if max_violation < tol:
            if verbose:
                print(f"  Converged at iteration {it}.")
                print(f"    BREGMAN_CONVERGED_AT_ITER = {it}")
                print(f"    BREGMAN_TOTAL_ITERS = {it + 1}")
            break
    else:
        if verbose:
            print(f"    BREGMAN_CONVERGED_AT_ITER = -1")
            print(f"    BREGMAN_TOTAL_ITERS = {max_iter}")

    # --- Project K onto PSD cone ---
    K = (K + K.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(K)

    n_negative = np.sum(eigvals < 0)
    if verbose:
        print(f"    BREGMAN_PSD_NEGATIVE_EIGVALS = {n_negative}")
        print(f"    BREGMAN_PSD_EIGVAL_MIN_BEFORE_CLIP = {eigvals.min():.6e}")
        print(f"    BREGMAN_PSD_EIGVAL_MAX = {eigvals.max():.6e}")

    eigvals = np.clip(eigvals, 0, None)        # remove negative eigenvalues
    K = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return K
