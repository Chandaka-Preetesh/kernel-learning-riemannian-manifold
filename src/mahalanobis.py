"""
mahalanobis.py
--------------
Learn a Mahalanobis distance matrix A using label-supervised constraints.

Goal (from paper):
    d_A(xi, xj)^2 = (xi - xj)^T  A  (xi - xj)

We learn A so that:
    - Same-class pairs have SMALL Mahalanobis distance
    - Different-class pairs have LARGE Mahalanobis distance

This is a simplified ITML (Information-Theoretic Metric Learning) approach
that uses rank-1 updates based on constraint violations.
"""

import numpy as np


def learn_mahalanobis(
    X: np.ndarray,
    D_geo: np.ndarray,
    labels: np.ndarray = None,
    n_iters: int = 20,
    lr: float = 0.1,
) -> np.ndarray:
    """
    Learn the Mahalanobis matrix A with supervised constraints.

    Parameters
    ----------
    X : np.ndarray  shape (N, d)
        Vectorised log-mapped covariance descriptors (should be pre-normalised).
    D_geo : np.ndarray  shape (N, N)
        Target geodesic distance matrix.
    labels : np.ndarray  shape (N,)
        Class labels for supervised constraint generation.
    n_iters : int
        Number of outer passes over constraint pairs.
    lr : float
        Learning rate for each rank-1 update.

    Returns
    -------
    A : np.ndarray  shape (d, d)
        Learned symmetric positive-definite matrix.
    """
    N, d = X.shape
    A = np.eye(d, dtype=np.float64)

    if labels is not None:
        # ----- Supervised: ITML-style constraint learning -----
        similar_pairs = []
        dissimilar_pairs = []
        geo_sim_dists = []
        geo_dis_dists = []

        for i in range(N):
            for j in range(i + 1, N):
                if labels[i] == labels[j]:
                    similar_pairs.append((i, j))
                    geo_sim_dists.append(D_geo[i, j])
                else:
                    dissimilar_pairs.append((i, j))
                    geo_dis_dists.append(D_geo[i, j])

        u_threshold = np.mean(geo_sim_dists) if geo_sim_dists else 0.5
        l_threshold = np.mean(geo_dis_dists) if geo_dis_dists else 2.0
        u_sq = u_threshold ** 2
        l_sq = l_threshold ** 2

        print(f"    ITML thresholds: similar < {u_threshold:.4f}, dissimilar > {l_threshold:.4f}")
        print(f"    Pairs: {len(similar_pairs)} similar, {len(dissimilar_pairs)} dissimilar")

        for it in range(n_iters):
            n_updates = 0
            total_sim_violation = 0.0
            total_dis_violation = 0.0

            # --- Similar pairs: push distances DOWN ---
            for i, j in similar_pairs:
                diff = (X[i] - X[j]).reshape(-1, 1)
                curr_sq = float((diff.T @ A @ diff).item())
                if curr_sq > u_sq:
                    outer = diff @ diff.T
                    norm4 = max(float((diff.T @ diff).item()) ** 2, 1e-12)
                    A -= lr * (curr_sq - u_sq) * outer / norm4
                    n_updates += 1
                    total_sim_violation += (curr_sq - u_sq)

            # --- Dissimilar pairs: push distances UP ---
            for i, j in dissimilar_pairs:
                diff = (X[i] - X[j]).reshape(-1, 1)
                curr_sq = float((diff.T @ A @ diff).item())
                if curr_sq < l_sq:
                    outer = diff @ diff.T
                    norm4 = max(float((diff.T @ diff).item()) ** 2, 1e-12)
                    A += lr * (l_sq - curr_sq) * outer / norm4
                    n_updates += 1
                    total_dis_violation += (l_sq - curr_sq)

            # --- Project A back to SPD with bounded condition number ---
            A = (A + A.T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(A)
            eigvals = np.clip(eigvals, 1e-2, 1e2)
            A = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # [GRAPH DATA] Log every 5th iteration for Mahalanobis convergence
            if it % 5 == 0 or it == n_iters - 1:
                print(f"    ITML_ITER_{it:03d}_UPDATES         = {n_updates}")
                print(f"    ITML_ITER_{it:03d}_EIGVAL_MIN      = {eigvals.min():.6f}")
                print(f"    ITML_ITER_{it:03d}_EIGVAL_MAX      = {eigvals.max():.6f}")
                print(f"    ITML_ITER_{it:03d}_COND_NUMBER     = {eigvals.max() / max(eigvals.min(), 1e-15):.6f}")
                print(f"    ITML_ITER_{it:03d}_SIM_VIOLATION   = {total_sim_violation:.6e}")
                print(f"    ITML_ITER_{it:03d}_DIS_VIOLATION   = {total_dis_violation:.6e}")
                print(f"    ITML_ITER_{it:03d}_A_FROB_NORM     = {np.linalg.norm(A, 'fro'):.6f}")

    else:
        # ----- Unsupervised fallback -----
        for it in range(n_iters):
            for i in range(N):
                for j in range(i + 1, N):
                    diff = (X[i] - X[j]).reshape(-1, 1)
                    curr_sq = float((diff.T @ A @ diff).item())
                    target_sq = D_geo[i, j] ** 2
                    gap = target_sq - curr_sq
                    outer = diff @ diff.T
                    norm4 = max(float((diff.T @ diff).item()) ** 2, 1e-12)
                    A += lr * gap * outer / norm4

            A = (A + A.T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(A)
            eigvals = np.clip(eigvals, 1e-2, 1e2)
            A = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return A
