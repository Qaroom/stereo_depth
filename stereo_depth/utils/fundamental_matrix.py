"""
Fundamental and Essential matrix estimation.
Ported from the original project (FundamentalEssentialMatrix.py).
Reference: https://github.com/sakshikakde/Buildings-built-in-minutes-An-SfM-Approach
"""
import numpy as np


def normalize(uv):
    """Hartley normalization for the 8-point algorithm."""
    uv_dash = np.mean(uv, axis=0)
    u_dash, v_dash = uv_dash[0], uv_dash[1]

    u_cap = uv[:, 0] - u_dash
    v_cap = uv[:, 1] - v_dash

    s = (2 / np.mean(u_cap ** 2 + v_cap ** 2)) ** 0.5
    T_scale = np.diag([s, s, 1])
    T_trans = np.array([[1, 0, -u_dash], [0, 1, -v_dash], [0, 0, 1]])
    T = T_scale.dot(T_trans)

    x_ = np.column_stack((uv, np.ones(len(uv))))
    x_norm = (T.dot(x_.T)).T
    return x_norm, T


def estimate_fundamental_matrix(feature_matches, normalised=True):
    """8-point algorithm to estimate F from N>=8 correspondences."""
    x1 = feature_matches[:, 0:2]
    x2 = feature_matches[:, 2:4]

    if x1.shape[0] < 8:
        return None

    if normalised:
        x1_norm, T1 = normalize(x1)
        x2_norm, T2 = normalize(x2)
    else:
        x1_norm, x2_norm = x1, x2
        T1 = T2 = np.eye(3)

    A = np.zeros((len(x1_norm), 9))
    for i in range(len(x1_norm)):
        x_1, y_1 = x1_norm[i][0], x1_norm[i][1]
        x_2, y_2 = x2_norm[i][0], x2_norm[i][1]
        A[i] = np.array([x_1 * x_2, x_2 * y_1, x_2,
                         y_2 * x_1, y_2 * y_1, y_2,
                         x_1, y_1, 1])

    _, _, VT = np.linalg.svd(A, full_matrices=True)
    F = VT.T[:, -1].reshape(3, 3)

    # Enforce rank-2 constraint
    u, s, vt = np.linalg.svd(F)
    s = np.diag(s)
    s[2, 2] = 0
    F = np.dot(u, np.dot(s, vt))

    if normalised:
        F = np.dot(T2.T, np.dot(F, T1))
    return F


def error_F(feature, F):
    x1, x2 = feature[0:2], feature[2:4]
    x1tmp = np.array([x1[0], x1[1], 1]).T
    x2tmp = np.array([x2[0], x2[1], 1])
    return np.abs(np.dot(x1tmp, np.dot(F, x2tmp)))


def get_inliers(features, n_iterations=1000, error_thresh=0.02):
    """RANSAC over feature correspondences to find the best F and its inliers."""
    inliers_thresh = 0
    chosen_indices = []
    chosen_f = None
    n_rows = features.shape[0]

    if n_rows < 8:
        return None, features

    for _ in range(n_iterations):
        indices = []
        random_indices = np.random.choice(n_rows, size=8, replace=False)
        features_8 = features[random_indices, :]
        f_8 = estimate_fundamental_matrix(features_8)
        if f_8 is None:
            continue
        for j in range(n_rows):
            if error_F(features[j], f_8) < error_thresh:
                indices.append(j)
        if len(indices) > inliers_thresh:
            inliers_thresh = len(indices)
            chosen_indices = indices
            chosen_f = f_8

    if chosen_f is None:
        return None, features
    return chosen_f, features[chosen_indices, :]


def get_essential_matrix(K1, K2, F):
    """E = K2^T F K1, then force singular values to (1,1,0)."""
    E = K2.T.dot(F).dot(K1)
    U, _, V = np.linalg.svd(E)
    s = [1, 1, 0]
    return np.dot(U, np.dot(np.diag(s), V))
