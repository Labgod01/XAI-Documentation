import numpy as np

def max_sensitivity(explain_fn, x, X_background, radius=0.1, n_samples=50, seed=42):
    """
    Max-Sensitivity (Yeh et al., 2019).

    Largest relative change of the explanation when the input is slightly
    perturbed within a neighbourhood of radius r around x:
        max_{||y - x|| <= r} ||Phi(y) - Phi(x)|| / ||Phi(x)||
    The maximum is estimated by Monte Carlo sampling.

    Follows the logic of get_exp_sens() in the authors' official code
    (github.com/chihkuanyeh/saliency_evaluation): uniform noise in an
    L-infinity ball, explanation recomputed at each perturbed point, change
    divided by the norm of the original explanation. Adapted to tabular data:
    the radius is scaled by the standard deviation of each feature, because
    features have different units (images in the paper all lie in [0, 1]).

    explain_fn   : function that takes a 1D instance and returns its
                   attributions as a 1D array (n_features,)
    x            : instance to explain, 1D array (n_features,)
    X_background : reference data, used only for the feature standard deviations,
                   2D array (N, n_features)
    radius       : neighbourhood radius, in standard deviations of each feature
    n_samples    : number of perturbed points (50 in the paper)
    seed         : random seed for reproducibility

    Returns max-sensitivity >= 0 (relative change): lower = more stable.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).ravel()
    X_background = np.asarray(X_background, dtype=float)
    n_features = x.shape[0]

    # 1. Original explanation Phi(x) and its norm (used to normalise the change)
    phi_x = np.asarray(explain_fn(x), dtype=float).ravel()
    norm_x = np.linalg.norm(phi_x)
    if norm_x == 0:
        raise ValueError("The explanation of x is all zeros; relative change is undefined.")

    # 2. Size of the neighbourhood for each feature, in its own units
    scale = radius * X_background.std(axis=0)

    max_diff = 0.0
    for _ in range(n_samples):
        # 3. Perturbed point y: uniform noise in [-r * std, +r * std] per feature
        y = x + rng.uniform(-scale, scale, size=n_features)

        # 4. Explanation recomputed at y
        phi_y = np.asarray(explain_fn(y), dtype=float).ravel()

        # 5. Relative change of the explanation; keep the largest
        diff = np.linalg.norm(phi_y - phi_x) / norm_x
        max_diff = max(max_diff, diff)

    return float(max_diff)