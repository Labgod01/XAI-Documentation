import numpy as np

def effective_complexity(model, x, attributions, X_background, eps, n_samples=100, seed=42):
    """
    Effective Complexity (Nguyen & Martinez, 2020).
 
    Smallest number k of top-ranked features (by |attribution|) that, when kept
    fixed at the values of x while the remaining features vary randomly, keep the
    model's prediction on average within a tolerance eps of the original prediction.
 
    model        : trained model with a .predict() method
    x            : instance to explain, 1D array (n_features,)
    attributions : explanation for x, 1D array (n_features,)
    X_background : reference data used to sample random values, 2D array (N, n_features)
    eps          : tolerance on the prediction deviation (in target units)
    n_samples    : number of random samples used to estimate the expectation
    seed         : random seed for reproducibility
 
    Returns k* (int): lower = more compact and correctly ranked explanation.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).ravel()
    attributions = np.asarray(attributions, dtype=float).ravel()
    X_background = np.asarray(X_background, dtype=float)
    n_features = x.shape[0]

    # 1. Original prediction f(x)
    f_x = model.predict(x.reshape(1, -1))[0]

    # 2. Feature ranking: from most to least important (by |attribution|)
    order = np.argsort(-np.abs(attributions))

    # 3. Try k = 1, 2, ..., n_features
    for k in range(1, n_features + 1):
        top_k = order[:k]

        # 3a. Sample random rows from the reference data
        idx = rng.integers(0, len(X_background), size=n_samples)
        Z = X_background[idx].copy()

        # 3b. Fix the top-k features at the values of x
        Z[:, top_k] = x[top_k]

        # 3c-3d. Mean absolute deviation from the original prediction
        deviation = np.mean(np.abs(model.predict(Z) - f_x))

        # Loss choice: the paper leaves the loss l(y*, f) generic. For regression
        # we use the absolute error |f(z) - y*| because it is in the same units as
        # the target (e.g. MW), so eps can be set and interpreted directly in those
        # units, and it is less sensitive than the squared error to a few extreme
        # predictions among the random samples.

        # 3e. Check if the top-k features are enough
        if deviation < eps:
            return k

    # 4. Deviation never fell below eps
    return n_features