import numpy as np

def neighbourhood_fidelity(model, surrogate_fn, x, X_background, sigma=0.1, n_samples=100, seed=42):
    """
    Neighbourhood-fidelity (NF) (Plumb et al., 2019).

    How well the local surrogate model g_x of an explanation (e.g. LIME's linear
    model) reproduces the real model f on points x' sampled around x:
        NF = E_{x' ~ N_x} [ (g_x(x') - f(x'))^2 ]

    Follows the "causal metric" of metrics_lime() in the authors' official code
    (github.com/GDPlumb/ExpO, Code/ExplanationMetrics.py): Gaussian neighbours
    x' = x + sigma * noise, squared error between surrogate and model.
    Adapted to unstandardised tabular data: the noise of each feature is scaled
    by its standard deviation (the authors standardise the data first). The
    authors use 5 neighbours per instance; more give a more stable estimate.

    model        : trained model with a .predict() method
    surrogate_fn : the explanation's local model g_x; takes a 2D array of
                   points and returns its predictions (see lime_surrogate below)
    x            : instance to explain, 1D array (n_features,)
    X_background : reference data, used only for the feature standard deviations
    sigma        : neighbourhood size, in standard deviations of each feature
    n_samples    : number of neighbours x'
    seed         : random seed for reproducibility

    Returns NF >= 0 (in squared target units): lower = more faithful surrogate.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).ravel()
    X_background = np.asarray(X_background, dtype=float)

    # 1. Neighbours x' = x + Gaussian noise, scaled per feature
    std = X_background.std(axis=0)
    X_neigh = x + sigma * std * rng.standard_normal((n_samples, x.shape[0]))

    # 2. Real model and surrogate on the same neighbours
    f_pred = model.predict(X_neigh)
    g_pred = surrogate_fn(X_neigh)

    # 3. Mean squared difference
    return float(np.mean((g_pred - f_pred) ** 2))


def lime_surrogate(lime_explainer, exp, x, label=1):
    """
    Builds LIME's local linear model g_x as a function of points in the
    original feature space, so it can be compared with the real model.

    Works for both LIME modes:
      - discretize_continuous=False (as in the authors' code): g is linear in
        the standardised features;
      - discretize_continuous=True (LIME's default, used in the framework):
        g is linear in binary features "x'_j falls in the same bin as x_j".

    lime_explainer : the LimeTabularExplainer that produced exp
    exp            : the explanation returned by explain_instance
                     (ask for all features: num_features = n_features)
    x              : the explained instance, 1D array
    label          : output label (1 for regression in LIME)
    """
    x = np.asarray(x, dtype=float).ravel()
    weights = np.zeros(x.shape[0])
    for j, w in exp.local_exp[label]:
        weights[j] = w
    intercept = exp.intercept[label]

    if lime_explainer.discretizer is None:
        # Continuous mode: features are standardised with LIME's own scaler
        mean = lime_explainer.scaler.mean_
        scale = lime_explainer.scaler.scale_

        def g(Z):
            return intercept + ((np.atleast_2d(Z) - mean) / scale) @ weights
    else:
        # Discretised mode: 1 if the feature is in the same bin as x, else 0
        disc = lime_explainer.discretizer
        x_bins = disc.discretize(x.reshape(1, -1))[0]

        def g(Z):
            same_bin = (disc.discretize(np.atleast_2d(Z)) == x_bins).astype(float)
            return intercept + same_bin @ weights

    return g