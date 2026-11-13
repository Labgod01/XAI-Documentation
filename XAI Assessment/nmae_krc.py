from itertools import combinations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# 1. Ground truth ("real feature importance", Liao et al., 2024, Section 3)
# ---------------------------------------------------------------------------

def global_ablation_importance(model, X_train, y_train, X_test, y_test,
                               error_fn=mean_squared_error, model_factory=None):
    """
    Real global feature importance by feature ablation (Liao et al., 2024, Fig. 5).

    Train the model with all features and measure the test error e_all.
    Then, for each feature i, remove it, retrain the model with the remaining
    features and measure the test error again, e_{-i}:

        FI_real,i = | e_{-i} - e_all |

    The absolute value is our choice (the paper says "the difference between
    the two forecast errors"); it follows the PFI definition of the paper
    (Eq. 3), so that a feature whose removal slightly improves the model does
    not get a negative importance.

    model         : model with the hyperparameters to use (it is cloned, not modified)
    X_train, y_train, X_test, y_test : data splits (2D / 1D arrays)
    error_fn      : error metric G(y_true, y_pred); use the same as PFI
    model_factory : optional function returning a new untrained model
                    (default: sklearn.base.clone(model); needed for non-sklearn models)

    Returns a 1D array (n_features,) of importances >= 0.
    Cost: one retraining per feature.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    make = model_factory or (lambda: clone(model))

    # Error with all features
    m_all = make().fit(X_train, y_train)
    e_all = error_fn(y_test, m_all.predict(X_test))

    # Error without each feature, after retraining
    importances = np.zeros(X_train.shape[1])
    for i in range(X_train.shape[1]):
        keep = [j for j in range(X_train.shape[1]) if j != i]
        m_i = make().fit(X_train[:, keep], y_train)
        e_i = error_fn(y_test, m_i.predict(X_test[:, keep]))
        importances[i] = abs(e_i - e_all)

    return importances


def local_perturbation_importance(model, x, y_true, X_background, sigma=0.1,
                                  n_samples=100, error_fn=mean_squared_error, seed=42):
    """
    Real feature importance for one instance by perturbation
    (Liao et al., 2024, Fig. 6).

    For each feature i, keep the other features of x fixed, add small random
    perturbations only to feature i, and measure the average forecast error
    of the perturbed samples:

        FI_real,i(x) = | mean error(perturbed samples) - error(x) |

    The perturbation size is not given in the paper; we use Gaussian noise
    with standard deviation sigma * std of the feature, as in NF.
    The absolute value is our choice, as in global_ablation_importance.

    model        : trained model with a .predict() method
    x            : instance, 1D array (n_features,)
    y_true       : true target value of x
    X_background : reference data, used only for the feature standard deviations
    sigma        : perturbation size, in standard deviations of each feature
    n_samples    : number of perturbed samples per feature
    error_fn     : error metric G(y_true, y_pred)
    seed         : random seed for reproducibility

    Returns a 1D array (n_features,) of importances >= 0.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).ravel()
    std = np.asarray(X_background, dtype=float).std(axis=0)

    # Error on the original instance
    e_x = error_fn([y_true], model.predict(x.reshape(1, -1)))

    importances = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        # Copies of x where only feature i is perturbed
        Z = np.tile(x, (n_samples, 1))
        Z[:, i] += sigma * std[i] * rng.standard_normal(n_samples)
        e_z = error_fn(np.full(n_samples, y_true), model.predict(Z))
        importances[i] = abs(e_z - e_x)

    return importances


# ---------------------------------------------------------------------------
# 2. Importance from a PDP (or ALE) curve (Liao et al., 2024, Eq. 7-8)
# ---------------------------------------------------------------------------

def pdp_slope_importance(grid, curve, feature_range=1.0):
    """
    Feature importance from a PDP curve: the average absolute slope.

        SL_k = (PDP_{k+1} - PDP_k) / (x_{k+1} - x_k)
        FI   = (1/m) * sum_k |SL_k|          (m intervals)

    Applied to ALE curves as well (our extension; the paper uses PDP only).

    A slope is "target units per unit of the feature", so features measured in
    different units (e.g. degrees vs MW) are not comparable. The paper uses
    min-max normalised features; equivalently, we multiply the slope by the
    feature's range (max - min in the training data), i.e. the slope per
    "whole range" of the feature.

    grid          : feature values of the curve, 1D array (m + 1,)
    curve         : PDP (or ALE) values at the grid points, 1D array (m + 1,)
    feature_range : max - min of the feature in the training data
                    (default 1.0 = features already min-max normalised)

    Returns one importance value >= 0 (in target units).
    """
    grid = np.asarray(grid, dtype=float).ravel()
    curve = np.asarray(curve, dtype=float).ravel()
    slopes = np.diff(curve) / np.diff(grid) * feature_range
    return float(np.mean(np.abs(slopes)))


# ---------------------------------------------------------------------------
# 3. The metric: NMAE and KRC (Liao et al., 2024, Eq. 13-14)
# ---------------------------------------------------------------------------

def nmae_krc(fi_real, fi_xai):
    """
    Agreement between the real and the estimated feature importance.

        NMAE = (1/M) * sum_i | FI_real,i - FI_xai,i |
        KRC  = (n_c - n_d) / (0.5 * M * (M - 1))

    n_c / n_d: number of feature pairs ordered in the same / opposite way by
    the two importance vectors (pairs tied in either vector count in neither).

    Before the NMAE, both vectors are made comparable (our choice; the paper
    does not specify the normalisation): absolute values (importance is a
    magnitude, the sign of SHAP/LIME/IG is ignored), then division by their
    sum, so each value is the share of the total importance.
    The KRC depends only on the order, so it needs no normalisation.

    fi_real : real importance (ground truth), 1D array (M,)
    fi_xai  : importance given by the XAI method, 1D array (M,)

    Returns a dict:
      "nmae" : in [0, 2/M], lower = closer to the real importance
      "krc"  : in [-1, 1], 1 = same ranking, 0 = unrelated, -1 = reversed
    """
    real = np.abs(np.asarray(fi_real, dtype=float).ravel())
    xai = np.abs(np.asarray(fi_xai, dtype=float).ravel())
    M = real.shape[0]

    # NMAE on the normalised vectors (shares of the total importance)
    real_n = real / real.sum() if real.sum() > 0 else real
    xai_n = xai / xai.sum() if xai.sum() > 0 else xai
    nmae = float(np.mean(np.abs(real_n - xai_n)))

    # KRC: compare the order of every pair of features
    n_c = n_d = 0
    for a, b in combinations(range(M), 2):
        s = np.sign(real[a] - real[b]) * np.sign(xai[a] - xai[b])
        if s > 0:
            n_c += 1
        elif s < 0:
            n_d += 1
    krc = (n_c - n_d) / (0.5 * M * (M - 1))

    return {"nmae": nmae, "krc": float(krc)}