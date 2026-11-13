import numpy as np


def proximity(x, counterfactuals, X_train, categorical_idx=None):
    """
    Proximity of counterfactual explanations (Mothilal et al., 2020).

    How close the counterfactuals are to the original instance x.

    Continuous features: negative mean distance, where each feature's absolute
    change is divided by its median absolute deviation (MAD) in the training data
        dist_cont(c, x) = (1/d_cont) * sum_p |c_p - x_p| / MAD_p
        Proximity_cont  = -(1/|C|) * sum_c dist_cont(c, x)

    Categorical features: one minus the mean fraction of features that changed
        dist_cat(c, x) = (1/d_cat) * sum_p 1[c_p != x_p]
        Proximity_cat  = 1 - (1/|C|) * sum_c dist_cat(c, x)

    The formulas follow the paper (Section 4.1). The MAD is computed as in
    DiCE's official code (github.com/interpretml/DiCE, dice_ml/data_interfaces/
    public_data_interface.py, get_mads / get_valid_mads):
    median(|values - median(values)|), replaced by 1.0 when it is 0.

    x               : original instance, 1D array (n_features,)
    counterfactuals : returned counterfactuals, 2D array (n_cfs, n_features)
    X_train         : training data, used only to compute the MAD of each feature
    categorical_idx : positions of the categorical features (default: none)

    Returns a dict with "continuous" (<= 0, closer to 0 = closer counterfactuals)
    and "categorical" (in [0, 1], higher = fewer categorical changes).
    A value is NaN when it cannot be computed (no counterfactuals, or no
    features of that type).
    """
    x = np.asarray(x, dtype=float).ravel()
    X_train = np.asarray(X_train, dtype=float)
    C = np.asarray(counterfactuals, dtype=float)
    result = {"continuous": float("nan"), "categorical": float("nan")}

    # 1. No counterfactual was returned: proximity is undefined
    if C.size == 0:
        return result
    C = np.atleast_2d(C)

    # 2. Split the features into continuous and categorical
    categorical_idx = list(categorical_idx or [])
    continuous_idx = [j for j in range(x.shape[0]) if j not in categorical_idx]

    if continuous_idx:
        # 3. MAD of each continuous feature in the training data
        values = X_train[:, continuous_idx]
        mad = np.median(np.abs(values - np.median(values, axis=0)), axis=0)
        mad[mad <= 0] = 1.0

        # 4. MAD-scaled absolute change, averaged over features, then over CFs
        dist = np.abs(C[:, continuous_idx] - x[continuous_idx]) / mad
        result["continuous"] = float(-np.mean(dist.mean(axis=1)))

    if categorical_idx:
        # 5. Fraction of categorical features changed, averaged over CFs
        changed = C[:, categorical_idx] != x[categorical_idx]
        result["categorical"] = float(1 - np.mean(changed.mean(axis=1)))

    return result