import numpy as np

def infidelity(model, x, attributions, baseline, n_samples=1000, seed=42):
    """
    Infidelity (Yeh et al., 2019) with binary subset-removal perturbations.

    Randomly removes subsets of features (sets them to a baseline) and checks
    whether the sum of the attributions of the removed features predicts the
    actual change in the model's output. The explanation is optimally scaled
    (beta) before the error is computed, so only the relative attributions matter.

    Follows the logic of get_exp_infid() in the authors' official code
    (github.com/chihkuanyeh/saliency_evaluation), adapted from images/PyTorch
    classification to tabular regression with any model that has .predict().

    model        : trained model with a .predict() method
    x            : instance to explain, 1D array (n_features,)
    attributions : explanation for x (contribution type: SHAP, IG, LIME),
                   1D array (n_features,)
    baseline     : value given to a removed feature, 1D array (n_features,),
                   e.g. the mean of the training data
    n_samples    : number of random subsets (perturbations)
    seed         : random seed for reproducibility

    Returns infidelity >= 0 (in squared target units): lower = more faithful.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).ravel()
    attributions = np.asarray(attributions, dtype=float).ravel()
    baseline = np.asarray(baseline, dtype=float).ravel()
    n_features = x.shape[0]

    if n_features < 2:
        raise ValueError("Infidelity with subset removal needs at least 2 features.")

    # 1. Original prediction f(x)
    f_x = model.predict(x.reshape(1, -1))[0]

    # 2. Random subsets: each feature is removed with probability 0.5.
    # As in the authors' code, empty and full subsets are re-sampled,
    # because they carry no information about individual features.
    masks = rng.random((n_samples, n_features)) < 0.5
    n_removed = masks.sum(axis=1)
    bad = (n_removed == 0) | (n_removed == n_features)
    while bad.any():
        masks[bad] = rng.random((bad.sum(), n_features)) < 0.5
        n_removed = masks.sum(axis=1)
        bad = (n_removed == 0) | (n_removed == n_features)

    # 3. Perturbed inputs: removed features take the baseline value
    Z = np.tile(x, (n_samples, 1))
    Z[masks] = np.tile(baseline, (n_samples, 1))[masks]

    # 4. Actual change in the output: f(x) - f(x - I)
    pred_diff = f_x - model.predict(Z)

    # 5. Change predicted by the explanation: I^T a, where I is the 0/1 mask,
    # i.e. the sum of the attributions of the removed features
    exp_sum = masks.astype(float) @ attributions

    # 6. Optimal scaling beta (from the authors' code), so that the overall
    # scale of the attributions does not affect the score
    denom = np.mean(exp_sum ** 2)
    beta = np.mean(pred_diff * exp_sum) / denom if denom > 0 else 0.0
    exp_sum = beta * exp_sum

    # 7. Infidelity: mean squared error between actual and predicted change
    return float(np.mean((pred_diff - exp_sum) ** 2))