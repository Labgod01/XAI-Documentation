import numpy as np


def sensitivity_n(model, x, attributions, baseline, n_values=None, n_subsets=100, seed=42):
    """
    Sensitivity-n (Ancona et al., 2018).

    For each subset size n, removes random subsets of n features (sets them to
    a baseline) and computes the Pearson correlation between
      - the sum of the attributions of the removed features, and
      - the actual drop in the model's output.
    An explanation satisfies Sensitivity-n perfectly when the correlation is 1.

    Code follows the definition in the aforementioned paper:
    100 random subsets per n, n from 1 to about 80% of the features, removed
    features set to a baseline (zero in the paper; here a user-given vector,
    e.g. the training mean, because zero is not meaningful for tabular data).


    model        : trained model with a .predict() method
    x            : instance to explain, 1D array (n_features,)
    attributions : explanation for x, 1D array (n_features,)
    baseline     : value given to a removed feature, 1D array (n_features,)
    n_values     : subset sizes to evaluate; default 1 .. 80% of the features
    n_subsets    : random subsets per size n (100 in the paper)
    seed         : random seed for reproducibility

    Returns a dict {n: Pearson correlation in [-1, 1]}: higher = better.
    The value is NaN when the correlation is undefined (e.g. all sums equal).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).ravel()
    attributions = np.asarray(attributions, dtype=float).ravel()
    baseline = np.asarray(baseline, dtype=float).ravel()
    n_features = x.shape[0]

    # 1. Subset sizes: from 1 to about 80% of the features (as in the paper)
    if n_values is None:
        n_values = range(1, max(1, int(0.8 * n_features)) + 1)

    # 2. Original prediction f(x)
    f_x = model.predict(x.reshape(1, -1))[0]

    results = {}
    for n in n_values:
        # 3. Random subsets of exactly n features: one 0/1 mask per row
        masks = np.zeros((n_subsets, n_features), dtype=bool)
        for i in range(n_subsets):
            masks[i, rng.choice(n_features, size=n, replace=False)] = True

        # 4. Perturbed inputs: removed features take the baseline value
        Z = np.tile(x, (n_subsets, 1))
        Z[masks] = np.tile(baseline, (n_subsets, 1))[masks]

        # 5. Actual drop of the output and sum of the removed attributions
        pred_diff = f_x - model.predict(Z)
        att_sum = masks.astype(float) @ attributions

        # 6. Pearson correlation between the two (undefined if one is constant)
        if np.std(pred_diff) == 0 or np.std(att_sum) == 0:
            results[n] = float("nan")
        else:
            results[n] = float(np.corrcoef(att_sum, pred_diff)[0, 1])

    return results