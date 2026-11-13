from itertools import combinations

import numpy as np

def stability_vsi(explain_fn, top_p, n_runs=10, seeds=None):
    """
    Stability across repeated runs: Variables Stability Index (VSI)
    (Visani et al., 2022).

    Runs the same explainer n_runs times with different random seeds and checks
    whether it selects the same top-p features each time. For every pair of
    runs, counts how many of the top-p features they have in common:

        VSI = sum_{pairs (a, b)} |F_a ∩ F_b|  /  ( p * C(n_runs, 2) )

    Follows the VSI computation in the authors' official code (package
    lime-stability, utils.compare_confints). Generalised from LIME to any
    explainer: F_r is the set of the top-p features by |attribution| in run r
    (in LIME, F_r is the set of features selected by the explainer).

    explain_fn : function that takes a random seed (int) and returns the
                 explanation as a 1D array (n_features,). For a local method it
                 explains a fixed instance x; for a global method (e.g. PFI)
                 it explains the whole model.
    top_p      : number of most important features compared in each run
    n_runs     : number of repeated runs (10 in the paper)
    seeds      : optional list of seeds; default 0, 1, ..., n_runs - 1

    Returns VSI in [0, 100]: 100 = the same top-p features in every run
    (most stable); lower = the selected features change between runs.
    """
    if seeds is None:
        seeds = list(range(n_runs))

    # 1. Run the explainer once per seed and keep the top-p features of each run
    top_sets = []
    for seed in seeds:
        attributions = np.asarray(explain_fn(seed), dtype=float).ravel()
        top = np.argsort(-np.abs(attributions))[:top_p]
        top_sets.append(set(top.tolist()))

    # 2. For every pair of runs, count the common features
    same_vars = 0
    n_pairs = 0
    for set_a, set_b in combinations(top_sets, 2):
        same_vars += len(set_a & set_b)
        n_pairs += 1

    # 3. Normalise by the maximum possible (p common features in every pair)
    return float(same_vars / (n_pairs * top_p) * 100)