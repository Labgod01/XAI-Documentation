import numpy as np

def feature_agreement(attr_a, attr_b, k):
    """
    Feature Agreement (FA) between two explanation methods
    (Krishna et al., 2022).

    Fraction of the top-k features (by absolute attribution) that two
    explanations of the same instance have in common:

        FA(E_a, E_b, k) = | top_k(E_a) ∩ top_k(E_b) | / k

    Follows agreement_fraction(..., metric='feature') in the authors' official
    code (github.com/AI4LIFE-GROUP/disagreement-problem,
    experiments/tabular/metrics.py).

    Diagnostic, not a quality metric: there is no ground truth. A low FA means
    that at least one of the two explanations is wrong (but not which one);
    a high FA does not prove that both are right.

    attr_a, attr_b : attributions of two methods for the same instance(s),
                     1D array (n_features,) for one instance or
                     2D array (n_instances, n_features), same row order
    k              : number of top features compared

    Returns a dict:
      "per_instance" : FA of each instance, 1D array (n_instances,)
      "mean"         : mean FA over the instances, in [0, 1]
                       (1 = same top-k features, 0 = none in common)
    """
    A = np.atleast_2d(np.asarray(attr_a, dtype=float))
    B = np.atleast_2d(np.asarray(attr_b, dtype=float))

    # 1. Indices of the top-k features of each instance, by |attribution|
    top_a = np.argsort(-np.abs(A), axis=1)[:, :k]
    top_b = np.argsort(-np.abs(B), axis=1)[:, :k]

    # 2. For each instance, count the common features and divide by k
    fa = np.array([len(set(ra) & set(rb)) / k for ra, rb in zip(top_a, top_b)])

    return {"per_instance": fa, "mean": float(fa.mean())}