import numpy as np


def aopcr(model, x, attributions, X_background, K=10, n_samples=100,
          direction="positive", seed=42):
    """
    Feature Deletion as AOPCR - Area Over the Perturbation Curve for Regression
    (Ozyegen et al., 2022).

    Removes the top-k features of the explanation (k = 1..K) by replacing them
    with random values drawn from each feature's distribution in the data, and
    measures how much the prediction changes. The changes are averaged over k.

        AOPCR = (1/K) * sum_{k=1..K} [ f(x) - E_r[ f(x without top-k) ] ]

    Code follows the definition in the paper (Eq. 6), with a fixed number of 
    Monte Carlo samples instead of their margin-of-error stopping rule, and a
    single forecast horizon.

    model        : trained model with a .predict() method
    x            : instance to explain, 1D array (n_features,)
    attributions : explanation for x, 1D array (n_features,)
    X_background : data from which replacement values are sampled,
                   2D array (N, n_features)
    K            : number of removal steps (10 in the paper)
    n_samples    : random replacements per step, to estimate the expectation
    direction    : "positive" - remove the features that push the prediction up
                   first (score should be large and positive);
                   "negative" - remove the features that push it down first
                   (score should be large and negative)
    seed         : random seed for reproducibility

    Returns AOPCR (in target units): for "positive", higher = more faithful
    ranking; for "negative", lower (more negative) = more faithful ranking.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).ravel()
    attributions = np.asarray(attributions, dtype=float).ravel()
    X_background = np.asarray(X_background, dtype=float)
    n_features = x.shape[0]
    K = min(K, n_features)

    # 1. Original prediction f(x)
    f_x = model.predict(x.reshape(1, -1))[0]

    # 2. Ranking by signed attribution (as in the paper, positive and negative
    # contributions are evaluated separately)
    if direction == "positive":
        order = np.argsort(-attributions)   # largest positive first
    elif direction == "negative":
        order = np.argsort(attributions)    # most negative first
    else:
        raise ValueError("direction must be 'positive' or 'negative'")

    total = 0.0
    for k in range(1, K + 1):
        top_k = order[:k]

        # 3. Copies of x where the top-k features get random values from the
        # data, sampled independently for each feature (marginal distribution)
        Z = np.tile(x, (n_samples, 1))
        rows = rng.integers(0, len(X_background), size=(n_samples, k))
        Z[:, top_k] = X_background[rows, top_k]

        # 4. Change of the prediction after removing the top-k features
        total += f_x - np.mean(model.predict(Z))

    # 5. Average over k = 1..K
    return float(total / K)