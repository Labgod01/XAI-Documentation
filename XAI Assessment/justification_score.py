import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

def is_justified(model, counterfactual, X_train, y_train, desired_range, n_ball=15000, seed=42):
    """
    Justification of one counterfactual (Laugel et al., 2019).

    A counterfactual is justified when it is connected, through a region where
    the model keeps giving the same outcome, to a real training instance that
    has that outcome and is correctly predicted. An unjustified counterfactual
    lies in an isolated pocket of the model that no real data supports.

    Follows evaluation_test() in the authors' official code
    (github.com/thibaultlaugel/truce, detection_evaluation.py):
      1. find the closest "ally" training instance b0 of the counterfactual;
      2. sample n_ball points uniformly in the ball centred on the counterfactual
         with radius d(counterfactual, b0);
      3. keep the points with the same outcome as the counterfactual;
      4. cluster them with DBSCAN (min_samples=2, eps = largest nearest-neighbour
         distance); justified if the counterfactual and b0 share a cluster.

    Adaptation to regression (the paper is defined for classification):
    "same class" becomes "prediction inside the desired range", and an ally is
    a training instance whose true target and prediction are both inside it.
    Distances are computed on standardised features, as in the authors' code.

    model          : trained model with a .predict() method
    counterfactual : one counterfactual, 1D array (n_features,)
    X_train        : training features, 2D array (N, n_features)
    y_train        : training targets, 1D array (N,)
    desired_range  : (low, high) target range of the counterfactual
    n_ball         : number of sampled points (15000 in the authors' code)
    seed           : random seed for reproducibility

    Returns 1 if the counterfactual is justified, 0 otherwise.
    """
    rng = np.random.default_rng(seed)
    low, high = desired_range
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    cf = np.asarray(counterfactual, dtype=float).ravel()
    n_features = cf.shape[0]

    # 0. Standardise, so that distances do not depend on the units of the features
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    def to_std(A):
        return (A - mean) / std

    def predict_std(A_std):
        return model.predict(A_std * std + mean)

    def in_range(values):
        return (values >= low) & (values <= high)

    # 1. Allies: training instances whose true target and prediction are both
    # inside the desired range ("correctly predicted, same class" in the paper)
    allies = X_train[in_range(y_train) & in_range(model.predict(X_train))]
    if len(allies) == 0:
        return 0

    # 2. Closest ally b0 and its distance to the counterfactual (the ball radius)
    cf_std = to_std(cf)
    allies_std = to_std(allies)
    distances = np.linalg.norm(allies_std - cf_std, axis=1)
    b0_std = allies_std[np.argmin(distances)]
    radius = distances.min()

    # 3. Points sampled uniformly inside the ball around the counterfactual
    directions = rng.standard_normal((n_ball, n_features))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radius * rng.uniform(0, 1, n_ball) ** (1 / n_features)
    ball = cf_std + directions * radii[:, None]

    # 4. Keep the points with the same outcome as the counterfactual
    same_outcome = ball[in_range(predict_std(ball))]
    points = np.vstack([cf_std, b0_std, same_outcome])

    # 5. eps = largest distance from a point to its nearest neighbour
    nn = NearestNeighbors(n_neighbors=2).fit(points)
    nearest = nn.kneighbors(points)[0][:, 1]
    eps = nearest.max()

    # 6. Justified if the counterfactual (row 0) and b0 (row 1) share a cluster
    labels = DBSCAN(eps=eps, min_samples=2).fit(points).labels_
    return int(labels[0] == labels[1] and labels[0] != -1)


def justification_score(model, counterfactuals, X_train, y_train, desired_range,
                        n_ball=15000, seed=42):
    """
    Justification score: fraction of the given counterfactuals that are justified.

    counterfactuals : 2D array (n_cfs, n_features)

    Returns a value in [0, 1] (higher = better), or NaN if there are no
    counterfactuals.
    """
    C = np.asarray(counterfactuals, dtype=float)
    if C.size == 0:
        return float("nan")
    C = np.atleast_2d(C)
    flags = [is_justified(model, c, X_train, y_train, desired_range, n_ball, seed) for c in C]
    return float(np.mean(flags))