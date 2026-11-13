import re
import numpy as np

def anchor_precision_cover(model, x, X_test, conditions):
    """
    Precision and Coverage of an anchor, measured on held-out test data
    (Ribeiro et al., 2018).

    An anchor A is a rule "IF conditions THEN prediction = f(x)".

        Precision = (1/|T_A|) * sum_{z in T_A} 1[ f(z) = f(x) ]
        Coverage  = |T_A| / |T|

    where T is the test set and T_A the test instances that satisfy all the
    conditions of the anchor.

    Follows the evaluation in the authors' official notebook
    (github.com/marcotcr/anchor, notebooks/"Anchor on tabular data.ipynb"):
    find the test instances where the anchor applies, then compare their
    predictions with the prediction for x. This is independent of
    exp.precision() / exp.coverage(), which the explainer estimates on its own
    perturbation samples. Classification only.

    model      : trained classifier with a .predict() method
    x          : explained instance, 1D array (n_features,)
    X_test     : held-out test data, 2D array (N, n_features)
    conditions : the anchor as a list of (feature_index, operator, value),
                 operator in {"<=", ">", "<", ">=", "=="}
                 e.g. [(0, ">", 30.0), (3, ">", 12.0), (3, "<=", 18.0)]

    Returns a dict:
      "precision" : in [0, 1], higher = the rule is more faithful to the model
                    (NaN if the anchor applies to no test instance)
      "coverage"  : in [0, 1], higher = the rule applies to more instances
      "n_covered" : number of test instances where the anchor applies
    """
    x = np.asarray(x, dtype=float).ravel()
    X_test = np.asarray(X_test, dtype=float)

    ops = {
        "<=": np.less_equal,
        "<": np.less,
        ">": np.greater,
        ">=": np.greater_equal,
        "==": np.isclose,
    }

    # 1. Which test instances satisfy every condition of the anchor?
    applies = np.ones(X_test.shape[0], dtype=bool)
    for j, op, value in conditions:
        applies &= ops[op](X_test[:, j], value)

    n_covered = int(applies.sum())

    # 2. Coverage: fraction of the test set where the anchor applies
    coverage = n_covered / X_test.shape[0]

    # 3. Precision: among those, fraction with the same prediction as x
    if n_covered == 0:
        precision = float("nan")
    else:
        pred_x = model.predict(x.reshape(1, -1))[0]
        precision = float(np.mean(model.predict(X_test[applies]) == pred_x))

    return {"precision": precision, "coverage": float(coverage), "n_covered": n_covered}


def parse_anchor_names(names, feature_names):
    """
    Turns the anchor returned by the anchor library (exp.names()) into the
    list of conditions used by anchor_precision_cover.

    Handles the three forms of numerical conditions the library writes,
        "temp > 30.12",  "temp <= 18.50",  "12.00 < hour <= 18.00",
    and categorical equalities "season = 2.00".
    Note: the library rounds thresholds to 2 decimals in the names.
    """
    index = {name: j for j, name in enumerate(feature_names)}
    number = r"(-?[\d.]+(?:e-?\d+)?)"
    conditions = []
    for name in names:
        name = name.strip()
        between = re.fullmatch(number + r"\s*<\s*(.+?)\s*<=\s*" + number, name)
        single = re.fullmatch(r"(.+?)\s*(<=|>=|<|>|=)\s*" + number, name)
        if between:
            low, feat, high = between.groups()
            conditions.append((index[feat], ">", float(low)))
            conditions.append((index[feat], "<=", float(high)))
        elif single:
            feat, op, value = single.groups()
            conditions.append((index[feat], "==" if op == "=" else op, float(value)))
        else:
            raise ValueError(f"Cannot parse anchor condition: {name!r}")
    return conditions