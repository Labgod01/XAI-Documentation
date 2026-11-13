import numpy as np

def validity(model, counterfactuals, desired_range, k_requested):
    """
    Validity (%Valid-CFs) of counterfactual explanations (Mothilal et al., 2020).

    Fraction of the requested counterfactuals that are unique and actually
    reach the desired outcome:

        Validity = |{unique c in C : f(c) in desired range}| / k

    The validity check for regression follows DiCE's official code
    (github.com/interpretml/DiCE, dice_ml/explainer_interfaces/explainer_base.py,
    decide_cf_validity):
    a counterfactual is valid when low <= f(c) <= high. The paper defines
    validity for classification (f(c) > 0.5); the desired range is its
    regression counterpart.

    model           : trained model with a .predict() method
    counterfactuals : returned counterfactuals, 2D array (n_cfs, n_features);
                      may have fewer rows than requested, or none
    desired_range   : (low, high) target range for the prediction
    k_requested     : number of counterfactuals that were asked for
                      (total_CFs in DiCE)

    Returns validity in [0, 1]: higher = better (1 = all k requested
    counterfactuals were found, are different and reach the desired range).
    """
    low, high = desired_range
    C = np.asarray(counterfactuals, dtype=float)

    # 1. No counterfactual found at all
    if C.size == 0:
        return 0.0
    C = np.atleast_2d(C)

    # 2. Keep only unique counterfactuals (duplicates count once, as in the paper)
    C = np.unique(C, axis=0)

    # 3. Is the model's prediction for each counterfactual inside the desired range?
    preds = model.predict(C)
    valid = (preds >= low) & (preds <= high)

    # 4. Divide by the number requested, not by the number returned
    return float(valid.sum() / k_requested)