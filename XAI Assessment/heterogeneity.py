import numpy as np


def ice_heterogeneity(ice):
    """
    Heterogeneity of ICE curves around the PDP (Goldstein et al., 2015;
    Herbinger et al., 2022).

    Measures how far the individual (ICE) curves of a feature are from their
    average (the PDP), after centering, i.e. how well the PDP represents the
    individual effects. Centering removes differences in level between the
    curves, so only differences in shape (interactions) remain.

    1. Center each ICE curve by its mean over the grid:
           ice_c[i, k] = ice[i, k] - (1/m) * sum_k' ice[i, k']
    2. Centered PDP = mean of the centered ICE curves at each grid point.
    3. Sum of squared deviations (R_L2 in Herbinger et al., 2022, Section 3):
           R_L2 = sum_k sum_i ( ice_c[i, k] - pdp_c[k] )^2

    Follows the SS_L2 objective and the centering in the authors' official
    code (github.com/JuliaHerbinger/repid, R/tree_splitting.R:
    "Y = Y - rowMeans(Y)" and "sum(t((t(y) - colMeans(y))^2))"), applied to
    the whole dataset (one region, no splitting).

    R_L2 grows with the number of curves N and grid points m, so the mean
    R_L2 / (N * m) is also returned, to compare features or runs with a
    different N or m (our addition).

    ice : ICE predictions of one feature, 2D array (N curves, m grid points),
          e.g. sklearn's partial_dependence(..., kind="individual")["individual"][0]

    Returns a dict:
      "r_l2" : R_L2 >= 0, as in the paper (squared target units)
      "mean" : R_L2 / (N * m) >= 0
    Lower = more homogeneous curves, the PDP is more representative;
    0 = no interaction of the feature with the others.
    """
    ice = np.asarray(ice, dtype=float)
    n_curves, n_grid = ice.shape

    # 1. Center each ICE curve (row) by its own mean over the grid
    ice_c = ice - ice.mean(axis=1, keepdims=True)

    # 2. Centered PDP: average of the centered curves at each grid point
    pdp_c = ice_c.mean(axis=0)

    # 3. Squared deviations of the curves from the PDP, summed
    r_l2 = float(np.sum((ice_c - pdp_c) ** 2))

    return {"r_l2": r_l2, "mean": r_l2 / (n_curves * n_grid)}