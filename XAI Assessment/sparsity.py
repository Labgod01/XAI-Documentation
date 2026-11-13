import numpy as np

def sparsity_gini(array):
    """
    Sparsity as the Gini index of the absolute attributions
    (Chalasani et al., 2020; Hurley & Rickard, 2009).

    Code adapted from the original implementation of Chalasani et al. (2020).

    Measures how concentrated the importance is on a few features.
    It does not use the model: only the explanation vector.

    array : explanation (attributions), 1D array (n_features,)
 
    Returns G in [0, (n-1)/n]: 0 = importance spread equally over all features,
    higher = importance concentrated on fewer features.
    """

    # All values are treated equally, arrays must be 1d:
    array = np.array(array, dtype=np.float64).ravel()
    array = np.abs(array)

    # Avoid division by zero if all values are 0:
    array += 0.0000001

    # Values must be sorted:
    array = np.sort(array)

    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)

    # Number of array elements:
    n = array.shape[0]
    
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))