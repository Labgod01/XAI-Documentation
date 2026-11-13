import platform
import time

import numpy as np

def runtime_latency(explain_fn, X_instances, n_repeats=3, warmup=1):
    """
    Runtime Latency: wall-clock time needed to produce one explanation.

    Measures how long explain_fn takes for each instance, repeats the whole
    measurement n_repeats times, and reports summary statistics per explanation.
    As in Teixeira et al. (2025) and Liao et al. (2024), the hardware and
    software must be reported together with the times.

    explain_fn  : function that takes a 1D instance and returns its explanation
    X_instances : instances to explain, 2D array (n_instances, n_features)
    n_repeats   : how many times the full set of instances is timed
    warmup      : untimed calls before measuring (first calls are often slower
                  because of loading, caching and compilation)

    Returns a dict with the mean, median and standard deviation of the seconds
    per explanation (lower = faster), the total seconds per repeat, and the
    platform used.
    """
    X_instances = np.atleast_2d(np.asarray(X_instances, dtype=float))

    # 1. Warm-up calls, not timed
    for i in range(min(warmup, len(X_instances))):
        explain_fn(X_instances[i])

    # 2. Time every explanation, n_repeats times
    per_explanation = []
    totals = []
    for _ in range(n_repeats):
        start_total = time.perf_counter()
        for x in X_instances:
            start = time.perf_counter()
            explain_fn(x)
            per_explanation.append(time.perf_counter() - start)
        totals.append(time.perf_counter() - start_total)

    # 3. Summary statistics and the environment the times refer to
    per_explanation = np.array(per_explanation)
    return {
        "mean_s_per_explanation": float(per_explanation.mean()),
        "median_s_per_explanation": float(np.median(per_explanation)),
        "std_s_per_explanation": float(per_explanation.std()),
        "total_s_per_repeat": float(np.mean(totals)),
        "n_instances": int(len(X_instances)),
        "n_repeats": int(n_repeats),
        "platform": f"{platform.processor() or platform.machine()} | "
                    f"{platform.system()} {platform.release()} | "
                    f"Python {platform.python_version()}",
    }