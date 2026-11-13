import numpy as np


def average_rule_length(anchors):
    """
    Average rule length (Lakkaraju et al., 2016).

    Average number of conditions (predicates) per rule, i.e. "the average
    number of predicates a human reader must parse to understand a rule"
    (Section 5.2):

        AvgRuleLength = (1/|R|) * sum_{r in R} length(r)

    where length(r) is the number of predicates of rule r (Definition 3).

    Applied to Anchors: R is the set of anchors produced for the explained
    instances (one anchor per instance). length(r) is the number of
    conditions shown to the user, i.e. len(exp.names()); a range such as
    "12.00 < hour <= 18.00" is displayed as one condition and counts as one.
    No official code; the formula is the paper's.

    anchors : list with one anchor per explained instance; each anchor is the
              list of its conditions, e.g. exp.names()
              [["temp <= 9.94"], ["temp > 30.10", "5.00 < hour <= 18.00"]]

    Returns the average length (>= 0): lower = shorter, more compact rules.
    NaN if no anchors are given.
    """
    # 1. No anchors: the average is undefined
    if len(anchors) == 0:
        return float("nan")

    # 2. Length of each rule = number of its conditions (an empty anchor has 0)
    lengths = [len(conditions) for conditions in anchors]

    # 3. Mean over all rules
    return float(np.mean(lengths))