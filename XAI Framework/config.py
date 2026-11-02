from pathlib import Path

# The dataset, the model, the target column(s) and the task are not set here:
# they are required arguments of main.py, e.g.
#   python main.py --data data.csv --model model.joblib --target label --task classification
#
# Every setting below already has a default value: change it only if you need to.
# "Used by" lists the XAI methods affected when the value changes.
# [CLI --name] means the setting can also be changed for a single run from the
# command line, e.g. --scope local. The command line wins over this file.

# Project folder (not a setting: used to build the default paths)
base_dir = Path(__file__).resolve().parent


# =========================================================================== #
#  General settings
# =========================================================================== #

# Folder for the results: <output_dir>/plots and <output_dir>/raw_results.
# Used by: all methods.   [CLI --output-dir]
output_dir = base_dir / "outputs"

# Which explanations to produce: "local" (one row), "global" (whole dataset) or "both".
# Used by: all methods. Local only: LIME, Anchors, DiCE, ICE.
#          Global only: ALE, PFI, PDP. Both: TreeSHAP, KernelSHAP, IG.   [CLI --scope]
explanation_scope = "both"

# Which XAI methods to run.   [CLI --methods]
xai_methods = [
    "lime",
    "tree_shap",
    "kernel_shap",
    "dice",
    "anchors",
    "ale",
    "pfi",
    "pdp_ice",
    "integrated_gradients",
]

# Row of the dataset explained by the local methods (0 = first row).
# Used by: LIME, TreeSHAP, KernelSHAP, DiCE, Anchors, IG (local explanations).
#          PDP/ICE uses it only to pick the predicted class when there are 3+ classes.
#          [CLI --query-index]
query_index = 0

# Seed, so that two runs give the same results.
# Used by: LIME, Anchors, DiCE, PFI, PDP/ICE, IG (surrogate / demo network, global sample).
#          [CLI --random-state]
random_state = 42


# =========================================================================== #
#  Classification settings
# =========================================================================== #

# Which classes to explain:
#   None         -> every class, one at a time (files end with _class<label>);
#                   binary: only the positive class (the other one is its mirror image)
#   "predicted"  -> only the class predicted for the explained row
#   a label      -> only that class, e.g. 1 or "yes". With several targets it
#                   applies to the targets that have this class; the others use None
#   a dict       -> one setting per target, e.g. {"risk": 2, "flag": "predicted"};
#                   targets not in the dict use None
# Used by: LIME, TreeSHAP, KernelSHAP, PDP/ICE, IG,
#          DiCE (the classes the counterfactuals should reach; the predicted
#          class itself is replaced by every other class).
#          Not used by: Anchors (always the prediction), ALE (all classes), PFI.
#          [CLI --class-to-explain, e.g. 2, predicted or risk=2 flag=1]
class_to_explain = None

# Names of the classes, in the order of model.classes_, e.g. ["no", "yes"].
# None -> the class labels of the model. A dict per target is also accepted.
# Used by: LIME, Anchors, ALE, DiCE (labels in the plots and in the results).
class_names = None


# =========================================================================== #
#  Dataset settings
# =========================================================================== #

# Categorical features written as numbers. Leave it empty unless the dataset
# has category codes, e.g. country = 0/1/2 or contract_type = 1/2/3: the
# number is a group label, not a quantity ("bigger" means nothing).
# Example: ["country", "contract_type"]
# Text and "category" columns are always detected automatically: do not list them.
# Used by: LIME, Anchors, DiCE. (SHAP, ALE, PFI, PDP/ICE and IG treat every
# feature as a number.)
categorical_number_features = []

# Features that DiCE must never change, e.g. ["age", "sex"].
# Used by: DiCE.
immutable_features = []

# Features shown in the plots, e.g. ["age", "income"].
# None -> the 6 most important features (PDP/ICE), the first 3 features (ALE).
# Used by: PDP/ICE, ALE.
features_to_plot = None

# Metric used to measure how much the score drops when a feature is shuffled,
# e.g. "accuracy", "f1", "r2", "neg_mean_squared_error".
# None -> "accuracy" (classification) or "r2" (regression).
# Used by: PFI.
scoring_metric = None

# Description of the features for an LLM that reads the raw results, e.g.
# {"age": {"description": "Age of the customer", "unit": "years"}}.
# Used by: no method. Saved to raw_results/feature_metadata.json when not empty.
feature_metadata = {}


# =========================================================================== #
#  Integrated Gradients (IG)
# =========================================================================== #
# IG needs a PyTorch network. For any other model it trains a small network
# (surrogate) that imitates the model, and explains that network.
# All settings below are used by: IG only.

# With --demo: explain a demo neural network trained on the data (True),
# or a surrogate network (False).
ENABLE_IG_NN = True

# A PyTorch network to explain instead of the main model. None -> automatic.
ig_model = None

# Train a surrogate when the main model is not a PyTorch network.
# False -> IG stops with an error for such models.
ig_use_surrogate = True

# Number of integration steps. More steps: more accurate, slower.
ig_n_steps = 100

# Reference input that IG starts from: "zero" (all features 0) or "mean"
# (the mean of every feature).
ig_baseline = "mean"

# Global importance: compute it, on how many rows, and how many features to plot.
ig_compute_global_importance = True
ig_global_n_samples = 100
ig_global_top_k = 10


# =========================================================================== #
#  PyTorch models
# =========================================================================== #
# Used only when the model is a PyTorch nn.Module (see torch_wrapper.py).
# Used by: all methods (they decide how the network output is read).

# What the network returns: "auto", "logits", "probabilities" or
# "log_probabilities". "auto" inspects the last layer (Sigmoid/Softmax ->
# probabilities, LogSoftmax -> log-probabilities, otherwise logits).
torch_output_type = "auto"

# Class labels in the order of the network outputs, e.g. [0, 1] or ["no", "yes"].
# None -> 0 .. n_classes-1
torch_class_labels = None
