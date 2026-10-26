from pathlib import Path

# Project paths
base_dir = Path(__file__).resolve().parent
data_path = base_dir / "data" / "my_dataset.csv"
model_path = base_dir / "models" / "my_model.joblib"
output_dir = base_dir / "outputs"

# Execution configuration
# "demo" or "custom"
run_mode = "custom"

# Used when run_mode = "demo":
# "classification", "regression", or "ig_nn"
demo_type = "classification"

# Dataset configuration
target_col = "target"

# Task configuration: "classification" or "regression"
task_type = "classification"

# Model configuration: "sklearn" or "custom"
model_type = "sklearn"

# Explanation configuration:
# "local", "global", or "both"
explanation_scope = "local"

# XAI methods to execute
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

# General execution settings
test_size = 0.2
sample_size = 100
background_size = 100

# Global Default Configuration
scoring_metric = None               # Fallback to model's default metric
immutable_features = []             # Default: no locked features for DiCE
class_names = None                  # Default: automatic class inference
# Classification explanation target:
# None -> use the predicted class
# Otherwise -> explain the selected class label
class_to_explain = None
query_index = 0                     # Default: explain the 1st instance (iloc[[0]])
features_to_plot = None             # Default: auto-select top features for PDP/ICE
random_state = 42                   # Default seed for reproducibility
active_target = None                # Default: auto-resolved if target_col is a single string
feature_metadata = {}               # e.g., {'feature_name': {'description': '...', 'unit': '...'}}
categorical_features_override = None # If None, automatically detected via dtypes

# --- Integrated Gradients (IG) configuration ---

# True if demo PyTorch NN model is used, False if surrogate is used
ENABLE_IG_NN = True

# Optional PyTorch model specifically for Integrated Gradients
ig_model = None

# Use a PyTorch surrogate if the main model is not differentiable
ig_use_surrogate = True

# Number of integration steps
ig_n_steps = 100

# Baseline strategy: "zero" or "mean"
ig_baseline = "mean"

# Classification target class:
# None -> explain the predicted class
# Integer -> explain the specified class index
ig_target_class = None

# Global Integrated Gradients configuration
ig_global_n_samples = 100
ig_compute_global_importance = True
ig_global_top_k = 10

# Output settings
save_plots = True
save_raw_results = True
