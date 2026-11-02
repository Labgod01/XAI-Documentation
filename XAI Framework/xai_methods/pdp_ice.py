import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.inspection import PartialDependenceDisplay
from sklearn.base import BaseEstimator, ClassifierMixin
from multi_output import class_suffix, to_json_value


class _SwappedBinaryClassifier(ClassifierMixin, BaseEstimator):
    """
    A binary classifier with its two classes swapped. sklearn's PDP/ICE always
    plot the probability of classes_[1]; this lets them plot classes_[0].
    """

    def __init__(self, model):
        self.model = model
        self.classes_ = np.asarray(model.classes_)[::-1]

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return np.asarray(self.model.predict_proba(X))[:, ::-1]

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X=None, y=None):
        # sklearn checks that a fit method exists; the model is already trained
        raise NotImplementedError("Wraps an already trained model.")


def run_pdp_ice(unified_payload, prepared_inputs, output_dir=None, df=None):
    """Generate PDP/ICE explanations for the active target."""

    model = prepared_inputs["model"]
    feature_names = prepared_inputs["feature_names"]
    reference_dataset = prepared_inputs["reference_dataset"]
    X_np = prepared_inputs["X_np"]
    task_type = prepared_inputs["task_type"]
    metadata = prepared_inputs["metadata"]
    current_target = metadata["active_target"]
    scope = unified_payload["scope"] 
    # targets = metadata.get("targets", [current_target])
    targets = [current_target]

    if df is None:
        raise ValueError(
            "The original DataFrame must be provided to run_pdp_ice()."
        )
    
    output_dir = Path(output_dir or "outputs")
    raw_dir = output_dir / "raw_results"
    plots_dir = output_dir / "plots"

    raw_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Store PDP/ICE explanation objects and raw values separately
    pdp_ice_explanations = {}
    pdp_ice_raw_values = {}

    # PDP is a global explanation, ICE is a local one
    run_pdp = scope in ["global", "both"]
    run_ice = scope in ["local", "both"]

    if scope == "local":
        print(
            "PDP/ICE: Only ICE supports local explanations. "
            "Skipping PDP for local scope."
            "Generating local explanations only; global explanation skipped."

        )
    if scope == "global":
        print(
            "PDP/ICE: Only PDP supports global explanations. "
            "Skipping ICE for global scope."
            "Generating global explanations only; local explanation skipped."
        )
    if scope == "both":
        print(
            "PDP/ICE:\n "
            "PDP supports only global explanations."
            "ICE supports only local explanations."
            "Generating both local and global explanations."
        )
    for current_target in targets:

        # Update unified payload for the active target
        metadata["active_target"] = current_target
        unified_payload["target"] = df[current_target]

        # if unified_payload["scope"] in ["global", "both"]:
        if run_pdp or run_ice:

            # 1. Select features
            configured_features = unified_payload["metadata"]["features_to_plot"]
            if configured_features is not None:
                selected_features = [
                    feature
                    for feature in configured_features
                    if feature in feature_names
                ]
            else:
                # Automatically select the most important features
                if hasattr(model, "feature_importances_"):
                    importance_df = pd.DataFrame({
                        "feature": feature_names,
                        "importance": model.feature_importances_
                    }).sort_values(by="importance", ascending=False)
                    selected_features = (importance_df.head(6)["feature"].tolist())
                else:
                    selected_features = feature_names[:6]

            print("\n================ PDP / ICE ================")
            print(f"Target: {current_target}")
            print(f"Task type: {task_type}")
            print(f"Selected features: {selected_features}")

            # 2. Classes to explain: one PDP/ICE per class (classification),
            # one for regression. sklearn needs target= for 3+ classes and
            # always plots classes_[1] for 2 classes.
            X_pdp = reference_dataset.astype(float)
            classes_to_explain = prepared_inputs["classes_to_explain"] or [(None, None)]
            classes = (
                model.classes_.tolist()
                if task_type == "classification"
                and hasattr(model, "classes_")
                else None
            )
            ice_features = selected_features[:2]

            for explained_class, class_idx in classes_to_explain:
                suffix = class_suffix(explained_class, len(classes_to_explain))
                title_class = f" - class {explained_class}" if explained_class is not None else ""

                estimator = model
                common_parameters = {
                    "grid_resolution": 20,
                    "subsample": 50,
                    "random_state": unified_payload["metadata"]["random_state"]
                }
                if task_type == "classification":
                    common_parameters["response_method"] = "predict_proba"
                    if len(model.classes_) > 2:
                        common_parameters["target"] = explained_class
                    elif class_idx == 0:
                        # Binary: plot the first class by swapping the two classes
                        estimator = _SwappedBinaryClassifier(model)
                    print(f"Explained class: {explained_class}")

                pdp_display = None
                ice_display = None

                # 3. Partial Dependence Plot (PDP) - global
                if run_pdp:
                    print("\nComputing Partial Dependence plots...")

                    pdp_display = PartialDependenceDisplay.from_estimator(
                        estimator=estimator,
                        X=X_pdp,
                        features=selected_features,
                        kind="average",
                        **common_parameters
                    )
                    pdp_display.figure_.suptitle(
                        f"Partial Dependence Plots (PDP){title_class}",
                        fontsize=16
                    )
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"pdp_{current_target}{suffix}.png", dpi=300)
                    plt.close("all")

                # 4. Individual Conditional Expectation (ICE) - local
                # Only the two most important features, to keep the plot readable
                if run_ice:
                    print("\nComputing ICE plots...")

                    ice_display = PartialDependenceDisplay.from_estimator(
                        estimator=estimator,
                        X=X_pdp,
                        features=ice_features,
                        kind="individual",
                        centered=False,
                        **common_parameters
                    )
                    ice_display.figure_.suptitle(
                        f"Individual Conditional Expectation (ICE){title_class}",
                        fontsize=16
                    )
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"ice_{current_target}{suffix}.png", dpi=300)
                    plt.close("all")

                # 5. Store explanations and extract raw values
                pdp_ice_explanations[f"{current_target}{suffix}"] = {
                    "selected_features": selected_features,
                    "ice_features": ice_features,
                    "pdp_display": pdp_display,
                    "ice_display": ice_display,
                    "task_type": task_type
                }

                # Extract raw PDP and ICE data from sklearn displays
                pdp_raw = {}
                ice_raw = {}

                if run_pdp:
                    for i, feature in enumerate(selected_features):
                        pdp_raw[feature] = {
                            "grid_values": [
                                float(v) for v in
                                pdp_display.pd_results[i]["grid_values"][0]
                            ],
                            "average_predictions": [
                                np.asarray(v).tolist()
                                for v in pdp_display.pd_results[i]["average"]
                            ]
                        }

                if run_ice:
                    for i, feature in enumerate(ice_features):
                        ice_raw[feature] = {
                            "grid_values": [
                                float(v) for v in
                                ice_display.pd_results[i]["grid_values"][0]
                            ],
                            "individual_predictions": np.asarray(
                                ice_display.pd_results[i]["individual"]
                            ).tolist()
                        }

                pdp_ice_raw_values[f"{current_target}{suffix}"] = {
                    "method": "PDP_ICE",
                    "target": str(current_target),
                    "task_type": task_type,
                    "classes": classes,
                    "explained_class": to_json_value(explained_class),
                    "pdp": pdp_raw,
                    "ice": ice_raw
                }

                print("\n--- PDP / ICE Raw Values Summary ---")
                print("Target:", current_target)
                print("Task type:", task_type)
                print("PDP features:", list(pdp_raw.keys()))
                print("ICE features:", list(ice_raw.keys()))

                with open(
                    raw_dir / f"pdp_ice_{current_target}{suffix}.json",
                    "w",
                    encoding="utf-8"
                ) as f:
                    json.dump(
                        pdp_ice_raw_values[f"{current_target}{suffix}"],
                        f, indent=4, ensure_ascii=False
                    )

                for feature, values in pdp_raw.items():
                    print(
                        f"PDP {feature}: "
                        f"{len(values['grid_values'])} grid points"
                    )
                for feature, values in ice_raw.items():
                    print(
                        f"ICE {feature}: "
                        f"{len(values['grid_values'])} grid points, "
                        f"{len(values['individual_predictions'])} individual curves"
                    )
            print("\n[OK] PDP / ICE explanation completed.")
    return pdp_ice_explanations