import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.inspection import PartialDependenceDisplay

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

            # 2. Define parameters according to task type
            X_pdp = reference_dataset.astype(float)
            pdp_parameters = {
                "estimator": model,
                "X": X_pdp,
                "features": selected_features,
                "grid_resolution": 20,
                "subsample": 50,
                "random_state": unified_payload["metadata"]["random_state"]
            }

            if task_type == "classification":
                pdp_parameters["response_method"] = "predict_proba"
            elif task_type == "regression":
                # For regression, sklearn uses predict()
                # No response_method is required.
                pass
            else:
                raise ValueError(
                    f"Unsupported task type: {task_type}"
                )

            pdp_display = None
            ice_display = None
            
            # 3. Partial Dependence Plot (PDP)
            if run_pdp:
                print("\nComputing Partial Dependence plots...")

                pdp_display = PartialDependenceDisplay.from_estimator(
                    **pdp_parameters,
                    kind="average"
                )
                pdp_display.figure_.suptitle(
                    "Partial Dependence Plots (PDP)",
                    fontsize=16
                )
                plt.tight_layout()
                plt.savefig(plots_dir / f"pdp_{current_target}.png", dpi=300)
                #plt.show()

            # Use the two most important features for ICE
            # to keep the visualization readable.
            ice_features = selected_features[:2]

            # 4. Individual Conditional Expectation (ICE) - local
            if run_ice:
                print("\nComputing ICE plots...")

                # Use the two most important features for ICE
                # to keep the visualization readable.

                ice_features = selected_features[:2]

                ice_parameters = {
                    "estimator": model,
                    "X": X_pdp,
                    "features": ice_features,
                    "kind": "individual",
                    "centered": False,
                    "grid_resolution": 20,
                    "subsample": 50,
                    "random_state": unified_payload["metadata"]["random_state"]
                }

                if task_type == "classification":
                    ice_parameters["response_method"] = "predict_proba"
                elif task_type == "regression":
                    # Regression uses model.predict()
                    pass
                ice_display = PartialDependenceDisplay.from_estimator(
                    **ice_parameters
                )
                ice_display.figure_.suptitle(
                    "Individual Conditional Expectation (ICE)",
                    fontsize=16
                )
                plt.tight_layout()
                plt.savefig(plots_dir / f"ice_{current_target}.png", dpi=300)
                #plt.show()

            # 5. Store explanations and extract raw values

            pdp_ice_explanations[current_target] = {
                "selected_features": selected_features,
                "ice_features": ice_features,
                "pdp_display": pdp_display,
                "ice_display": ice_display,
                "task_type": task_type
            }

            # Extract raw PDP and ICE data from sklearn displays
            pdp_raw = {}
            ice_raw = {}

            classes = (
                model.classes_.tolist()
                if task_type == "classification"
                and hasattr(model, "classes_")
                else None
            )

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

            pdp_ice_raw_values[current_target] = {
                "method": "PDP_ICE",
                "target": str(current_target),
                "task_type": task_type,
                "classes": classes,
                "pdp": pdp_raw,
                "ice": ice_raw
            }

            print("\n--- PDP / ICE Raw Values Summary ---")
            print("Target:", current_target)
            print("Task type:", task_type)
            print("PDP features:", list(pdp_raw.keys()))
            print("ICE features:", list(ice_raw.keys()))

            with open(
                raw_dir / f"pdp_ice_{current_target}.json",
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(pdp_ice_raw_values[current_target], f, indent=4, ensure_ascii=False)

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