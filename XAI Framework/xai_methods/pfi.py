import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.inspection import permutation_importance


def run_pfi(unified_payload, prepared_inputs, output_dir=None, df=None):
    """Generate PFI explanations for the active target."""

    model = prepared_inputs["model"]
    feature_names = prepared_inputs["feature_names"]
    X_np = prepared_inputs["X_np"]
    task_type = prepared_inputs["task_type"]
    metadata = prepared_inputs["metadata"]
    current_target = metadata["active_target"]
    scope = unified_payload["scope"] 
    # targets = metadata.get("targets", [current_target])
    targets = [current_target]

    if df is None:
            raise ValueError(
                "The original DataFrame must be provided to run_pfi()."
            )
    
    output_dir = Path(output_dir or "outputs")
    raw_dir = output_dir / "raw_results"
    plots_dir = output_dir / "plots"

    raw_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Store PFI explanation objects and raw values separately
    pfi_explanations = {}
    pfi_raw_values = {}

    if scope == "local":
                print(
                    "PFI: Only global explanations are supported. "
                    "Skipping PFI for local scope."
                )
                return pfi_explanations
    
    if scope == "both":
        print(
            "PFI: Only global explanations are supported. "
            "Generating global explanations only; local explanation skipped."
        )

    # Iterate dynamically over each target
    for current_target in targets:

        # Update unified payload for the active target
        metadata["active_target"] = current_target
        unified_payload["target"] = df[current_target]

        # Execute only if the explanation scope is global or both
        if unified_payload["scope"] in ["global", "both"]:

            # Retrieve user-defined scoring metric or fallback dynamically based on task type
            user_metric = unified_payload["metadata"].get("scoring_metric")
            if user_metric is not None:
                active_scoring = user_metric
            else:
                active_scoring = (
                    "accuracy"
                    if unified_payload["metadata"]["task_type"] == "classification"
                    else "r2"
                )

            # Compute Permutation Feature Importance
            pfi_result = permutation_importance(
                estimator=model,
                X=df[feature_names],
                y=unified_payload["target"].values,
                scoring=active_scoring,
                n_repeats=10,
                random_state=42,
                n_jobs=-1
            )

            # Store explanation object in dictionary
            pfi_explanations[current_target] = pfi_result

            # Sort features by mean importance in descending order
            sorted_indices = pfi_result.importances_mean.argsort()[::-1]
            sorted_features = [feature_names[i] for i in sorted_indices]
            sorted_means = pfi_result.importances_mean[sorted_indices]
            sorted_stds = pfi_result.importances_std[sorted_indices]

            # Prepare structured PFI raw values for LLM use
            pfi_raw = {
                "method": "PFI",
                "target": str(current_target),
                "task_type": unified_payload["metadata"]["task_type"],
                "scoring": active_scoring,
                "n_repeats": 10,
                "features": [
                    {
                        "feature": str(name),
                        "importance_mean": float(mean_val),
                        "importance_std": float(std_val)
                    }
                    for name, mean_val, std_val in zip(
                        sorted_features, sorted_means, sorted_stds
                    )
                ]
            }
            pfi_raw_values[current_target] = pfi_raw

            print("\n--- PFI Raw Values ---")
            print(pfi_raw)
            with open(
                raw_dir / f"pfi_{current_target}.json",
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(pfi_raw, f, indent=4, ensure_ascii=False)

            # Select top features for visualization (e.g., top 10)
            top_k = min(10, len(feature_names))
            top_features = sorted_features[:top_k][::-1]
            top_means = sorted_means[:top_k][::-1]
            top_stds = sorted_stds[:top_k][::-1]

            # Visualize Permutation Feature Importance with standard deviation bars
            plt.figure(figsize=(9, 6))
            plt.barh(
                top_features,
                top_means,
                xerr=top_stds,
                color="teal",
                edgecolor="black",
                alpha=0.8,
                capsize=4
            )
            plt.xlabel(f"Importance Score (Mean Metric Drop: {active_scoring})")
            plt.title(f"Permutation Feature Importance - Target: {current_target}", fontsize=12)
            plt.grid(axis="x", linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(plots_dir / f"pfi_{current_target}.png", dpi=300)
            #plt.show()

            # Print top feature importance summary
            print(f"\n--- Top Feature Importances (Target: {current_target}) ---")
            for rank, (name, mean_val, std_val) in enumerate(
                zip(sorted_features[:top_k], sorted_means[:top_k], sorted_stds[:top_k]), 1
            ):
                print(f"{rank:2d}. {name:<25} Mean: {mean_val:.4f} +/- {std_val:.4f}")

    return pfi_explanations