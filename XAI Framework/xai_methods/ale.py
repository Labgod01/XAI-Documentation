import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from alibi.explainers.ale import ALE, plot_ale

def run_ale(unified_payload, prepared_inputs, output_dir=None, df=None):
    """Generate ALE explanations for the active target."""

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
                "The original DataFrame must be provided to run_ale()."
            )
    
    output_dir = Path(output_dir or "outputs")
    raw_dir = output_dir / "raw_results"
    plots_dir = output_dir / "plots"

    raw_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Fix for NumPy 2.x compatibility with older alibi version
    if not hasattr(np, "object"):
        np.object = object

    # Store ALE explanation objects and raw values separately
    ale_explanations = {}
    ale_raw_values = {}

    if scope == "local":
                print(
                    "ALE: Only global explanations are supported. "
                    "Skipping ALE for local scope."
                )
                return ale_explanations
        
    if scope == "both":
        print(
            "ALE: Only global explanations are supported. "
            "Generating global explanations only; local explanation skipped."
        )

    # Iterate dynamically over each target
    for current_target in targets:

        # Update unified payload for the active target
        metadata["active_target"] = current_target
        unified_payload["target"] = df[current_target]

        # Execute only if the explanation scope is global or both
        if unified_payload["scope"] in ["global", "both"]:

            # Determine prediction function dynamically based on task type
            if task_type == "classification":
                # ALE expects class probability outputs for classification tasks
                predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_names))
            else:
                # ALE expects continuous outputs for regression tasks
                predict_fn = lambda x: model.predict(pd.DataFrame(x, columns=feature_names))

            # Initialize the ALE explainer
            ale_explainer = ALE(
                predictor=predict_fn,
                feature_names=feature_names,
                target_names=unified_payload["metadata"].get("class_names", None)
            )

            # Compute Accumulated Local Effects across the training dataset
            ale_exp = ale_explainer.explain(X_np)

            # Store explanation object in the dictionary
            ale_explanations[current_target] = ale_exp


            # Select features to visualize
            features_to_plot = feature_names[:3]

            # Store raw ALE values for each plotted feature
            ale_llm_payload = {
                "method": "ALE",
                "target": str(current_target),
                "task_type": task_type,
                "features": {}
            }

            for target_feature in features_to_plot:
                feat_idx = feature_names.index(target_feature)

                ale_values = np.asarray(
                    ale_exp.ale_values[feat_idx]
                ).reshape(-1)

                feature_grid = np.asarray(
                    ale_exp.feature_values[feat_idx]
                ).reshape(-1)

                ale_llm_payload["features"][target_feature] = {
                    "feature_grid": [float(v) for v in feature_grid],
                    "ale_values": [float(v) for v in ale_values]
                }

            ale_raw_values[current_target] = ale_llm_payload

            print("\n--------------- ALE - Raw Values ---------------")
            print(ale_llm_payload)

            with open(
                raw_dir / f"ale_{current_target}.json",
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(ale_llm_payload, f, indent=4, ensure_ascii=False)

            # Generate and render ALE plots
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_ale(ale_exp, features=features_to_plot, ax=ax)
            plt.suptitle(f"ALE Plots - Target: {current_target}", fontsize=14)
            plt.tight_layout()
            plt.savefig(plots_dir / f"ale_{current_target}.png", dpi=300)
            #plt.show()
    return ale_explanations