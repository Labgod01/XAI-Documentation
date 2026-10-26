import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import shap
from pathlib import Path

def run_tree_shap(unified_payload, prepared_inputs, output_dir=None):
    """Generate TreeSHAP explanations for the active target."""

    model = prepared_inputs["model"]
    feature_names = prepared_inputs["feature_names"]
    X_np = prepared_inputs["X_np"]
    x_1d = prepared_inputs["x_1d"]
    x_2d = prepared_inputs["x_2d"]
    task_type = prepared_inputs["task_type"]
    metadata = prepared_inputs["metadata"]
    reference_dataset = prepared_inputs["reference_dataset"]
    current_target = metadata["active_target"]
    scope = unified_payload["scope"]

    output_dir = Path(output_dir or "outputs")
    raw_dir = output_dir / "raw_results"
    plots_dir = output_dir / "plots"

    raw_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_tree_shap_explanations = {}
   
    # Determine prediction function and mode dynamically
    if task_type == "classification":
        tree_shap_mode = "classification"

        def predict_fn(x):
            return model.predict_proba(
                pd.DataFrame(x, columns=feature_names)
            )
    else:
        tree_shap_mode = "regression"

        def predict_fn(x):
            return model.predict(
                pd.DataFrame(x, columns=feature_names)
            )

    # Map categorical column names to integer indices
    cat_cols = metadata["categorical_features"]
    cat_indices = [
        feature_names.index(col)
        for col in cat_cols
        if col in feature_names
    ]

    # Initialize TreeSHAP Explainer
    tree_explainer = shap.TreeExplainer(model)
    
    if scope in ("local", "both"):

    
        # Compute TreeSHAP Explanation for a single instance
        local_tree_shap_values = tree_explainer(x_2d)
        print(
            f"Local TreeSHAP values shape: "
            f"{local_tree_shap_values.values.shape}"
        )
        # Container to collect SHAP values across targets for downstream LLM ingestion
        local_tree_shap_llm_payload = {}

        # print(f"\n================= TreeSHAP Local - Target: {current_target} =================")
        print(f"\n================= TreeSHAP Local =================")

        # ------------------------- Raw Values Extraction -------------------------

        # Select SHAP values for the query instance and correct output
        local_values_shape = local_tree_shap_values.values.shape
        
        if len(local_values_shape) == 3:
            if task_type == "classification":
                # predicted_class = model.predict(x_2d)[0]
                predicted_class = model.predict(
                    pd.DataFrame(x_2d, columns=feature_names)
                )[0]
                class_idx = list(model.classes_).index(predicted_class)
                exp_instance = local_tree_shap_values[0, :, class_idx]
            else:
                # Multi-output regression: retain target-specific selection
                # exp_instance = local_tree_shap_values[0, :, target_idx]
                 raise NotImplementedError(
                    "Multi-output regression requires explicit output selection."
                )
        else:
            # Single-output regression or binary SHAP explanation
            exp_instance = local_tree_shap_values[0]

        shap_vals = exp_instance.values
        base_val = float(exp_instance.base_values)

        # Map each feature to its corresponding SHAP value
        feature_contributions = [
            {"feature": feat, "shap_value": float(val)}
            for feat, val in zip(feature_names, shap_vals)
        ]
        # Sort features by absolute contribution impact
        feature_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        # Check

        x_df = pd.DataFrame(x_2d, columns=feature_names)

        if task_type == "classification":
            predicted_class = model.predict(x_df)[0]
            predicted_proba = model.predict_proba(x_df)[0]
        else:
            prediction = float(model.predict(x_df)[0])

        local_tree_shap_llm_payload[current_target] = {
            "method": "TreeSHAP",
            "target": current_target,
            "base_value": base_val,
            "prediction": float(
                model.predict_proba(
                    pd.DataFrame(x_2d, columns=feature_names))[0][class_idx]
            ) if task_type == "classification" else float(
                model.predict(pd.DataFrame(x_2d, columns=feature_names)
                )[0]
            ),
            "feature_contributions": feature_contributions
        }
        all_tree_shap_explanations["local"] = local_tree_shap_llm_payload

        # ------------------------- TreeSHAP Local Output -------------------------
        print("\n--------------- Local TreeSHAP - Raw Values: ---------------")
        print(local_tree_shap_llm_payload[current_target])

        with open(
            raw_dir / f"treeshap_local_{current_target}.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(local_tree_shap_llm_payload, f, indent=4, ensure_ascii=False)

        # print("\n--------------- Local TreeSHAP - Plots ({current_target}): ---------------")
        print("\n--------------- Local TreeSHAP - Plots ---------------")

        # Waterfall Plot
        plt.figure()
        shap.plots.waterfall(exp_instance, show=False)
        plt.title(f"SHAP Waterfall Plot - Target: {current_target}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"tree_shap_local_waterfall_{current_target}.png",
                            dpi=300, bbox_inches="tight")
        #plt.show()
        plt.close()

        # Force Plot (Rendered via Matplotlib for inline display)
        shap.plots.force(
            base_value=exp_instance.base_values,
            shap_values=exp_instance.values,
            features=exp_instance.data,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f"Local TreeSHAP Force Plot - Target: {current_target}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"tree_shap_local_force_{current_target}.png",
                                    dpi=300, bbox_inches="tight")
        #plt.show()

    if scope in ("global", "both"):

        # print(f"\n================= TreeSHAP Global - Target: {current_target} =================")
        print(f"\n================= TreeSHAP Global =================")

        # Compute TreeSHAP Explanation across the entire reference dataset (or sample)
        global_tree_shap_values = tree_explainer(reference_dataset)
        print(f"Global TreeSHAP values shape: {global_tree_shap_values.values.shape}")

        # --------- MULTI-TARGET ---------------------------

        # # Resolve dynamic output labels (classes for classification, targets for regression)
        # if task_type == "classification":
        #     # Extract class names from metadata or fallback to numeric class indices
        #     output_labels = unified_payload["metadata"].get("class_names")
        #     if output_labels is None:
        #         n_classes = global_tree_shap_values.values.shape[-1] if len(global_tree_shap_values.values.shape) == 3 else 2
        #         output_labels = [f"Class_{i}" for i in range(n_classes)]
        # else:
        #     # Regression: target names (handles both single target string and list of targets)
        #     output_labels = targets if isinstance(targets, list) else [targets]

        # # Slice Explanation object safely for current output
        # if len(global_tree_shap_values.values.shape) == 3:
        #     # 3D Array: (n_samples, n_features, n_classes/targets) -> Slice to 2D
        #     exp_global = global_tree_shap_values[:, :, out_idx]
        # else:
        #     # 2D Array: Single-target regression
        #     exp_global = global_tree_shap_values

        # Container to collect global SHAP feature importances for downstream LLM
        global_tree_shap_llm_payload = {}    

        # ----------- SINGLE-TARGET ----------------------

        # Resolve the output to explain
        global_values_shape = global_tree_shap_values.values.shape

        if len(global_values_shape) == 3:
            if task_type == "classification":
                # predicted_class = model.predict(x_2d)[0]
                predicted_class = model.predict(
                    pd.DataFrame(x_2d, columns=feature_names)
                )[0]
                out_idx = list(model.classes_).index(predicted_class)
                current_label = str(predicted_class)
            else:
                raise NotImplementedError(
                    "Multi-output regression is not supported yet."
                )

            exp_global = global_tree_shap_values[:, :, out_idx]

        else:
            exp_global = global_tree_shap_values

            if task_type == "classification":
                # predicted_class = model.predict(x_2d)[0]
                predicted_class = model.predict(
                    pd.DataFrame(x_2d, columns=feature_names)
                )[0]
                current_label = str(predicted_class)
            else:
                current_label = current_target

        # ------------------------- TreeSHAP Raw Values Extraction -------------------------

        # TreeSHAP Global Feature Importance extraction for LLM
        mean_abs_shap = np.mean(np.abs(exp_global.values), axis=0)
        global_contributions = [
            {"feature": feat, "mean_abs_shap": float(val)}
            for feat, val in zip(feature_names, mean_abs_shap)
        ]
        global_contributions.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

        global_tree_shap_llm_payload[current_label] = {
            "method": "TreeSHAP_Global",
            "output_label": current_label,
            "feature_importance": global_contributions
        }

        all_tree_shap_explanations["global"] = global_tree_shap_llm_payload

        print("\n--------------- Global TreeSHAP - Raw Values: ---------------")
        print(global_tree_shap_llm_payload[current_label])

        with open(
            raw_dir / f"treeshap_global_{current_target}.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(global_tree_shap_llm_payload, f, indent=4, ensure_ascii=False)

        # ------------------------- TreeSHAP Gloabl Output -------------------------
        # print(f"\n--------------- Global TreeSHAP - Plots ({current_target}): ---------------")
        print(f"\n--------------- Global TreeSHAP - Plots: ---------------")

        # TreeSHAP Beeswarm / Summary Plot
        plt.figure()
        shap.plots.beeswarm(exp_global, show=False)
        # plt.title(f"Global TreeSHAP Beeswarm Plot - Target: {current_target}", pad=15)
        plt.title(f"Global TreeSHAP Beeswarm Plot - Output: {current_label}", pad=15)
        plt.tight_layout()
        plt.savefig(plots_dir / f"tree_shap_global_beeswarm_{current_target}.png",
                    dpi=300, bbox_inches="tight")
        #plt.show()

        # TreeSHAP Global Feature Importance Bar Plot
        plt.figure()
        shap.plots.bar(exp_global, show=False)
        # plt.title(f"Global TreeSHAP Global Bar Plot - Target: {current_target}", pad=15)
        plt.title(f"Global TreeSHAP Bar Plot - Output: {current_label}", pad=15)
        plt.tight_layout()
        plt.savefig(plots_dir / f"tree_shap_global_bar_{current_target}.png",
                            dpi=300, bbox_inches="tight")
        #plt.show()

        # TreeSHAP Global Dynamic Dependence Plot (Top Feature)
        top_feature = global_contributions[0]["feature"]
        plt.figure()
        shap.plots.scatter(
            exp_global[:, top_feature],   # Strictly 1D slice
            color=exp_global,             # 2D slice for interaction color
            show=False
        )
        # plt.title(f"Global TreeSHAP Dependence: {top_feature} (Output: {current_label})", pad=15)
        plt.title(f"Global TreeSHAP Dependence: {top_feature} (Output: {current_label})", pad=15)
        plt.tight_layout()
        plt.savefig(plots_dir / f"tree_shap_global_dependence_{current_target}.png",
                            dpi=300, bbox_inches="tight")
        #plt.show()

    return all_tree_shap_explanations

if __name__ == "__main__":
    print("tree_shap.py correcctly loaded.")

