import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import shap
from pathlib import Path
from multi_output import class_suffix, shap_for_class, to_json_value

def run_kernel_shap(unified_payload, prepared_inputs, output_dir=None):
    """Generate KernelSHAP explanations for the active target."""

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
    # Background data summary (50 samples for speed)
    background_data = shap.sample(reference_dataset, 50)

    output_dir = Path(output_dir or "outputs")
    raw_dir = output_dir / "raw_results"
    plots_dir = output_dir / "plots"

    raw_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_kernel_shap_explanations = {}
   
    # Determine prediction function and mode dynamically
    if task_type == "classification":
        kernel_shap_mode = "classification"

        def predict_fn(x):
            return model.predict_proba(
                pd.DataFrame(x, columns=feature_names)
            )
    else:
        kernel_shap_mode = "regression"

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

    # Direct KernelExplainer initialization
    # Wrapped with lambda to guarantee DataFrame input with feature names
    kernel_explainer = shap.KernelExplainer(
        predict_fn,
        background_data
    )
    
    # Classes to explain (classification); regression has one output
    classes_to_explain = prepared_inputs["classes_to_explain"] or [(None, None)]
    x_df = pd.DataFrame(x_2d, columns=feature_names)

    if scope in ("local", "both"):

        # Compute KernelSHAP Explanation for a single instance
        local_values = kernel_explainer(x_2d)
        print(f"Local KernelSHAP values shape: {local_values.values.shape}")

        print("\n================= KernelSHAP Local =================")

        for explained_class, class_idx in classes_to_explain:
            suffix = class_suffix(explained_class, len(classes_to_explain))
            title_class = f" (class {explained_class})" if explained_class is not None else ""

            # ------------------------- Raw Values Extraction -------------------------

            # SHAP values of the query instance for this class
            exp_instance = shap_for_class(local_values, class_idx)[0]

            shap_vals = exp_instance.values
            base_val = float(exp_instance.base_values)

            # Map each feature to its corresponding SHAP value, sorted by impact
            feature_contributions = [
                {"feature": feat, "shap_value": float(val)}
                for feat, val in zip(feature_names, shap_vals)
            ]
            feature_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

            local_llm_payload = {
                current_target: {
                    "method": "KernelSHAP",
                    "target": current_target,
                    "explained_class": to_json_value(explained_class),
                    "base_value": base_val,
                    "prediction": float(
                        model.predict_proba(x_df)[0][class_idx]
                    ) if task_type == "classification" else float(
                        model.predict(x_df)[0]
                    ),
                    "feature_contributions": feature_contributions
                }
            }
            all_kernel_shap_explanations.setdefault("local", {})[f"{current_target}{suffix}"] = (
                local_llm_payload[current_target]
            )

            # ------------------------- KernelSHAP Local Output -------------------------
            print(f"\n--------------- Local KernelSHAP - Raw Values{title_class}: ---------------")
            print(local_llm_payload[current_target])

            with open(
                raw_dir / f"kernel_shap_local_{current_target}{suffix}.json",
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(local_llm_payload, f, indent=4, ensure_ascii=False)

            print(f"\n--------------- Local KernelSHAP - Plots{title_class} ---------------")

            # Waterfall Plot
            plt.figure()
            shap.plots.waterfall(exp_instance, show=False)
            plt.title(f"KernelSHAP Waterfall Plot - Target: {current_target}{title_class}")
            plt.tight_layout()
            plt.savefig(plots_dir / f"kernel_shap_local_waterfall_{current_target}{suffix}.png",
                        dpi=300, bbox_inches="tight")
            plt.close()

            # Force Plot (rendered via Matplotlib)
            shap.plots.force(
                base_value=exp_instance.base_values,
                shap_values=exp_instance.values,
                features=exp_instance.data,
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            plt.title(f"Local KernelSHAP Force Plot - Target: {current_target}{title_class}")
            plt.tight_layout()
            plt.savefig(plots_dir / f"kernel_shap_local_force_{current_target}{suffix}.png",
                        dpi=300, bbox_inches="tight")
            plt.close("all")

    if scope in ("global", "both"):

        print("\n================= KernelSHAP Global =================")

        # Compute KernelSHAP Explanation across the entire reference dataset
        global_values = kernel_explainer(reference_dataset)
        print(f"Global KernelSHAP values shape: {global_values.values.shape}")

        for explained_class, class_idx in classes_to_explain:
            suffix = class_suffix(explained_class, len(classes_to_explain))
            current_label = str(explained_class) if explained_class is not None else current_target

            exp_global = shap_for_class(global_values, class_idx)

            # ------------------------- Raw Values Extraction -------------------------

            # Global feature importance (mean |SHAP|) for the LLM
            mean_abs_shap = np.mean(np.abs(exp_global.values), axis=0)
            global_contributions = [
                {"feature": feat, "mean_abs_shap": float(val)}
                for feat, val in zip(feature_names, mean_abs_shap)
            ]
            global_contributions.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

            global_llm_payload = {
                current_label: {
                    "method": "KernelSHAP_Global",
                    "target": current_target,
                    "explained_class": to_json_value(explained_class),
                    "output_label": current_label,
                    "feature_importance": global_contributions
                }
            }
            all_kernel_shap_explanations.setdefault("global", {})[f"{current_target}{suffix}"] = (
                global_llm_payload[current_label]
            )

            print(f"\n--------------- Global KernelSHAP - Raw Values (output: {current_label}): ---------------")
            print(global_llm_payload[current_label])

            with open(
                raw_dir / f"kernel_shap_global_{current_target}{suffix}.json",
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(global_llm_payload, f, indent=4, ensure_ascii=False)

            # ------------------------- KernelSHAP Global Output -------------------------
            print(f"\n--------------- Global KernelSHAP - Plots (output: {current_label}): ---------------")

            # Beeswarm / Summary Plot
            plt.figure()
            shap.plots.beeswarm(exp_global, show=False)
            plt.title(f"Global KernelSHAP Beeswarm Plot - Output: {current_label}", pad=15)
            plt.tight_layout()
            plt.savefig(plots_dir / f"kernel_shap_global_beeswarm_{current_target}{suffix}.png",
                        dpi=300, bbox_inches="tight")
            plt.close("all")

            # Global Feature Importance Bar Plot
            plt.figure()
            shap.plots.bar(exp_global, show=False)
            plt.title(f"Global KernelSHAP Bar Plot - Output: {current_label}", pad=15)
            plt.tight_layout()
            plt.savefig(plots_dir / f"kernel_shap_global_bar_{current_target}{suffix}.png",
                        dpi=300, bbox_inches="tight")
            plt.close("all")

            # Dependence Plot of the top feature
            top_feature = global_contributions[0]["feature"]
            plt.figure()
            shap.plots.scatter(
                exp_global[:, top_feature],   # Strictly 1D slice
                color=exp_global,             # 2D slice for interaction color
                show=False
            )
            plt.title(f"Global KernelSHAP Dependence: {top_feature} (Output: {current_label})", pad=15)
            plt.tight_layout()
            plt.savefig(plots_dir / f"kernel_shap_global_dependence_{current_target}{suffix}.png",
                        dpi=300, bbox_inches="tight")
            plt.close("all")

    return all_kernel_shap_explanations

if __name__ == "__main__":
    print("kernel_shap.py correcctly loaded.")

