import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dice_ml
import json
from pathlib import Path

def run_counterfactuals(unified_payload, prepared_inputs, output_dir=None, df=None):
    """Generate Counterfactuals explanations for the active target."""

    model = prepared_inputs["model"]
    feature_names = prepared_inputs["feature_names"]
    x_2d = prepared_inputs["x_2d"]
    task_type = prepared_inputs["task_type"]
    metadata = prepared_inputs["metadata"]
    current_target = metadata["active_target"]
    scope = unified_payload["scope"] 
    # targets = metadata.get("targets", [current_target])
    targets = [current_target]

    if df is None:
        raise ValueError(
            "The original DataFrame must be provided to run_counterfactuals()."
        )

    output_dir = Path(output_dir or "outputs")
    raw_dir = output_dir / "raw_results"
    plots_dir = output_dir / "plots"

    raw_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_cf_explanations = {}

    if scope == "global":
            print(
                "Counterfactuals: Only local explanations are supported. "
                "Skipping Counterfactuals for global scope."
            )
            return all_cf_explanations
    
    if scope == "both":
        print(
            "Counterfactuals: Only local explanations are supported. "
            "Generating local explanations only; global explanation skipped."
        )

    for target_idx, current_target in enumerate(targets):
        # Print distinct header per target
        print(f"\n================ Target: {current_target} (Index: {target_idx}) ================")

        # Update unified payload for the active target
        metadata["active_target"] = current_target
        unified_payload["target"] = df[current_target]

        if scope in ("local", "both"):
            # 1. Reuse existing dataframe slice and identify continuous features
            df_dice_train = df[feature_names + [current_target]].copy()
            cat_cols = unified_payload["metadata"].get("categorical_features", [])
            continuous_features = [col for col in feature_names if col not in cat_cols]

            # ------------------- MULTI-TARGET -------------------------------

            # 2. Setup active model using a simple if statement
            # if len(targets) == 1:
            #     active_model = model
            # else:
            #     from types import SimpleNamespace
            #     active_model = SimpleNamespace(
            #         predict=lambda X: (
            #             model.predict(X)[:, target_idx]
            #             if getattr(model.predict(X), "ndim", 1) > 1
            #             else model.predict(X)
            #         ),
            #         predict_proba=lambda X: (
            #             model.predict_proba(X)[target_idx]
            #             if isinstance(model.predict_proba(X), list)
            #             else model.predict_proba(X)
            #         ) if hasattr(model, "predict_proba") else None
            #     )

            # ----------- SINGLE-TARGET FOR NOW ---------------------------
            # TODO: Add multi-target model support in a future update.
            # For now, Counterfactuals supports single-target models only.
            active_model = model

            # 3. Initialize DiCE data and model objects
            d_dice = dice_ml.Data(
                dataframe=df_dice_train,
                continuous_features=continuous_features,
                outcome_name=current_target,
            )
            m_dice = dice_ml.Model(
                model=active_model,
                backend="sklearn",
                model_type="classifier" if task_type == "classification" else "regressor",
            )
            exp_dice = dice_ml.Dice(d_dice, m_dice, method="random")

            # 4. Prepare single query instance as DataFrame
            query_instance = pd.DataFrame(x_2d, columns=feature_names)

            # Extract current prediction
            raw_pred = active_model.predict(query_instance)[0]
            predicted_val = raw_pred.item() if isinstance(raw_pred, np.generic) else raw_pred       

            # 5. Configure counterfactual targets
            cf_kwargs = {
                "query_instances": query_instance,
                "total_CFs": 3,
                "features_to_vary": "all",
                "random_seed": unified_payload["metadata"].get("random_state", 42),
            }

            if task_type == "classification":
                classes = unified_payload["metadata"].get("class_names")

                if classes is None:
                    if not hasattr(model, "classes_"):
                        raise ValueError(
                            "Classification counterfactuals require class_names in metadata "
                            "or a model with a classes_ attribute."
                        )
                    classes = list(model.classes_)

                cf_kwargs["desired_class"] = (
                    "opposite" if len(classes) == 2
                    else next(cls for cls in classes if cls != raw_pred)
                )
            else:
                predicted_value = float(predicted_val)
                cf_kwargs["desired_range"] = sorted([
                    predicted_value * 1.05,
                    predicted_value * 1.50
                ])

            # 6. Generate explanations and extract results
            dice_exp = exp_dice.generate_counterfactuals(**cf_kwargs)
            cf_examples = dice_exp.cf_examples_list
            cf_df = (
                cf_examples[0].final_cfs_df
                if cf_examples and cf_examples[0] is not None
                else None
            )

            # 7. Build structured LLM payload with explicit feature deltas
            orig_dict = query_instance.iloc[0].to_dict()
            cf_llm_payload = {
                "method": "DiCE_Counterfactuals",
                "target": current_target,
                "target_index": target_idx,
                "predicted_value": predicted_val,
                "desired_target": cf_kwargs.get("desired_class", cf_kwargs.get("desired_range")),
                "counterfactuals": [],
            }

            if cf_df is not None and not cf_df.empty:
                for idx, row in cf_df.iterrows():
                    cf_row = row.to_dict()
                    cf_target_val = cf_row.pop(current_target, None)
                    modifications = {}

                    for col, orig_val in orig_dict.items():
                        new_val = cf_row.get(col)
                        if isinstance(orig_val, (int, float, np.number)) and isinstance(new_val, (int, float, np.number)):
                            if not np.isclose(orig_val, new_val, atol=1e-4):
                                modifications[col] = {
                                    "from": float(orig_val),
                                    "to": float(new_val),
                                    "delta": float(new_val - orig_val),
                                }
                        elif orig_val != new_val:
                            modifications[col] = {"from": orig_val, "to": new_val}

                    cf_llm_payload["counterfactuals"].append({
                        "cf_index": idx + 1,
                        "target_prediction": float(cf_target_val) if isinstance(cf_target_val, (int, float, np.number)) else str(cf_target_val),
                        "modifications": modifications,
                        "full_features": cf_row,
                    })

            all_cf_explanations[current_target] = cf_llm_payload

            # 8. Print raw values and visualize
            print(f"\n--------------- Counterfactuals Payload ({current_target}) ---------------")
            print(cf_llm_payload)

            with open(
                raw_dir / f"counterfactuals_{current_target}.json",
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(cf_llm_payload, f, indent=4, ensure_ascii=False)

            print(f"\n----------------- Counterfactuals Plots ({current_target}) -----------------")
            if cf_df is not None and not cf_df.empty:
                
                # Bar plot of feature deltas for all counterfactuals
                counterfactuals = cf_llm_payload["counterfactuals"]

                plot_data = []
                for cf in counterfactuals:
                    for feature, modification in cf["modifications"].items():
                        if "delta" in modification:
                            plot_data.append({
                                "label": f"CF {cf['cf_index']} - {feature}",
                                "delta": modification["delta"],
                            })

                if plot_data:
                    labels = [item["label"] for item in plot_data]
                    deltas = [item["delta"] for item in plot_data]

                    fig_height = max(4, 0.6 * len(labels))
                    plt.figure(figsize=(10, fig_height))

                    y_pos = np.arange(len(labels))
                    plt.barh(y_pos, deltas, color="#2b5c8f")
                    plt.yticks(y_pos, labels)
                    plt.axvline(0, color="black", linewidth=0.8)
                    plt.xlabel("Delta (Counterfactual - Original)")
                    plt.title(
                        f"Counterfactual Feature Perturbations ({current_target})"
                    )
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plt.savefig(
                        plots_dir / f"cf_deltas_{current_target}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()

                # Display formatted table of changes
                dice_exp.visualize_as_dataframe(show_only_changes=True)

    return all_cf_explanations