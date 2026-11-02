import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from lime.lime_tabular import LimeTabularExplainer
from multi_output import class_suffix, to_json_value


def run_lime(unified_payload, prepared_inputs, output_dir=None):
    """Generate local LIME explanations for the active target."""

    model = prepared_inputs["model"]
    feature_names = prepared_inputs["feature_names"]
    X_np = prepared_inputs["X_np"]
    x_1d = prepared_inputs["x_1d"]
    x_2d = prepared_inputs["x_2d"]
    task_type = prepared_inputs["task_type"]
    metadata = prepared_inputs["metadata"]

    current_target = metadata["active_target"]
    scope = unified_payload["scope"]

    all_lime_explanations = {}
    
    if scope == "global":
        print(
            "LIME: Only local explanations are supported. "
            "Skipping LIME for global scope."
        )
        return all_lime_explanations

    if scope == "both":
        print(
            "LIME: Only local explanations are supported. "
            "Generating local explanations only; global explanation skipped."
        )

    # Determine prediction function and mode dynamically
    if task_type == "classification":
        lime_mode = "classification"

        def predict_fn(x):
            return model.predict_proba(
                pd.DataFrame(x, columns=feature_names)
            )
    else:
        lime_mode = "regression"

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

    # Initialize LIME Tabular Explainer
    lime_explainer = LimeTabularExplainer(
        training_data=X_np,
        feature_names=feature_names,
        categorical_features=cat_indices if cat_indices else None,
        class_names=metadata["class_names"],
        mode=lime_mode,
        random_state=metadata["random_state"],
    )
    # Classification: the classes chosen by class_to_explain (all classes by
    # default). LIME explains label 1 unless told otherwise.
    # Regression: one explanation (the label is ignored).
    classes_to_explain = prepared_inputs["classes_to_explain"] or [(None, 1)]
    label_indices = tuple(index for _, index in classes_to_explain)

    # Generate instance explanation (one call for every explained class)
    exp_lime = lime_explainer.explain_instance(
        data_row=x_1d,
        predict_fn=predict_fn,
        num_features=min(5, len(feature_names)),
        labels=label_indices,
    )

    raw_pred = model.predict(
        pd.DataFrame(x_2d, columns=feature_names)
    )[0]

    predicted_val = (
        float(raw_pred)
        if isinstance(raw_pred, (int, float))
        else str(raw_pred)
    )

    for explained_class, label_idx in classes_to_explain:
        suffix = class_suffix(explained_class, len(classes_to_explain))
        title_class = f" (class {explained_class})" if explained_class is not None else ""

        # Extract raw explanation weights
        lime_weights = exp_lime.as_list(label=label_idx)

        lime_llm_payload = {
            "target": current_target,
            "prediction": predicted_val,
            "explained_class": to_json_value(explained_class),
            "rules": {
                rule: float(weight)
                for rule, weight in lime_weights
            },
        }

        all_lime_explanations[f"{current_target}{suffix}"] = lime_llm_payload

        # Save raw explanation values
        if output_dir is not None:
            raw_dir = Path(output_dir) / "raw_results"
            raw_dir.mkdir(parents=True, exist_ok=True)

            raw_file = raw_dir / f"lime_{current_target}{suffix}.json"
            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump(
                    lime_llm_payload,
                    f,
                    indent=4,
                    ensure_ascii=False
                )

        print(f"\n--------------- LIME - Raw Values{title_class}: ---------------")
        print(lime_llm_payload)

        # Visualize local explanation
        print(f"\n----------------- LIME - Plot{title_class}: -----------------")
        fig = exp_lime.as_pyplot_figure(label=label_idx)
        fig.suptitle(f"LIME Local Explanation - {current_target}{title_class}")
        fig.tight_layout()

        if output_dir is not None:
            plot_dir = Path(output_dir) / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                plot_dir / f"lime_local_{current_target}{suffix}.png",
                bbox_inches="tight",
            )
        plt.close(fig)

    return all_lime_explanations
