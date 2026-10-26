import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from lime.lime_tabular import LimeTabularExplainer


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
    label_idx = 1
    if task_type == "classification":
        pred = model.predict(pd.DataFrame(x_2d, columns=feature_names))[0]
        target_class = metadata["class_to_explain"] if metadata["class_to_explain"] is not None else pred
        label_idx = list(model.classes_).index(target_class)
    # Generate instance explanation
    exp_lime = lime_explainer.explain_instance(
        data_row=x_1d,
        predict_fn=predict_fn,
        num_features=min(5, len(feature_names)),
    )

    # Extract raw explanation weights
    lime_weights = exp_lime.as_list()

    raw_pred = model.predict(
        pd.DataFrame(x_2d, columns=feature_names)
    )[0]

    predicted_val = (
        float(raw_pred)
        if isinstance(raw_pred, (int, float))
        else str(raw_pred)
    )

    lime_llm_payload = {
        "target": current_target,
        "prediction": predicted_val,
        "rules": {
            rule: float(weight)
            for rule, weight in lime_weights
        },
    }

    all_lime_explanations[current_target] = lime_llm_payload

    # Save raw explanation values
    if output_dir is not None:
        raw_dir = Path(output_dir) / "raw_results"
        raw_dir.mkdir(parents=True, exist_ok=True)

        raw_file = raw_dir / f"lime_{current_target}.json"
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(
                lime_llm_payload,
                f,
                indent=4,
                ensure_ascii=False
            )

    print("\n--------------- LIME - Raw Values: ---------------")
    print(lime_llm_payload)

    # Visualize local explanation
    print("\n----------------- LIME - Plot: -----------------")
    fig = exp_lime.as_pyplot_figure()
    fig.suptitle(f"LIME Local Explanation - {current_target}")
    fig.tight_layout()

    if output_dir is not None:
        plot_dir = Path(output_dir) / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            plot_dir / f"lime_local_{current_target}.png",
            bbox_inches="tight",
        )
    #plt.show()

    return all_lime_explanations
