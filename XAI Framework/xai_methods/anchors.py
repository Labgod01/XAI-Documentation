import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from anchor import anchor_tabular

def run_anchors(unified_payload, prepared_inputs, output_dir=None, df=None):
    """Generate Anchors explanations for the active target."""

    model = prepared_inputs["model"]
    feature_names = prepared_inputs["feature_names"]
    X_np = prepared_inputs["X_np"]
    x_1d = prepared_inputs["x_1d"]
    x_2d = prepared_inputs["x_2d"]
    task_type = prepared_inputs["task_type"]
    metadata = prepared_inputs["metadata"]
    current_target = metadata["active_target"]
    scope = unified_payload["scope"] 
    # targets = metadata.get("targets", [current_target])
    targets = [current_target]

    if df is None:
            raise ValueError(
                "The original DataFrame must be provided to run_anchors()."
            )
    
    output_dir = Path(output_dir or "outputs")
    raw_dir = output_dir / "raw_results"
    plots_dir = output_dir / "plots"

    raw_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Store Anchors explanation objects and raw values separately
    anchors_explanations = {}
    anchors_raw_values = {}

    if scope == "global":
            print(
                "Anchors: Only local explanations are supported. "
                "Skipping Anchors for global scope."
            )
            return anchors_explanations
    
    if scope == "both":
        print(
            "Anchors: Only local explanations are supported. "
            "Generating local explanations only; global explanation skipped."
        )

    # Iterate dynamically over each target
    for current_target in targets:

        # Update unified payload for the active target
        metadata["active_target"] = current_target
        unified_payload["target"] = df[current_target]

        if unified_payload["scope"] in ["local", "both"]:

            # Determine prediction function and mode dynamically based on task type
            if task_type == "classification":
                # Wrapper to retain feature names
                predict_fn = lambda x: model.predict(pd.DataFrame(x, columns=feature_names))
            else:
                print("Skipping Anchors: supported only for classification.")
                continue

            # Map categorical column names to integer indices for AnchorTabularExplainer
            cat_cols = unified_payload["metadata"].get("categorical_features", [])
            cat_indices = [feature_names.index(col) for col in cat_cols if col in feature_names]

            categorical_names = (
                {idx: list(np.unique(X_np[:, idx])) for idx in cat_indices}
                if cat_indices else {}
            )

            class_names = unified_payload["metadata"].get("class_names")

            if class_names is None:
                if not hasattr(model, "classes_"):
                    raise ValueError(
                        "Anchors requires class_names in metadata or a model with a classes_ attribute."
                    )
                class_names = [str(c) for c in model.classes_]

            # Initialize Anchor Tabular Explainer
            explainer = anchor_tabular.AnchorTabularExplainer(
                class_names=class_names,
                feature_names=feature_names,
                train_data=X_np,
                categorical_names=categorical_names
            )

            # Make Anchors reproducible (the library uses the global np.random state)
            np.random.seed(unified_payload["metadata"]["random_state"])

            exp = explainer.explain_instance(
                data_row=x_1d,
                classifier_fn=predict_fn,
                threshold=0.95,
                verbose=True
            )

            # Extract raw Anchors values
            anchor_names = exp.names()
            anchor_precision = float(exp.precision())
            anchor_coverage = float(exp.coverage())
            anchor_prediction = model.predict(
                pd.DataFrame([x_1d], columns=feature_names)
            )[0]

            anchor_raw = {
                "method": "Anchors",
                "target": str(current_target),
                "prediction": anchor_prediction.item() if isinstance(anchor_prediction, np.generic) else anchor_prediction,                "rule": anchor_names,
                "rule_text": " AND ".join(anchor_names),
                "precision": anchor_precision,
                "coverage": anchor_coverage
            }

            anchors_explanations[current_target] = exp
            anchors_raw_values[current_target] = anchor_raw

            print("\n--- Anchors Raw Values ---")
            print(anchor_raw)

            with open(
                raw_dir / f"anchors_{current_target}.json",
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(anchor_raw, f, indent=4, ensure_ascii=False)

            # Anchors Output
            print("\n--- Anchors Explanation ---")
            print("Rule:", " AND ".join(anchor_names))
            print("Precision:", anchor_precision)
            print("Coverage:", anchor_coverage)

            # # Anchors plot
            # import textwrap
            # plt.figure(figsize=(8, 4))
            # plt.barh(["Coverage", "Precision"], [anchor_coverage, anchor_precision],
            #         color=["#d98c2b", "#2b5c8f"])
            # plt.xlim(0, 1)
            # plt.title(f"Anchors - Target: {current_target}\nIF " +
            #         textwrap.fill(" AND ".join(anchor_names), 60), fontsize=11)
            # plt.tight_layout()
            # plt.savefig(plots_dir / f"anchors_{current_target}.png", dpi=300)
            # #plt.show()
            # plt.close()

            # Anchors Plot

                        # 1) Save the interactive HTML (exactly the view you saw in the notebook)
            with open(plots_dir / f"anchors_{current_target}.html", "w", encoding="utf-8") as f:
                f.write(exp.as_html())

            # 2) Save a PNG version of the same layout
            from matplotlib.patches import Circle

            fig = plt.figure(figsize=(14, 4.5))
            gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.6, 1.8])

            # Panel 1: Example (the explained instance)
            ax1 = fig.add_subplot(gs[0]); ax1.axis("off")
            ax1.set_title("Example", loc="left", fontweight="bold")
            ax1.text(0.5, 0.5,
                     "\n".join(f"{n} = {v:.2f}" for n, v in zip(feature_names, x_1d)),
                     ha="center", va="center", fontsize=10, linespacing=1.8)

            # Panel 2: A.I. prediction
            ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
            ax2.set_title("A.I. prediction", loc="left", fontweight="bold")
            ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
            ax2.scatter([0.2], [0.6], s=600, color="#ff7f0e")
            ax2.text(0.38, 0.6, str(anchor_raw["prediction"]), va="center", fontsize=12)

            # Panel 3: Explanation (the anchor rule)
            ax3 = fig.add_subplot(gs[2]); ax3.axis("off")
            ax3.set_title("Explanation of A.I. prediction", loc="left", fontweight="bold")
            ax3.text(0.0, 0.85, "If ALL of these are true:", fontsize=9)
            for i, name in enumerate(anchor_names):
                ax3.text(0.0, 0.68 - 0.13 * i, "✓ " + name, fontsize=10, color="white",
                         bbox=dict(boxstyle="round,pad=0.4", facecolor="#1f77d0", edgecolor="none"))
            ax3.text(0.0, 0.1,
                     f"The A.I. will predict {anchor_raw['prediction']}  "
                     f"{anchor_precision * 100:.1f}% of the time", fontsize=11)

            fig.suptitle(f"Anchors - Target: {current_target}", fontsize=13)
            fig.savefig(plots_dir / f"anchors_{current_target}.png",
                        dpi=300, bbox_inches="tight")
            plt.close(fig)

    return anchors_explanations
