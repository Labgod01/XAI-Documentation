from pathlib import Path
import shutil
import config
import argparse
import matplotlib.pyplot as plt
from demo_models import create_demo
from data_loader import load_inputs
from pipeline import build_unified_payload
from framework_adapter import prepare_xai_inputs
from xai_methods.lime import run_lime
from xai_methods.tree_shap import run_tree_shap
from xai_methods.kernel_shap import run_kernel_shap
from xai_methods.counterfactuals import run_counterfactuals
from xai_methods.anchors import run_anchors
from xai_methods.ale import run_ale
from xai_methods.pfi import run_pfi
from xai_methods.pdp_ice import run_pdp_ice
from xai_methods.ig import run_ig

# key -> (label, module, function, needs_df). The dict order is the run order.
# The keys are the same names used in config.xai_methods.

METHODS = {
    "lime":                 ("LIME",                 run_lime,            False),
    "tree_shap":            ("TreeSHAP",             run_tree_shap,       False),
    "kernel_shap":          ("KernelSHAP",           run_kernel_shap,     False),
    "dice":                 ("Counterfactuals",      run_counterfactuals, True),
    "anchors":              ("Anchors",              run_anchors,         True),
    "ale":                  ("ALE",                  run_ale,             True),
    "pfi":                  ("PFI",                  run_pfi,             True),
    "pdp_ice":              ("PDP/ICE",              run_pdp_ice,         True),
    "integrated_gradients": ("Integrated Gradients", run_ig,              True),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["local", "global", "both"])
    parser.add_argument("--data")
    parser.add_argument("--model")
    parser.add_argument("--target")
    parser.add_argument("--task", choices=["classification", "regression"])
    parser.add_argument("--methods", nargs="+", choices=list(METHODS))
    return parser.parse_args()

def main():
    args = parse_args()
    scope = args.scope or config.explanation_scope
    custom_inputs_given = any(
        value is not None for value in (args.data, args.model, args.target, args.task)
    )
    run_mode = "custom" if custom_inputs_given else config.run_mode
    requested = args.methods or config.xai_methods
    selected = [key for key in METHODS if key in requested]

    print("=" * 60)
    print("XAI Framework")
    print("=" * 60)

    # 1. Load demo or custom model and dataset
    if run_mode == "demo":
        demo = create_demo(
            config.demo_type,
            random_state=config.random_state,
        )

        model = demo["model"]
        X = demo["X"]
        y = demo["y"]
        target_col = demo["target_col"]
        task_type = demo["task_type"]

        df = X.copy()
        df[target_col] = y

        print(f"Mode: demo ({config.demo_type})")

    elif run_mode == "custom":
        target_col = args.target or config.target_col
        task_type = args.task or config.task_type

        data, X, y, model = load_inputs(
            args.data or config.data_path,
            args.model or config.model_path,
            target_col,
        )

        df = data

        print("Mode: custom")

    else:
        raise ValueError(
            f"Unsupported run_mode: {run_mode}"
        )

    # 2. Build unified payload
    payload = build_unified_payload(
        model=model,
        df=df,
        target_col=target_col,
        task_type=task_type,
        scope=scope,
        query_index=config.query_index,
        class_names=config.class_names,
        class_to_explain=config.class_to_explain,
        categorical_features_override=config.categorical_features_override,
        immutable_features=config.immutable_features,
        scoring_metric=config.scoring_metric,
        features_to_plot=config.features_to_plot,
        random_state=config.random_state,
        feature_metadata=config.feature_metadata,
    )

    # 3. Prepare inputs
    prepared_inputs = prepare_xai_inputs(payload)

    # 4. Run Methods
    output_dir = Path(config.output_dir)
    clean_output_dir(output_dir)

    print(f"Task: {task_type}")
    print(f"Scope: {scope}")
    print("XAI methods: " + ", ".join(METHODS[key][0] for key in selected))

    results = {}
    for key in selected:
        label, run_method, needs_df = METHODS[key]

        kwargs = {
            "unified_payload": payload,
            "prepared_inputs": prepared_inputs,
            "output_dir": output_dir,
        }
        if needs_df:
            kwargs["df"] = df

        plt.close("all")
        results[key] = run_method(**kwargs)

    print("\nFramework execution completed.")
    print(f"Results directory: {output_dir.resolve()}")
    
    for key in selected:
        print(f"{METHODS[key][0]} explanations returned: {len(results[key])}")

def clean_output_dir(output_dir):
    output_dir = Path(output_dir)

    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    main()
