from pathlib import Path
import json
import shutil
import sys
import traceback
import config
import argparse
import matplotlib.pyplot as plt
from demo_models import create_demo
from data_loader import load_inputs
from pipeline import build_unified_payload
from framework_adapter import fill_unknown_targets, prepare_xai_inputs
from multi_output import as_target_list
from xai_methods.lime import run_lime
from xai_methods.tree_shap import run_tree_shap
from xai_methods.kernel_shap import run_kernel_shap
from xai_methods.counterfactuals import run_counterfactuals
from xai_methods.anchors import run_anchors
from xai_methods.ale import run_ale
from xai_methods.pfi import run_pfi
from xai_methods.pdp_ice import run_pdp_ice
from xai_methods.ig import run_ig

# key -> (label, function, needs_df). The dict order is the run order.
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

# --demo choice -> (demo_type, task_type) for create_demo()
DEMOS = {
    "classification":    ("classification", "classification"),
    "regression":        ("regression", "regression"),
    "nn_classification": ("ig_nn", "classification"),
    "nn_regression":     ("ig_nn", "regression"),
}

REQUIRED = ("data", "model", "target", "task")


def parse_label(value):
    """A class label from the command line: 2 -> int, 0.5 -> float, else str."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run XAI methods on a trained model and its dataset.",
        epilog=(
            "example: python main.py --data data.csv --model model.joblib "
            "--target label --task classification\n"
            "Every other setting is in config.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required: they describe the problem, so they never come from config.py
    required = parser.add_argument_group("required (unless --demo is given)")
    required.add_argument("--data", help="CSV file with the features and the target column(s)")
    required.add_argument("--model", help="trained model file (.joblib, .pkl, .pt)")
    # One target column, or several for multi-label / multi-output /
    # multi-target regression models, e.g. --target y1 y2 y3
    required.add_argument("--target", nargs="+", help="target column(s), in the order of the model outputs")
    required.add_argument("--task", choices=["classification", "regression"])

    # Optional: when not given, the value comes from config.py
    optional = parser.add_argument_group("optional (default: the value in config.py)")
    optional.add_argument("--methods", nargs="+", choices=list(METHODS),
                          default=config.xai_methods, metavar="METHOD",
                          help="XAI methods to run: " + ", ".join(METHODS)
                               + " (default: config.xai_methods)")
    optional.add_argument("--scope", choices=["local", "global", "both"],
                          default=config.explanation_scope,
                          help="default: %(default)s")
    optional.add_argument("--query-index", type=int, default=config.query_index,
                          help="row of the dataset explained by the local methods (default: %(default)s)")
    optional.add_argument("--class-to-explain", nargs="+", metavar="CLASS",
                          help="classification: 'predicted', a class label (e.g. 2), or one "
                               "label per target (e.g. risk=2 flag=1). "
                               "Default: every class (binary: the positive class)")
    optional.add_argument("--output-dir", type=Path, default=Path(config.output_dir),
                          help="folder for the plots and raw results (default: %(default)s)")
    optional.add_argument("--random-state", type=int, default=config.random_state,
                          help="default: %(default)s")

    parser.add_argument("--demo", choices=list(DEMOS),
                        help="run a built-in demo model and dataset instead")

    args = parser.parse_args()

    given = [name for name in REQUIRED if getattr(args, name) is not None]
    if args.demo and given:
        parser.error("--demo cannot be combined with "
                     + ", ".join(f"--{name}" for name in given))
    missing = [name for name in REQUIRED if getattr(args, name) is None]
    if not args.demo and missing:
        parser.error("the following arguments are required: "
                     + ", ".join(f"--{name}" for name in missing))

    # --class-to-explain: one value for every target, or target=label pairs
    if args.class_to_explain is None:
        args.class_to_explain = config.class_to_explain
    else:
        values = args.class_to_explain
        pairs = [value for value in values if "=" in value]
        if not pairs and len(values) == 1:
            args.class_to_explain = parse_label(values[0])
        elif len(pairs) == len(values):
            args.class_to_explain = {
                target: parse_label(label)
                for target, label in (value.split("=", 1) for value in values)
            }
        else:
            parser.error("--class-to-explain takes one class (e.g. 2) "
                         "or one target=class pair per target (e.g. risk=2 flag=1)")
    return args


def main():
    args = parse_args()
    scope = args.scope
    selected = [key for key in METHODS if key in args.methods]

    # Settings read by the methods from config at run time
    config.run_mode = "demo" if args.demo else "custom"
    config.random_state = args.random_state

    print("=" * 60)
    print("XAI Framework")
    print("=" * 60)

    # 1. Load demo or custom model and dataset
    if args.demo:
        demo_type, demo_task = DEMOS[args.demo]
        demo = create_demo(
            demo_type,
            demo_task,
            random_state=args.random_state,
        )

        model = demo["model"]
        # The demo PyTorch network was trained on standardized features:
        # keep its scaler with it, so the wrapper feeds it scaled inputs
        if demo.get("preprocessing") is not None:
            model = {"model": model, "scaler": demo["preprocessing"]}
        X = demo["X"]
        y = demo["y"]
        target_col = demo["target_col"]
        task_type = demo["task_type"]

        df = X.copy()
        df[target_col] = y

        print(f"Mode: demo ({args.demo})")

    else:
        target_col = args.target[0] if len(args.target) == 1 else args.target
        task_type = args.task

        data, X, y, model = load_inputs(args.data, args.model, target_col)

        df = data

        print("Mode: custom")

    output_dir = args.output_dir
    clean_output_dir(output_dir)

    # Several target columns (multi-label, multi-output, multi-target
    # regression): every method runs once per target, and every output file
    # name ends with the target name
    targets = as_target_list(target_col)
    results = {}

    # Rows without a true target value get the model's prediction
    df, target_known = fill_unknown_targets(model, df, targets, task_type)

    save_feature_metadata(config.feature_metadata, df, targets, output_dir)
    check_class_to_explain(args.class_to_explain, targets, task_type)

    # 2-3. Build the payload and the method inputs of every target first, so
    # that setting problems show up before the slow methods run
    prepared = []
    for active_target in targets:
        payload = build_unified_payload(
            model=model,
            df=df,
            target_col=target_col,
            task_type=task_type,
            scope=scope,
            active_target=active_target,
            query_index=args.query_index,
            class_names=config.class_names,
            class_to_explain=args.class_to_explain,
            categorical_number_features=config.categorical_number_features,
            immutable_features=config.immutable_features,
            scoring_metric=config.scoring_metric,
            features_to_plot=config.features_to_plot,
            random_state=args.random_state,
            feature_metadata=config.feature_metadata,
            target_known=target_known,
        )
        prepared.append((active_target, payload, prepare_xai_inputs(payload)))

    # One class given for every target, but no target has it: most likely a typo
    if (task_type == "classification" and len(targets) > 1
            and all(p[2]["metadata"].get("class_to_explain_skipped") for p in prepared)):
        raise ValueError(
            f"class_to_explain={args.class_to_explain!r} is not a class of any target."
        )

    failures = []
    for active_target, payload, prepared_inputs in prepared:

        # 4. Run Methods
        if active_target == targets[0]:
            print(f"Task: {task_type} ({payload['metadata']['problem_type']})")
            print(f"Targets: {', '.join(map(str, targets))}")
            print(f"Scope: {scope}")
            print("XAI methods: " + ", ".join(METHODS[key][0] for key in selected))

        if len(targets) > 1:
            print("\n" + "#" * 60)
            print(f"Target {targets.index(active_target) + 1}/{len(targets)}: {active_target}")
            print("#" * 60)

        results[active_target] = {}
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
            # A failing method is reported, and the other methods still run
            try:
                results[active_target][key] = run_method(**kwargs)
            except Exception as exc:
                traceback.print_exc()
                print(f"\n[ERROR] {label} failed for target '{active_target}': "
                      f"{type(exc).__name__}: {exc}")
                failures.append((active_target, label, f"{type(exc).__name__}: {exc}"))
                results[active_target][key] = None

    print("\nFramework execution completed.")
    print(f"Results directory: {output_dir.resolve()}")

    for active_target in targets:
        if len(targets) > 1:
            print(f"\nTarget: {active_target}")
        for key in selected:
            result = results[active_target][key]
            status = "FAILED" if result is None else len(result)
            print(f"{METHODS[key][0]} explanations returned: {status}")

    if failures:
        print(f"\n{len(failures)} method run(s) failed:")
        for active_target, label, error in failures:
            print(f"  {label} [{active_target}]: {error}")
        sys.exit(1)

def check_class_to_explain(class_to_explain, targets, task_type):
    """A dict class_to_explain may only name target columns."""
    if class_to_explain is None:
        return
    if task_type != "classification":
        print("Note: class_to_explain is ignored for regression.")
        return
    if isinstance(class_to_explain, dict):
        unknown = [name for name in class_to_explain if name not in targets]
        if unknown:
            raise ValueError(
                f"class_to_explain names {unknown}, which are not target columns {targets}."
            )


def save_feature_metadata(feature_metadata, df, targets, output_dir):
    """Save the feature descriptions next to the raw results, for an LLM."""
    if not feature_metadata:
        return
    columns = [col for col in df.columns if col not in targets] + targets
    unknown = [name for name in feature_metadata if name not in columns]
    if unknown:
        print(f"Warning: feature_metadata mentions columns that are not in the dataset: {unknown}")

    raw_dir = Path(output_dir) / "raw_results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(feature_metadata, f, indent=4, ensure_ascii=False)


def clean_output_dir(output_dir):
    # Only the folders the methods write to: --output-dir may point anywhere
    output_dir = Path(output_dir)
    for name in ("plots", "raw_results"):
        shutil.rmtree(output_dir / name, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    main()
