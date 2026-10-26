"""
Runs the XAI framework for the 4 basic cases and reports, per case and per
method, whether it worked and which files it produced.

    python test_cases.py                          # all methods, all 8 cases
    python test_cases.py anchors                  # one method, all 8 cases
    python test_cases.py treeshap kernelshap      # several methods
    python test_cases.py anchors --case demo_regression custom_nn_classification
    python test_cases.py anchors --scope local

Method keys: lime treeshap kernelshap counterfactuals anchors ale pfi pdp ig

Cases : sklearn model  -> demo / custom  x  classification / regression   (4 cases)
        PyTorch NN     -> demo / custom  x  classification / regression   (4 cases, only IG can run:
                          a bare nn.Module has no predict()/predict_proba() for the other methods)
Scope : "both" (local + global) - change SCOPE below to test "local" or "global"
Output: outputs_test/<case>/{plots,raw_results}   (your normal outputs/ is untouched)
"""
import argparse
import importlib
import shutil
import tempfile
import traceback
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no windows, save only
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

import config
from data_loader import load_inputs
from pipeline import build_unified_payload
from framework_adapter import prepare_xai_inputs

SCOPE = "both"
TEST_ROOT = Path(config.base_dir) / "outputs_test"

# (name, module, function, needs_df)
METHODS = [
    ("LIME",            "xai_methods.lime",            "run_lime",            False),
    ("TreeSHAP",        "xai_methods.tree_shap",       "run_tree_shap",       False),
    ("KernelSHAP",      "xai_methods.kernel_shap",     "run_kernel_shap",     False),
    ("Counterfactuals", "xai_methods.counterfactuals", "run_counterfactuals", True),
    ("Anchors",         "xai_methods.anchors",         "run_anchors",         True),
    ("ALE",             "xai_methods.ale",             "run_ale",             True),
    ("PFI",             "xai_methods.pfi",             "run_pfi",             True),
    ("PDP/ICE",         "xai_methods.pdp_ice",         "run_pdp_ice",         True),
    ("IntegratedGrad",  "xai_methods.ig",              "run_ig",              True),
]

METHOD_KEYS = {
    "lime": "LIME", "treeshap": "TreeSHAP", "kernelshap": "KernelSHAP",
    "counterfactuals": "Counterfactuals", "dice": "Counterfactuals",
    "anchors": "Anchors", "ale": "ALE", "pfi": "PFI",
    "pdp": "PDP/ICE", "ice": "PDP/ICE", "pdp_ice": "PDP/ICE",
    "ig": "IntegratedGrad",
}

ALL_CASES = ["demo_regression", "demo_classification",
             "demo_nn_regression", "demo_nn_classification",
             "custom_regression", "custom_classification",
             "custom_nn_regression", "custom_nn_classification"]


def make_custom_regression_files(tmp_dir):
    """Create a regression dataset + saved model, like a user would provide."""
    X, y = make_regression(n_samples=400, n_features=6, n_informative=4,
                           noise=10.0, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["target"] = y
    csv_path = Path(tmp_dir) / "reg_dataset.csv"
    model_path = Path(tmp_dir) / "reg_model.joblib"
    df.to_csv(csv_path, index=False)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(df.drop(columns="target"), df["target"])
    joblib.dump(model, model_path)
    return csv_path, model_path


def build_case(case, tmp_dir):
    """Return (model, df, target_col, task_type) for the requested case."""
    if case == "demo_classification":
        from demo_models import create_demo
        d = create_demo("classification", random_state=config.random_state)
        df = d["X"].copy(); df[d["target_col"]] = d["y"]
        return d["model"], df, d["target_col"], "classification"

    if case == "demo_regression":
        from demo_models import create_demo
        d = create_demo("regression", random_state=config.random_state)
        df = d["X"].copy(); df[d["target_col"]] = d["y"]
        return d["model"], df, d["target_col"], "regression"

    if case == "custom_classification":
        data, X, y, model = load_inputs(config.data_path, config.model_path, "target")
        return model, data, "target", "classification"

    if case == "custom_regression":
        csv_path, model_path = make_custom_regression_files(tmp_dir)
        data, X, y, model = load_inputs(csv_path, model_path, "target")
        return model, data, "target", "regression"

    if case in ("demo_nn_classification", "demo_nn_regression"):
        from demo_models import create_demo
        task = case.split("_")[-1]
        d = create_demo("ig_nn", task, random_state=config.random_state)
        df = d["X"].copy(); df[d["target_col"]] = d["y"]
        return d["model"], df, d["target_col"], task

    if case in ("custom_nn_classification", "custom_nn_regression"):
        # Same NN as the demo, but saved to disk and loaded back like a user's own model
        from demo_models import create_demo
        task = case.split("_")[-1]
        d = create_demo("ig_nn", task, random_state=config.random_state)
        df = d["X"].copy(); df[d["target_col"]] = d["y"]
        csv_path = Path(tmp_dir) / f"nn_{task}.csv"
        model_path = Path(tmp_dir) / f"nn_{task}.joblib"
        df.to_csv(csv_path, index=False)
        joblib.dump(d["model"], model_path)
        data, X, y, model = load_inputs(csv_path, model_path, d["target_col"])
        return model, data, d["target_col"], task

    raise ValueError(case)


def run_case(case, tmp_dir, selected):
    out_dir = TEST_ROOT / case
    shutil.rmtree(out_dir, ignore_errors=True)  # fresh folder, so new files are detected
    out_dir.mkdir(parents=True, exist_ok=True)

    # ig.py reads config.run_mode at call time (demo -> trains nn_cl/nn_reg NN,
    # custom -> surrogate), so it must match the case being tested.
    config.run_mode = "demo" if case.startswith("demo") else "custom"

    model, df, target_col, task_type = build_case(case, tmp_dir)
    payload = build_unified_payload(
        model=model, df=df, target_col=target_col, task_type=task_type,
        scope=SCOPE, query_index=config.query_index,
        class_names=config.class_names, class_to_explain=config.class_to_explain,
        categorical_features_override=config.categorical_features_override,
        immutable_features=config.immutable_features,
        scoring_metric=config.scoring_metric,
        features_to_plot=config.features_to_plot,
        random_state=config.random_state, feature_metadata=config.feature_metadata,
    )
    prepared = prepare_xai_inputs(payload)

    rows = []
    methods = [m for m in METHODS if m[0] in selected]
    if "_nn_" in case:
        # a bare nn.Module has no predict()/predict_proba(): only IG can run
        skipped = [m[0] for m in methods if m[0] != "IntegratedGrad"]
        methods = [m for m in methods if m[0] == "IntegratedGrad"]
    else:
        skipped = []
    for name, module, func, needs_df in methods:
        before = {p for p in out_dir.rglob("*") if p.is_file()}
        plt.close("all")  # each method starts with a clean matplotlib state
        try:
            fn = getattr(importlib.import_module(module), func)
            kwargs = dict(unified_payload=payload, prepared_inputs=prepared,
                          output_dir=out_dir)
            if needs_df:
                kwargs["df"] = df
            result = fn(**kwargs)
            new = sorted(p.name for p in out_dir.rglob("*")
                         if p.is_file() and p not in before)
            n_png = sum(f.endswith(".png") for f in new)
            n_json = sum(f.endswith(".json") for f in new)
            status = "OK" if new else "OK (no files - skipped?)"
            rows.append((name, status, f"{n_png} png, {n_json} json", ""))
        except Exception as exc:
            # Locate the failing line: innermost frame + innermost project frame
            frames = traceback.extract_tb(exc.__traceback__)
            last = frames[-1]
            proj = [f for f in frames if "site-packages" not in f.filename.replace("\\", "/")]
            where = f"{Path(last.filename).name}:{last.lineno} in {last.name}"
            if proj:
                where += f"  <- called from {Path(proj[-1].filename).name}:{proj[-1].lineno}: {proj[-1].line}"
            rows.append((name, "ERROR", "", f"{type(exc).__name__}: {exc} @ {where}"))
            traceback.print_exc()
    for name in skipped:
        rows.append((name, "N/A", "", "nn.Module has no predict() - IG only"))
    return rows


def main():
    global SCOPE
    ap = argparse.ArgumentParser()
    ap.add_argument("methods", nargs="*", help="method keys (default: all)")
    ap.add_argument("--case", nargs="*", default=None, choices=ALL_CASES)
    ap.add_argument("--scope", default=SCOPE, choices=["local", "global", "both"])
    args = ap.parse_args()
    SCOPE = args.scope

    unknown = [m for m in args.methods if m.lower() not in METHOD_KEYS]
    if unknown:
        ap.error(f"unknown method(s) {unknown}. Choose from: {sorted(set(METHOD_KEYS))}")
    selected = ({METHOD_KEYS[m.lower()] for m in args.methods}
                if args.methods else {m[0] for m in METHODS})
    cases = args.case or ALL_CASES

    summary = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        for case in cases:
            print("\n" + "=" * 70 + f"\nCASE: {case}  (scope={SCOPE})\n" + "=" * 70)
            try:
                summary[case] = run_case(case, tmp_dir, selected)
            except Exception as exc:
                traceback.print_exc()
                summary[case] = [("SETUP", "ERROR", "", f"{type(exc).__name__}: {exc}")]

    print("\n\n" + "#" * 70 + "\nSUMMARY\n" + "#" * 70)
    for case, rows in summary.items():
        print(f"\n[{case}]")
        for name, status, files, err in rows:
            print(f"  {name:<16} {status:<26} {files:<18} {err}")
    print(f"\nFiles are in: {TEST_ROOT}")


if __name__ == "__main__":
    main()