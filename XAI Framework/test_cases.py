"""
Runs the XAI framework for the basic and the multi-output cases and reports,
per case, target and method, whether it worked and which files it produced.

    python test_cases.py                          # all methods, the 8 basic cases
    python test_cases.py anchors                  # one method, the 8 basic cases
    python test_cases.py treeshap kernelshap      # several methods
    python test_cases.py anchors --case demo_regression custom_nn_classification
    python test_cases.py anchors --scope local
    python test_cases.py --multi                  # all methods, the 7 multi-output cases
    python test_cases.py --all                    # all methods, all 15 cases

Method keys: lime treeshap kernelshap counterfactuals anchors ale pfi pdp ig

Cases : sklearn model  -> demo / custom  x  classification / regression   (4 cases)
        PyTorch NN     -> demo / custom  x  classification / regression   (4 cases; the network is
                          wrapped by torch_wrapper.py, so every method runs. TreeSHAP skips
                          itself: it only explains tree models)
        Multi-output   -> multi-class, multi-label, multi-output classification and
                          multi-target regression, with sklearn and PyTorch (7 cases).
                          Every method runs once per target column.
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
import torch
import torch.nn as nn
from sklearn.datasets import (make_classification, make_multilabel_classification,
                              make_regression)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

import config
from data_loader import load_inputs
from pipeline import build_unified_payload
from framework_adapter import prepare_xai_inputs
from multi_output import as_target_list

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

BASIC_CASES = ["demo_regression", "demo_classification",
               "demo_nn_regression", "demo_nn_classification",
               "custom_regression", "custom_classification",
               "custom_nn_regression", "custom_nn_classification"]

MULTI_CASES = ["custom_multiclass", "custom_nn_multiclass",
               "custom_multilabel", "custom_nn_multilabel",
               "custom_multioutput_classification",
               "custom_multitarget_regression", "custom_nn_multitarget_regression"]

ALL_CASES = BASIC_CASES + MULTI_CASES


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


def train_torch_net(X, Y, n_outputs, loss_fn, epochs=400):
    """Small MLP trained on standardized features. Returns (net, scaler)."""
    torch.manual_seed(config.random_state)
    scaler = StandardScaler().fit(X)
    xt = torch.tensor(scaler.transform(X), dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], 32), nn.ReLU(), nn.Linear(32, n_outputs))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(net(xt), Y)
        loss.backward()
        optimizer.step()
    return net.eval(), scaler


def save_and_load(df, model, targets, tmp_dir, name):
    """Save the dataset and model to disk and load them back, like a user's own files."""
    csv_path = Path(tmp_dir) / f"{name}.csv"
    model_path = Path(tmp_dir) / f"{name}.joblib"
    df.to_csv(csv_path, index=False)
    joblib.dump(model, model_path)
    target_col = targets[0] if len(targets) == 1 else targets
    data, X, y, model = load_inputs(csv_path, model_path, target_col)
    return model, data, target_col


def build_multi_case(case, tmp_dir):
    """Cases with 3+ classes or several target columns."""
    rs = config.random_state
    features = [f"f{i}" for i in range(5)]

    if case in ("custom_multiclass", "custom_nn_multiclass"):
        # One target column, 3 classes
        X, y = make_classification(n_samples=300, n_features=5, n_informative=4,
                                   n_redundant=0, n_classes=3, random_state=rs)
        df = pd.DataFrame(X, columns=features)
        df["species"] = y
        targets = ["species"]
        if case == "custom_multiclass":
            model = RandomForestClassifier(n_estimators=50, random_state=rs)
            model.fit(df[features], df["species"])
        else:
            net, scaler = train_torch_net(X, torch.tensor(y, dtype=torch.long), 3,
                                          nn.CrossEntropyLoss())
            model = {"model": net, "scaler": scaler}
        return (*save_and_load(df, model, targets, tmp_dir, case), "classification")

    if case in ("custom_multilabel", "custom_nn_multilabel"):
        # Three 0/1 target columns
        X, Y = make_multilabel_classification(n_samples=300, n_features=5, n_classes=3,
                                              random_state=rs)
        df = pd.DataFrame(X, columns=features)
        targets = ["label_a", "label_b", "label_c"]
        df[targets] = Y
        if case == "custom_multilabel":
            # Native multi-output forest: TreeSHAP skips it (see multi_output.py)
            model = RandomForestClassifier(n_estimators=50, random_state=rs)
            model.fit(df[features], df[targets])
        else:
            net, scaler = train_torch_net(X, torch.tensor(Y, dtype=torch.float32), 3,
                                          nn.BCEWithLogitsLoss())
            model = {"model": net, "scaler": scaler}
        return (*save_and_load(df, model, targets, tmp_dir, case), "classification")

    if case == "custom_multioutput_classification":
        # One 0/1 target and one 3-class target, one forest per target
        X, y = make_classification(n_samples=300, n_features=5, n_informative=4,
                                   n_redundant=0, n_classes=3, random_state=rs)
        df = pd.DataFrame(X, columns=features)
        df["risk"] = y
        df["flag"] = (X[:, 0] + X[:, 1] > 0).astype(int)
        targets = ["flag", "risk"]
        model = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=rs))
        model.fit(df[features], df[targets])
        return (*save_and_load(df, model, targets, tmp_dir, case), "classification")

    if case in ("custom_multitarget_regression", "custom_nn_multitarget_regression"):
        # Several numeric target columns
        n_targets = 3 if case == "custom_multitarget_regression" else 2
        X, Y = make_regression(n_samples=300, n_features=5, n_informative=4,
                               n_targets=n_targets, noise=10.0, random_state=rs)
        df = pd.DataFrame(X, columns=features)
        targets = [f"y{i + 1}" for i in range(n_targets)]
        df[targets] = Y
        if case == "custom_multitarget_regression":
            model = RandomForestRegressor(n_estimators=50, random_state=rs)
            model.fit(df[features], df[targets])
        else:
            net, scaler = train_torch_net(X, torch.tensor(Y, dtype=torch.float32),
                                          n_targets, nn.MSELoss(), epochs=800)
            model = {"model": net, "scaler": scaler}
        return (*save_and_load(df, model, targets, tmp_dir, case), "regression")

    raise ValueError(case)


def build_case(case, tmp_dir):
    """Return (model, df, target_col, task_type) for the requested case."""
    if case in MULTI_CASES:
        return build_multi_case(case, tmp_dir)

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
        data, X, y, model = load_inputs(Path(config.base_dir) / "data" / "my_dataset.csv",
                                         Path(config.base_dir) / "models" / "my_model.joblib", "target")
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
        # Same as main.py: the network travels with the scaler it was trained with
        model = {"model": d["model"], "scaler": d["preprocessing"]}
        return model, df, d["target_col"], task

    if case in ("custom_nn_classification", "custom_nn_regression"):
        # Same NN as the demo, but saved to disk and loaded back like a user's own model.
        # It is saved together with its scaler, as a dict.
        from demo_models import create_demo
        task = case.split("_")[-1]
        d = create_demo("ig_nn", task, random_state=config.random_state)
        df = d["X"].copy(); df[d["target_col"]] = d["y"]
        csv_path = Path(tmp_dir) / f"nn_{task}.csv"
        model_path = Path(tmp_dir) / f"nn_{task}.joblib"
        df.to_csv(csv_path, index=False)
        joblib.dump({"model": d["model"], "scaler": d["preprocessing"]}, model_path)
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
    targets = as_target_list(target_col)

    rows = []
    # Same as main.py: a model with several targets is explained one target at a time
    for active_target in targets:
        payload = build_unified_payload(
            model=model, df=df, target_col=target_col, task_type=task_type,
            scope=SCOPE, active_target=active_target, query_index=config.query_index,
            class_names=config.class_names, class_to_explain=config.class_to_explain,
            categorical_number_features=config.categorical_number_features,
            immutable_features=config.immutable_features,
            scoring_metric=config.scoring_metric,
            features_to_plot=config.features_to_plot,
            random_state=config.random_state, feature_metadata=config.feature_metadata,
        )
        prepared = prepare_xai_inputs(payload)
        print(f"\n--- {payload['metadata']['problem_type']} | target: {active_target} ---")
        label = f" [{active_target}]" if len(targets) > 1 else ""
        rows += run_methods(out_dir, payload, prepared, df, selected, label)
    return rows


def run_methods(out_dir, payload, prepared, df, selected, label):
    rows = []
    methods = [m for m in METHODS if m[0] in selected]
    for name, module, func, needs_df in methods:
        name += label
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
    return rows


def main():
    global SCOPE
    ap = argparse.ArgumentParser()
    ap.add_argument("methods", nargs="*", help="method keys (default: all)")
    ap.add_argument("--case", nargs="*", default=None, choices=ALL_CASES)
    ap.add_argument("--scope", default=SCOPE, choices=["local", "global", "both"])
    ap.add_argument("--multi", action="store_true", help="run the multi-output cases")
    ap.add_argument("--all", action="store_true", help="run the basic and the multi-output cases")
    args = ap.parse_args()
    SCOPE = args.scope

    unknown = [m for m in args.methods if m.lower() not in METHOD_KEYS]
    if unknown:
        ap.error(f"unknown method(s) {unknown}. Choose from: {sorted(set(METHOD_KEYS))}")
    selected = ({METHOD_KEYS[m.lower()] for m in args.methods}
                if args.methods else {m[0] for m in METHODS})
    if args.case:
        cases = args.case
    elif args.all:
        cases = ALL_CASES
    elif args.multi:
        cases = MULTI_CASES
    else:
        cases = BASIC_CASES

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
            print(f"  {name:<28} {status:<26} {files:<18} {err}")
    print(f"\nFiles are in: {TEST_ROOT}")


if __name__ == "__main__":
    main()