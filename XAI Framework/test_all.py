"""
Tests every case of the framework in one run: the 8 basic cases and the 7
multi-class / multi-label / multi-output / multi-target cases.

For every case and target it checks two things:

1. Every XAI method runs and writes its files (or skips itself where it is
   expected to: TreeSHAP on neural networks and on a native multi-label
   forest, Anchors on regression).
2. The results are correct:
   view      the target view returns the same predictions as the full model
   proba     probabilities sum to 1, and the most likely class is the prediction
   outputs   a wrong number of target columns stops with an error
   additive  TreeSHAP: base value + sum of SHAP values = model prediction
   lime      the default explains every class (binary: the positive one),
             and LIME wrote one file per explained class
   names     every output file name ends with its target name

    python test_all.py                          # all methods, all 15 cases
    python test_all.py --multi                  # only the 7 multi-output cases
    python test_all.py --basic                  # only the 8 basic cases
    python test_all.py lime treeshap            # some methods only
    python test_all.py --case custom_multilabel custom_multiclass
    python test_all.py --scope local

Method keys: lime treeshap kernelshap counterfactuals anchors ale pfi pdp ig

Exit code 0 if nothing failed, 1 otherwise.
Output: outputs_test/<case>/{plots,raw_results}   (your normal outputs/ is untouched)
"""
import argparse
import importlib
import json
import re
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no windows, save only
import matplotlib.pyplot as plt
import numpy as np
import shap

import config
from pipeline import build_unified_payload
from framework_adapter import prepare_xai_inputs
from multi_output import (_TargetViewBase, as_target_list, check_output_count, class_suffix,
                          per_target, to_json_value)
from test_cases import (ALL_CASES, BASIC_CASES, METHOD_KEYS, METHODS, MULTI_CASES,
                        TEST_ROOT, build_case)


def expected_skip(case, method, task_type):
    """Methods that skip themselves on purpose in this case."""
    if method == "TreeSHAP":
        # Only tree models; a native multi-label forest has no per-target SHAP values
        return "_nn_" in case or case == "custom_multilabel"
    if method == "Anchors":
        return task_type == "regression"
    return False


# --------------------------------------------------------------------------- #
#  Running the methods
# --------------------------------------------------------------------------- #

def run_method(out_dir, payload, prepared, df, method):
    """Run one method. Returns (list of new file names, error text or None)."""
    name, module, func, needs_df = method
    before = {p for p in out_dir.rglob("*") if p.is_file()}
    plt.close("all")  # each method starts with a clean matplotlib state
    try:
        fn = getattr(importlib.import_module(module), func)
        kwargs = dict(unified_payload=payload, prepared_inputs=prepared,
                      output_dir=out_dir)
        if needs_df:
            kwargs["df"] = df
        fn(**kwargs)
    except Exception as exc:
        traceback.print_exc()
        frames = traceback.extract_tb(exc.__traceback__)
        proj = [f for f in frames if "site-packages" not in f.filename.replace("\\", "/")]
        where = f" @ {Path(proj[-1].filename).name}:{proj[-1].lineno}" if proj else ""
        return [], f"{type(exc).__name__}: {exc}{where}"
    new = sorted(p.name for p in out_dir.rglob("*") if p.is_file() and p not in before)
    return new, None


def method_row(case, method_name, task_type, files, error):
    """(status, detail) of one method run."""
    if error:
        return "FAIL", error
    skip = expected_skip(case, method_name, task_type)
    if files and skip:
        return "FAIL", "should have skipped itself but wrote files"
    if not files and skip:
        return "SKIP", "expected"
    if not files:
        return "FAIL", "no files written"
    n_png = sum(f.endswith(".png") for f in files)
    n_json = sum(f.endswith(".json") for f in files)
    return "OK", f"{n_png} png, {n_json} json"


# --------------------------------------------------------------------------- #
#  Correctness checks
# --------------------------------------------------------------------------- #

def full_model_of(model):
    """The model with all outputs (the view wraps it; single-target: itself)."""
    return model.model if isinstance(model, _TargetViewBase) else model


def check_view(model, X):
    """The view returns the active target's column of the full model."""
    if not isinstance(model, _TargetViewBase):
        return None
    full = np.asarray(model.model.predict(X))
    mine = np.asarray(model.predict(X))
    if not np.array_equal(full[:, model.target_index], mine):
        return "the view predictions differ from the full model column"
    return ""


def check_proba(model, X, task_type):
    """Valid probabilities that agree with predict()."""
    if task_type != "classification":
        return None
    proba = np.asarray(model.predict_proba(X))
    classes = np.asarray(model.classes_)
    if proba.shape != (len(X), len(classes)):
        return f"predict_proba shape {proba.shape}, expected {(len(X), len(classes))}"
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
        return "probabilities do not sum to 1"
    agree = np.mean(classes[proba.argmax(axis=1)] == np.asarray(model.predict(X)))
    if agree < 0.99:  # allows for exact ties
        return f"most likely class = prediction for only {agree:.0%} of rows"
    return ""


def check_output_mismatch(model, X, targets):
    """A wrong number of target columns must raise a clear ValueError."""
    full = full_model_of(model)
    wrong = targets[:1] if len(targets) > 1 else targets + ["__extra__"]
    try:
        check_output_count(full, X, wrong)
    except ValueError:
        return ""
    return f"no error for the wrong target list {wrong}"


def check_treeshap_additivity(model, X, task_type):
    """base value + sum of SHAP values = prediction of this target."""
    if getattr(model, "is_torch_wrapper", False):
        return None
    if isinstance(model, _TargetViewBase):
        tree_model, output_index = model.tree_model_for_shap()
        if tree_model is None:
            return None
    else:
        tree_model, output_index = model, None
    try:
        explanation = shap.TreeExplainer(tree_model)(X)
    except Exception:
        return None  # not a tree model

    values, base = explanation.values, explanation.base_values
    if output_index is not None:
        values, base = values[..., output_index], base[..., output_index]
    reconstructed = base + values.sum(axis=1)

    if task_type == "classification":
        if reconstructed.ndim == 1:
            return None  # log-odds output, not comparable with probabilities
        prediction = model.predict_proba(X)
    else:
        prediction = model.predict(X)
    error = np.max(np.abs(reconstructed - prediction))
    if error > 1e-4:
        return f"SHAP sum differs from the prediction by {error:.2e}"
    return ""


def check_lime_classes(model, classes_to_explain, out_dir, target, task_type):
    """
    The default explains every class (binary: the positive one), and LIME
    wrote one file per explained class, with that class inside.
    """
    raw_dir = out_dir / "raw_results"
    if task_type != "classification" or not list(raw_dir.glob(f"lime_{target}*.json")):
        return None
    labels = [to_json_value(label) for label, _ in classes_to_explain]
    if per_target(config.class_to_explain, target) is None:
        classes = [to_json_value(c) for c in model.classes_]
        expected = classes[1:] if len(classes) == 2 else classes
        if labels != expected:
            return f"default explains classes {labels}, expected {expected}"
    for label in labels:
        path = raw_dir / f"lime_{target}{class_suffix(label, len(labels))}.json"
        if not path.exists():
            return f"missing {path.name}"
        explained = json.loads(path.read_text(encoding="utf-8")).get("explained_class")
        if explained != label:
            return f"{path.name} explains class {explained!r}, expected {label!r}"
    return ""


def check_file_names(files, target, targets):
    """
    With several targets, every file name ends with its own target name
    (followed by _class<label> when several classes are explained).
    """
    if len(targets) == 1 or not files:
        return None
    pattern = re.compile(rf"_{re.escape(str(target))}(_class.+)?$")
    wrong = [f for f in files if not pattern.search(Path(f).stem)]
    if wrong:
        return f"file names without the target name: {wrong[:3]}"
    return ""


def run_checks(model, classes_to_explain, payload, out_dir, target, targets, task_type, files):
    """[(check name, status, detail)] for one target."""
    X = payload["reference_dataset"].head(50)
    checks = [
        ("view",     lambda: check_view(model, X)),
        ("proba",    lambda: check_proba(model, X, task_type)),
        ("outputs",  lambda: check_output_mismatch(model, X, targets)),
        ("additive", lambda: check_treeshap_additivity(model, X, task_type)),
        ("lime",     lambda: check_lime_classes(model, classes_to_explain, out_dir, target, task_type)),
        ("names",    lambda: check_file_names(files, target, targets)),
    ]
    rows = []
    for name, check in checks:
        try:
            problem = check()
        except Exception as exc:
            traceback.print_exc()
            problem = f"{type(exc).__name__}: {exc}"
        if problem is None:
            continue  # does not apply to this case
        rows.append((f"check: {name}", "FAIL" if problem else "OK", problem))
    return rows


# --------------------------------------------------------------------------- #
#  Cases
# --------------------------------------------------------------------------- #

def run_case(case, tmp_dir, methods, scope):
    out_dir = TEST_ROOT / case
    shutil.rmtree(out_dir, ignore_errors=True)  # fresh folder, so new files are detected
    out_dir.mkdir(parents=True, exist_ok=True)

    # ig.py reads config.run_mode at call time, so it must match the case
    config.run_mode = "demo" if case.startswith("demo") else "custom"

    model, df, target_col, task_type = build_case(case, tmp_dir)
    targets = as_target_list(target_col)

    rows = []
    # Same as main.py: a model with several targets is explained one target at a time
    for target in targets:
        payload = build_unified_payload(
            model=model, df=df, target_col=target_col, task_type=task_type,
            scope=scope, active_target=target, query_index=config.query_index,
            class_names=config.class_names, class_to_explain=config.class_to_explain,
            categorical_number_features=config.categorical_number_features,
            immutable_features=config.immutable_features,
            scoring_metric=config.scoring_metric,
            features_to_plot=config.features_to_plot,
            random_state=config.random_state, feature_metadata=config.feature_metadata,
        )
        prepared = prepare_xai_inputs(payload)
        problem_type = payload["metadata"]["problem_type"]
        print(f"\n--- {case} | {problem_type} | target: {target} ---")

        label = f" [{target}]" if len(targets) > 1 else ""
        target_files = []
        for method in methods:
            files, error = run_method(out_dir, payload, prepared, df, method)
            target_files += files
            status, detail = method_row(case, method[0], task_type, files, error)
            rows.append((method[0] + label, status, detail))

        for name, status, detail in run_checks(prepared["model"], prepared["classes_to_explain"],
                                               payload, out_dir, target, targets, task_type,
                                               target_files):
            rows.append((name + label, status, detail))
    return problem_type, rows


def main():
    ap = argparse.ArgumentParser(description="Test every case of the XAI framework.")
    ap.add_argument("methods", nargs="*", help="method keys (default: all)")
    ap.add_argument("--case", nargs="*", default=None, choices=ALL_CASES)
    ap.add_argument("--basic", action="store_true", help="only the 8 basic cases")
    ap.add_argument("--multi", action="store_true", help="only the 7 multi-output cases")
    ap.add_argument("--scope", default="both", choices=["local", "global", "both"])
    args = ap.parse_args()

    unknown = [m for m in args.methods if m.lower() not in METHOD_KEYS]
    if unknown:
        ap.error(f"unknown method(s) {unknown}. Choose from: {sorted(set(METHOD_KEYS))}")
    selected = ({METHOD_KEYS[m.lower()] for m in args.methods}
                if args.methods else {m[0] for m in METHODS})
    methods = [m for m in METHODS if m[0] in selected]

    if args.case:
        cases = args.case
    elif args.basic:
        cases = BASIC_CASES
    elif args.multi:
        cases = MULTI_CASES
    else:
        cases = ALL_CASES

    summary = {}
    start = time.time()
    with tempfile.TemporaryDirectory() as tmp_dir:
        for case in cases:
            print("\n" + "=" * 70 + f"\nCASE: {case}  (scope={args.scope})\n" + "=" * 70)
            try:
                summary[case] = run_case(case, tmp_dir, methods, args.scope)
            except Exception as exc:
                traceback.print_exc()
                summary[case] = ("?", [("SETUP", "FAIL", f"{type(exc).__name__}: {exc}")])

    print("\n\n" + "#" * 70 + "\nSUMMARY\n" + "#" * 70)
    counts = {"OK": 0, "SKIP": 0, "FAIL": 0}
    failed_cases = []
    for case, (problem_type, rows) in summary.items():
        case_failed = any(status == "FAIL" for _, status, _ in rows)
        if case_failed:
            failed_cases.append(case)
        print(f"\n[{case}]  {problem_type}  -> {'FAIL' if case_failed else 'OK'}")
        for name, status, detail in rows:
            counts[status] += 1
            print(f"  {name:<36} {status:<5} {detail}")

    minutes = (time.time() - start) / 60
    print("\n" + "#" * 70)
    print(f"{len(cases)} cases, {minutes:.1f} min:  "
          f"{counts['OK']} OK, {counts['SKIP']} expected skips, {counts['FAIL']} FAIL")
    if failed_cases:
        print("Failed cases: " + ", ".join(failed_cases))
    print(f"Files are in: {TEST_ROOT}")
    sys.exit(1 if failed_cases else 0)


if __name__ == "__main__":
    main()
