"""
Support for models with several targets (multi-output).

    multi-class             one target column, 3+ classes
    multi-label             several 0/1 target columns
    multi-output            several categorical target columns (some with 3+ classes)
    multi-target regression several numeric target columns

The XAI methods explain one output at a time. For a model with several
targets, the framework runs every method once per target column, through a
"target view": a small sklearn-style model that returns only that target's
predict() / predict_proba() / classes_. The methods then see an ordinary
single-target model.
"""

import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.multioutput import (ClassifierChain, MultiOutputClassifier,
                                 MultiOutputRegressor, RegressorChain)


def as_target_list(target_col):
    """'y' -> ['y'], ['y1', 'y2'] -> ['y1', 'y2']"""
    return [target_col] if isinstance(target_col, str) else list(target_col)


def per_target(setting, target):
    """
    A setting may be one value for every target, or a dict {target: value}.
    Return the value for this target (None if the dict does not mention it).
    """
    if isinstance(setting, dict):
        return setting.get(target)
    return setting


def describe_problem(df, targets, task_type):
    """Name of the problem type, used in the console output."""
    if task_type == "regression":
        return "regression" if len(targets) == 1 else "multi-target regression"

    n_classes = [df[t].nunique() for t in targets]
    if len(targets) == 1:
        return "binary classification" if n_classes[0] <= 2 else "multi-class classification"
    if all(n <= 2 for n in n_classes):
        return "multi-label classification"
    return "multi-output classification"


def explained_classes(model, x_df, class_to_explain=None):
    """
    The classes to explain: a list of (label, index in model.classes_).

    class_to_explain:
        None         -> every class, one at a time; binary: the positive
                        class only (the other one is its mirror image)
        "predicted"  -> the class predicted for the explained row (x_df)
        a label      -> that class
    """
    classes = list(model.classes_)
    if class_to_explain is None:
        if len(classes) == 2:
            return [(classes[1], 1)]
        return list(zip(classes, range(len(classes))))

    label = model.predict(x_df)[0] if class_to_explain == "predicted" else class_to_explain
    if label not in classes:
        raise ValueError(
            f"class_to_explain={label!r} is not one of the model classes {classes}."
        )
    index = classes.index(label)
    return [(classes[index], index)]


def class_suffix(label, n_classes_explained):
    """File name suffix: '' for one explained class, '_class<label>' for several."""
    return "" if n_classes_explained == 1 else f"_class{label}"


def shap_for_class(explanation, class_index):
    """
    The SHAP values of one class (class_index None: regression, one output).
    A model with one output per row (e.g. gradient boosting log-odds) gives
    the score of classes_[1]; the score of classes_[0] is its mirror image.
    """
    if class_index is None:
        return explanation
    if explanation.values.ndim == np.ndim(explanation.data) + 1:
        return explanation[..., class_index]
    if class_index == 1:
        return explanation
    return type(explanation)(
        values=-explanation.values,
        base_values=-np.asarray(explanation.base_values),
        data=explanation.data,
        feature_names=explanation.feature_names,
    )


def to_json_value(value):
    """numpy scalars -> plain Python values, for json.dump."""
    return value.item() if isinstance(value, np.generic) else value


# --------------------------------------------------------------------------- #
#  Target views
# --------------------------------------------------------------------------- #

class _OutputSlice(nn.Module):
    """A torch module that keeps only one output column of another module."""

    def __init__(self, module, index):
        super().__init__()
        self.module = module
        self.index = index

    def forward(self, x):
        return self.module(x)[:, self.index:self.index + 1]


class _TargetViewBase(BaseEstimator):
    """Shared logic of the classifier and regressor views."""

    def __init__(self, model, target_index, target_name):
        # sklearn convention: store every constructor argument under its own name
        self.model = model
        self.target_index = target_index
        self.target_name = target_name

        # PyTorch wrappers are recognised by this flag (TreeSHAP, PFI, IG)
        self.is_torch_wrapper = bool(getattr(model, "is_torch_wrapper", False))
        if hasattr(model, "feature_names_in_"):
            self.feature_names_in_ = model.feature_names_in_
        if hasattr(model, "n_features_in_"):
            self.n_features_in_ = model.n_features_in_

    def predict(self, X):
        out = np.asarray(self.model.predict(X))
        return out[:, self.target_index] if out.ndim == 2 else out

    # ---- TreeSHAP ---------------------------------------------------------- #
    def tree_model_for_shap(self):
        """
        (tree model, output index) for TreeSHAP, or (None, reason).

        * MultiOutputRegressor / MultiOutputClassifier hold one model per
          target: that model is explained directly (output index None).
        * A native multi-output regressor (e.g. RandomForestRegressor) gives
          one SHAP column per target: the column is target_index.
        """
        if isinstance(self.model, (MultiOutputRegressor, MultiOutputClassifier)):
            return self.model.estimators_[self.target_index], None
        if isinstance(self.model, (ClassifierChain, RegressorChain)):
            return None, (
                "In a chain every model also takes the previous predictions as "
                "input, so TreeSHAP cannot explain it. Use KernelSHAP."
            )
        if isinstance(self, RegressorMixin):
            return self.model, self.target_index
        return None, (
            "SHAP does not give per-target values for a classifier trained on "
            "several target columns at once. Wrap it in "
            "sklearn.multioutput.MultiOutputClassifier, or use KernelSHAP."
        )

    @property
    def feature_importances_(self):
        # Used by PDP/ICE to pick the most important features
        tree_model, _ = self.tree_model_for_shap()
        source = tree_model if tree_model is not None else self.model
        if not hasattr(source, "feature_importances_"):
            raise AttributeError("feature_importances_")
        return source.feature_importances_

    # ---- Integrated Gradients ---------------------------------------------- #
    def as_torch_module(self):
        """The network output for this target only (PyTorch models)."""
        if not self.is_torch_wrapper:
            raise AttributeError("as_torch_module")
        return _OutputSlice(self.model.as_torch_module(), self.target_index)

    def __sklearn_is_fitted__(self):
        # Tells sklearn tools (PDP, PFI) that this model is already trained
        return True

    def fit(self, X=None, y=None):
        raise NotImplementedError(
            "The target view wraps an already trained model."
        )


class TargetClassifierView(ClassifierMixin, _TargetViewBase):
    """One classification target of a multi-label / multi-output classifier."""

    def __init__(self, model, target_index, target_name, classes):
        super().__init__(model, target_index, target_name)
        self.classes = classes
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X):
        proba = self.model.predict_proba(X)

        # sklearn RandomForest / MultiOutputClassifier: list with one (n, n_classes)
        # array per target
        if isinstance(proba, list):
            return np.asarray(proba[self.target_index])

        # MLPClassifier, KNeighbors (multi-label): (n, n_targets) array holding the
        # probability of the positive class of every target
        proba = np.asarray(proba)
        if proba.ndim == 2 and len(self.classes_) == 2:
            p1 = proba[:, self.target_index]
            return np.column_stack([1.0 - p1, p1])

        raise ValueError(
            f"Cannot read the probabilities of target '{self.target_name}' from a "
            f"predict_proba output of shape {proba.shape}."
        )


class TargetRegressorView(RegressorMixin, _TargetViewBase):
    """One target of a multi-target regressor."""


def _classes_of_target(model, target_index, y_target):
    """Class labels of one target, in the order of its predict_proba columns."""
    model_classes = getattr(model, "classes_", None)
    # sklearn multi-output classifiers: one array of classes per target
    if isinstance(model_classes, list) and len(model_classes) > target_index:
        return np.asarray(model_classes[target_index])
    # Multi-label models that only output the positive probability per target
    return np.sort(pd.unique(y_target))


def make_target_view(model, targets, active_target, task_type, df):
    """
    A single-target view of a model trained on several targets.
    For one target the model is returned unchanged.
    """
    if len(targets) == 1:
        return model

    index = targets.index(active_target)
    if task_type == "classification":
        classes = _classes_of_target(model, index, df[active_target])
        return TargetClassifierView(model, index, active_target, classes)
    return TargetRegressorView(model, index, active_target)


# --------------------------------------------------------------------------- #
#  Checks
# --------------------------------------------------------------------------- #

def check_output_count(model, query_df, targets):
    """
    Fail early, with a clear message, when the number of model outputs does
    not match the number of target columns (otherwise the methods would run
    and silently explain the wrong thing).
    """
    out = np.asarray(model.predict(query_df))
    n_outputs = 1 if out.ndim == 1 else out.shape[1]

    if n_outputs != len(targets):
        raise ValueError(
            f"{len(targets)} target columns were given {targets}, but the model "
            f"predicts {n_outputs} output(s) per row. "
            f"Check the given target columns to match the "
            f"{n_outputs} output(s)."
        )
