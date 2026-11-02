"""
Wrapper that lets any tabular PyTorch nn.Module be used by all XAI methods.

The XAI methods (LIME, SHAP, DiCE, Anchors, ALE, PFI, PDP/ICE) expect the
scikit-learn interface:
    model.predict(X), model.predict_proba(X), model.classes_
A bare nn.Module has none of these. The wrapper adds them on top of the network.
"""

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

_PROB_LAYERS = (nn.Sigmoid, nn.Softmax)
_LOG_PROB_LAYERS = (nn.LogSoftmax,)


def is_wrapped(obj):
    """True if the object is already a wrapper defined in this file."""
    return bool(getattr(obj, "is_torch_wrapper", False))


def _infer_output_type(net):
    """Guess whether the network outputs logits or probabilities."""
    leaves = [m for m in net.modules() if not list(m.children())]
    last = leaves[-1] if leaves else None
    if isinstance(last, _LOG_PROB_LAYERS):
        return "log_probabilities"
    if isinstance(last, _PROB_LAYERS):
        return "probabilities"
    return "logits"


def _infer_n_features(net):
    """Number of input features from the first nn.Linear layer, else None."""
    for m in net.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    return None

def _affine_from_scaler(scaler, feature_names, n_features):
    """
    Return (shift, scale) such that  x_scaled = (x - shift) / scale.
    Supports StandardScaler, MinMaxScaler and {"mean": ..., "std": ...}.
    """
    if scaler is None:
        return None, None

    def as_vector(values, name):
        # A pandas Series is aligned to the feature order by its index
        if isinstance(values, pd.Series) and feature_names is not None:
            if set(feature_names).issubset(set(values.index)):
                values = values.reindex(feature_names)
        # np.array (not asarray) makes a writable copy; pandas may hand back read-only data
        arr = np.array(values, dtype=np.float64).ravel()
        if arr.size != n_features:
            raise ValueError(
                f"Scaler '{name}' has {arr.size} values but the model has "
                f"{n_features} features."
            )
        return arr

    if isinstance(scaler, dict):
        if "mean" not in scaler or "std" not in scaler:
            raise ValueError("A scaler dict needs the keys 'mean' and 'std'.")
        shift = as_vector(scaler["mean"], "mean")
        scale = as_vector(scaler["std"], "std")
    elif hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        # sklearn StandardScaler (mean_ / scale_ may be None if disabled)
        shift = (as_vector(scaler.mean_, "mean_")
                 if getattr(scaler, "mean_", None) is not None
                 else np.zeros(n_features))
        scale = (as_vector(scaler.scale_, "scale_")
                 if getattr(scaler, "scale_", None) is not None
                 else np.ones(n_features))
    elif hasattr(scaler, "min_") and hasattr(scaler, "scale_"):
        # sklearn MinMaxScaler computes x_scaled = x * scale_ + min_,
        # which is the same affine form with shift = -min_/scale_, scale = 1/scale_
        mm_scale = as_vector(scaler.scale_, "scale_")
        mm_min = as_vector(scaler.min_, "min_")
        scale = 1.0 / mm_scale
        shift = -mm_min / mm_scale
    else:
        raise TypeError(
            "Unsupported scaler. Use sklearn StandardScaler, MinMaxScaler or a "
            "dict {'mean': ..., 'std': ...}, or put the preprocessing inside the "
            "model's forward()."
        )

    # Avoid division by zero for constant features
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return shift, scale


class _FusedModule(nn.Module):
    """
    x_raw -> (x_raw - shift) / scale -> net -> 2-D output

    This is what Integrated Gradients receives, so attributions are in raw
    feature units. The output is always 2-D: (n_samples, n_outputs).
    """

    def __init__(self, net, shift=None, scale=None):
        super().__init__()
        self.net = net
        try:
            self._dtype = next(net.parameters()).dtype
        except StopIteration:
            self._dtype = torch.float32
        if shift is None:
            self.register_buffer("shift", None)
            self.register_buffer("scale", None)
        else:
            self.register_buffer("shift", torch.as_tensor(shift, dtype=self._dtype))
            self.register_buffer("scale", torch.as_tensor(scale, dtype=self._dtype))

    def forward(self, x):
        x = x.to(self._dtype)
        if self.shift is not None:
            x = (x - self.shift) / self.scale
        out = self.net(x)
        # Normalize the output shape so callers can always rely on 2-D
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        elif out.dim() > 2:
            out = out.reshape(out.shape[0], -1)
        return out


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # subtract max for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # numerically stable form

class _TorchWrapperBase(BaseEstimator):
    """Shared logic: input handling, batching, access to the fused module."""

    # Other parts of the framework check this flag (e.g. TreeSHAP, PFI)
    is_torch_wrapper = True

    def _setup(self, model, feature_names, scaler, device, batch_size):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module.")

        # Number of features: from the names if given, else from the first Linear layer
        n_features = (len(feature_names) if feature_names is not None
                      else _infer_n_features(model))
        if n_features is None:
            raise ValueError(
                "Could not determine the number of input features. Pass "
                "feature_names (the list of feature columns)."
            )

        shift, scale = _affine_from_scaler(scaler, feature_names, n_features)

        # Trailing underscore = sklearn convention for "learned/derived" attributes
        self.n_features_in_ = int(n_features)
        self.feature_names_in_ = (np.asarray(list(feature_names), dtype=object)
                                  if feature_names is not None else None)
        self._device = torch.device(device)
        self._batch_size = int(batch_size)

        # scaler + network in one module, in eval mode (no dropout, fixed batchnorm)
        self._fused = _FusedModule(model, shift, scale).to(self._device).eval()
        self._checked_output = False

    # ---- input / output plumbing ------------------------------------------ #
    def _to_array(self, X):
        """Turn a DataFrame / array / single row into a float32 array (n, features)."""
        if isinstance(X, pd.DataFrame):
            # Match columns by name, so a different column order still works
            names = self.feature_names_in_
            if names is not None and set(X.columns) == set(names):
                X = X[list(names)]
            try:
                X = X.to_numpy(dtype=np.float32)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    "PyTorch models need numeric features. Encode categorical "
                    "columns before training the network."
                ) from exc
        else:
            X = np.asarray(X, dtype=np.float32)

        # A single row [a, b, c] becomes [[a, b, c]]
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected input with {self.n_features_in_} features, "
                f"got shape {X.shape}."
            )
        return X

    def _raw_output(self, X):
        """Network output as a 2-D float64 array, computed in batches."""
        X = self._to_array(X)
        self._fused.eval()
        chunks = []
        with torch.no_grad():  # no gradients needed, saves memory and time
            for start in range(0, len(X), self._batch_size):
                # torch.tensor copies the data; pandas may hand back read-only arrays
                xb = torch.tensor(X[start:start + self._batch_size])
                chunks.append(self._fused(xb.to(self._device)).cpu())
        out = torch.cat(chunks, dim=0).numpy().astype(np.float64)
        if not np.isfinite(out).all():
            raise ValueError("The network produced NaN or infinite outputs.")
        return out

    # ---- access for Integrated Gradients ---------------------------------- #
    def as_torch_module(self):
        """
        The network as a differentiable module taking RAW features and
        returning a 2-D output (logits for classification). Used by IG.
        """
        return self._fused

    @property
    def torch_model(self):
        """The original, unmodified nn.Module."""
        return self._fused.net

    def __sklearn_is_fitted__(self):
        # Tells sklearn tools (PDP, PFI) that this model is already trained
        return True

    def fit(self, X=None, y=None):
        """
        Present only because scikit-learn tools (permutation_importance,
        PartialDependenceDisplay) require the attribute. The network is already
        trained, so training through the wrapper is deliberately not supported.
        """
        raise NotImplementedError(
            "The wrapped PyTorch network is already trained; train it with "
            "PyTorch and wrap the result."
        )

class TorchClassifierWrapper(ClassifierMixin, _TorchWrapperBase):
    """
    sklearn-style classifier on top of a PyTorch network.

    Parameters
    ----------
    model : nn.Module
        Returns (n, 1) for binary problems, or (n, k) for k classes.
    feature_names : list of str, optional
        Column order the network was trained with.
    scaler : StandardScaler | MinMaxScaler | dict, optional
        Preprocessing applied before the network.
    output_type : {"auto", "logits", "probabilities", "log_probabilities"}
        What the network returns. "auto" looks at its last layer
        (Sigmoid/Softmax -> probabilities, LogSoftmax -> log-probabilities,
        otherwise logits).
    classes : sequence, optional
        Class labels, in output order. Default: 0 .. n_classes-1.
    """

    def __init__(self, model, feature_names=None, scaler=None,
                 output_type="auto", classes=None, device="cpu",
                 batch_size=4096):
        # sklearn convention: store every constructor argument under its own name
        self.model = model
        self.feature_names = feature_names
        self.scaler = scaler
        self.output_type = output_type
        self.classes = classes
        self.device = device
        self.batch_size = batch_size

        valid = ("auto", "logits", "probabilities", "log_probabilities")
        if output_type not in valid:
            raise ValueError(f"output_type must be one of {valid}.")

        self._setup(model, feature_names, scaler, device, batch_size)
        self._output_type_auto = output_type == "auto"
        self.output_type_ = (_infer_output_type(model) if self._output_type_auto
                             else output_type)

        # Dry run on zeros to learn how many output units the network has
        n_out = _dry_run_outputs(self)
        self._n_outputs = n_out

        # One output unit = binary problem = 2 classes
        n_classes = 2 if n_out == 1 else n_out
        if classes is None:
            self.classes_ = np.arange(n_classes)
        else:
            self.classes_ = np.asarray(classes)
            if len(self.classes_) != n_classes:
                raise ValueError(
                    f"The network has {n_classes} classes but {len(self.classes_)} "
                    f"class labels were given."
                )

    def _warn_if_probabilities(self, raw):
        """One-time hint when 'auto' guessed logits but outputs look like probs."""
        if self._checked_output:
            return
        self._checked_output = True
        if not (self._output_type_auto and self.output_type_ == "logits"):
            return
        if len(raw) < 2:
            return
        in_unit = (raw.min() >= 0.0) and (raw.max() <= 1.0)
        sums_to_one = raw.shape[1] == 1 or np.allclose(raw.sum(axis=1), 1.0, atol=1e-3)
        if in_unit and sums_to_one:
            warnings.warn(
                "The network outputs look like probabilities, but they are being "
                "treated as logits. If the model already applies sigmoid/softmax, "
                "set output_type='probabilities' (config.torch_output_type).",
                stacklevel=3,
            )

    def predict_proba(self, X):
        raw = self._raw_output(X)
        self._warn_if_probabilities(raw)
        kind = self.output_type_

        if raw.shape[1] == 1:  # binary, single output unit
            if kind == "logits":
                p1 = _sigmoid(raw[:, 0])
            elif kind == "log_probabilities":
                p1 = np.exp(raw[:, 0])
            else:
                p1 = raw[:, 0]
            p1 = np.clip(p1, 0.0, 1.0)
            return np.column_stack([1.0 - p1, p1])

        # Multi-class: k output units
        if kind == "logits":
            return _softmax(raw)
        if kind == "log_probabilities":
            return np.exp(raw)
        proba = np.clip(raw, 0.0, None)
        totals = proba.sum(axis=1, keepdims=True)
        return proba / np.where(totals == 0, 1.0, totals)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

def _dry_run_outputs(wrapper):
    """Run the network once on zeros and return its number of output units."""
    try:
        return wrapper._raw_output(
            np.zeros((1, wrapper.n_features_in_), dtype=np.float32)).shape[1]
    except Exception as exc:
        raise RuntimeError(
            "The network could not be run on a (1, n_features) input. "
            "Check feature_names / the scaler / the model's input size."
        ) from exc


class TorchMultiLabelWrapper(ClassifierMixin, _TorchWrapperBase):
    """
    sklearn-style multi-label classifier on top of a PyTorch network.

    The network has one output unit per target (each a 0/1 label), usually
    trained with BCEWithLogitsLoss. The interface matches sklearn's
    multi-output classifiers:
        predict(X)       -> (n, n_targets) labels
        predict_proba(X) -> list with one (n, 2) array per target
        classes_         -> list with one array of 2 labels per target

    Parameters
    ----------
    model : nn.Module
        Returns (n, n_targets).
    n_targets : int
        Number of target columns; must equal the number of output units.
    classes : sequence of 2 labels, optional
        Labels of every target, e.g. ["no", "yes"]. Default: [0, 1].
    feature_names, scaler, output_type, device, batch_size
        As in TorchClassifierWrapper.
    """

    def __init__(self, model, n_targets, feature_names=None, scaler=None,
                 output_type="auto", classes=None, device="cpu",
                 batch_size=4096):
        self.model = model
        self.n_targets = n_targets
        self.feature_names = feature_names
        self.scaler = scaler
        self.output_type = output_type
        self.classes = classes
        self.device = device
        self.batch_size = batch_size

        valid = ("auto", "logits", "probabilities", "log_probabilities")
        if output_type not in valid:
            raise ValueError(f"output_type must be one of {valid}.")

        self._setup(model, feature_names, scaler, device, batch_size)
        self.output_type_ = (_infer_output_type(model) if output_type == "auto"
                             else output_type)

        n_out = _dry_run_outputs(self)
        if n_out != n_targets:
            raise ValueError(
                f"The network has {n_out} output units but {n_targets} target "
                f"columns were given. A multi-label network needs one output "
                f"unit per target."
            )

        labels = np.arange(2) if classes is None else np.asarray(classes)
        if len(labels) != 2:
            raise ValueError(
                "A multi-label network needs exactly 2 class labels per target, "
                f"got {list(labels)}."
            )
        self.classes_ = [labels.copy() for _ in range(n_targets)]

    def _positive_proba(self, X):
        """(n, n_targets) probability of the positive label of every target."""
        raw = self._raw_output(X)
        if self.output_type_ == "logits":
            p1 = _sigmoid(raw)
        elif self.output_type_ == "log_probabilities":
            p1 = np.exp(raw)
        else:
            p1 = raw
        return np.clip(p1, 0.0, 1.0)

    def predict_proba(self, X):
        p1 = self._positive_proba(X)
        return [np.column_stack([1.0 - p1[:, j], p1[:, j]])
                for j in range(self.n_targets)]

    def predict(self, X):
        positive = self._positive_proba(X) >= 0.5
        labels = self.classes_[0]
        return np.where(positive, labels[1], labels[0])


class TorchRegressorWrapper(RegressorMixin, _TorchWrapperBase):
    """
    sklearn-style regressor on top of a PyTorch network.

    Parameters
    ----------
    model : nn.Module
        Returns (n, 1) or (n,) for a single target, (n, n_targets) for
        several targets.
    n_targets : int
        Number of target columns; must equal the number of output units.
    feature_names, scaler, device, batch_size
        As in TorchClassifierWrapper.
    """

    def __init__(self, model, feature_names=None, scaler=None,
                 device="cpu", batch_size=4096, n_targets=1):
        self.model = model
        self.feature_names = feature_names
        self.scaler = scaler
        self.device = device
        self.batch_size = batch_size
        self.n_targets = n_targets

        self._setup(model, feature_names, scaler, device, batch_size)

        # Dry run: fail early with a clear message if the input or output size is wrong
        n_out = _dry_run_outputs(self)
        if n_out != n_targets:
            raise ValueError(
                f"The network has {n_out} output units but {n_targets} target "
                f"column(s) were given. A regression network needs one output "
                f"unit per target."
            )

    def predict(self, X):
        out = self._raw_output(X)
        # One target -> 1-D array (n,), the shape sklearn and the XAI methods expect
        return out[:, 0] if out.shape[1] == 1 else out


def wrap_torch_model(model, task_type, feature_names=None, scaler=None,
                     output_type="auto", classes=None, device="cpu",
                     batch_size=4096, n_targets=1):
    """
    Wrap an nn.Module for the given task ("classification" / "regression").

    Classification with several targets is treated as multi-label (one
    output unit per 0/1 target).
    """
    if task_type == "classification":
        if n_targets > 1:
            return TorchMultiLabelWrapper(
                model, n_targets, feature_names=feature_names, scaler=scaler,
                output_type=output_type, classes=classes, device=device,
                batch_size=batch_size)
        return TorchClassifierWrapper(
            model, feature_names=feature_names, scaler=scaler,
            output_type=output_type, classes=classes, device=device,
            batch_size=batch_size)
    if task_type == "regression":
        return TorchRegressorWrapper(
            model, feature_names=feature_names, scaler=scaler,
            device=device, batch_size=batch_size, n_targets=n_targets)
    raise ValueError("task_type must be 'classification' or 'regression'.")


def unpack_torch_bundle(obj):
    """
    Accept the ways a user may have saved a network together with its scaler.

        net                                   -> (net, {})
        (net, scaler)                         -> (net, {"scaler": scaler})
        {"model": net, "scaler": s,           -> (net, {"scaler": s,
         "classes": [...], "output_type": ..}      "classes": [...], ...})

    Anything that is not a torch network (e.g. a sklearn model) is returned as
    (obj, None) so callers can leave it untouched.
    """
    if isinstance(obj, nn.Module):
        return obj, {}

    if (isinstance(obj, (tuple, list)) and len(obj) == 2
            and isinstance(obj[0], nn.Module)):
        return obj[0], {"scaler": obj[1]}

    if isinstance(obj, dict) and isinstance(obj.get("model"), nn.Module):
        options = {}
        scaler = obj.get("scaler", obj.get("preprocessing"))
        if scaler is not None:
            options["scaler"] = scaler
        for key in ("classes", "output_type"):
            if obj.get(key) is not None:
                options[key] = obj[key]
        return obj["model"], options

    return obj, None


def prepare_model(obj, task_type, feature_names, n_targets=1, **defaults):
    """
    Return a model ready for the XAI methods.

    * sklearn-style models are returned unchanged.
    * torch networks (alone or bundled with their scaler) are wrapped.
    * already-wrapped models are returned unchanged.

    defaults (scaler, output_type, classes, device) are used unless the
    bundle supplies its own value. Values that are None are ignored.
    """
    if is_wrapped(obj):
        return obj

    net, bundle_options = unpack_torch_bundle(obj)
    if bundle_options is None:
        return obj  # not a torch model, e.g. sklearn

    options = {k: v for k, v in defaults.items() if v is not None}
    options.update(bundle_options)  # what was saved with the model wins
    return wrap_torch_model(net, task_type, feature_names=feature_names,
                            n_targets=n_targets, **options)