import numpy as np
import pandas as pd
import config
import warnings
from sklearn.base import is_classifier
from torch_wrapper import prepare_model
from multi_output import check_output_count, explained_classes, make_target_view


def check_model_matches(model, feature_names, task_type):
    """
    Stop early, with a clear message, when the model was trained on other
    features or does not fit --task (otherwise a method fails later with an
    unclear error, or a regression run silently explains class labels).
    """
    trained_on = getattr(model, "feature_names_in_", None)
    if trained_on is not None:
        trained_on = [str(name) for name in trained_on]
        missing = [f for f in trained_on if f not in feature_names]
        extra = [f for f in feature_names if f not in trained_on]
        if missing or extra:
            raise ValueError(
                "The dataset features do not match the features the model was "
                f"trained on. Missing from the dataset: {missing}. "
                f"Not used by the model: {extra}."
            )
        if trained_on != list(feature_names):
            raise ValueError(
                "The dataset columns are in a different order than the features the "
                f"model was trained on. Expected order: {trained_on}."
            )
    elif getattr(model, "n_features_in_", None) not in (None, len(feature_names)):
        raise ValueError(
            f"The model was trained on {model.n_features_in_} features, but the "
            f"dataset has {len(feature_names)}: {feature_names}."
        )
    else:
            warnings.warn(
        "The model has no feature names (e.g. it was trained on a NumPy array), "
        "so the dataset columns could not be checked against the training features. "
        "Make sure they are the same features, in the same order."
    )

    if task_type == "classification" and not hasattr(model, "predict_proba"):
        raise ValueError(
            "--task classification, but the model has no predict_proba(). "
            "Either it is a regression model (use --task regression), or it is a "
            "classifier that does not give class probabilities (e.g. LinearSVC; "
            "use one that does, such as SVC(probability=True))."
        )
    if task_type == "regression" and (is_classifier(model) or hasattr(model, "classes_")):
        raise ValueError(
            "--task regression, but the model is a classifier. "
            "Use --task classification."
        )


def fill_unknown_targets(model, df, targets, task_type):
    """
    Rows without a true target value (e.g. a forecast for tomorrow, whose
    real value is not measured yet) get the model's prediction, so that every
    method can run on them: the methods explain the model's predictions, not
    the true values.

    Returns the dataset and a table of the same shape as df[targets] that is
    True where the value is a true one. Only PFI needs the true values (it
    measures how much the score drops), so it uses only those rows.
    """
    known = df[targets].notna()
    if known.all().all():
        return df, known

    feature_names = [col for col in df.columns if col not in targets]
    full_model = prepare_model(
        model,
        task_type=task_type,
        feature_names=feature_names,
        n_targets=len(targets),
        output_type=getattr(config, "torch_output_type", "auto"),
        classes=getattr(config, "torch_class_labels", None),
    )
    # The same checks as prepare_xai_inputs(), before predict() is called
    check_model_matches(full_model, feature_names, task_type)
    check_output_count(full_model, df[feature_names].iloc[[0]], targets)

    predictions = np.asarray(full_model.predict(df[feature_names]))
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    df = df.copy()
    for index, target in enumerate(targets):
        n_unknown = int((~known[target]).sum())
        if n_unknown == 0:
            continue
        predicted = pd.Series(predictions[:, index], index=df.index)
        values = df[target].astype(object).where(known[target], predicted)
        # Let pandas pick the column type again (a column with empty cells
        # is read as float: 1.0 instead of the class label 1)
        values = pd.Series(values.tolist(), index=df.index, name=target)
        if (task_type == "classification" and values.dtype.kind == "f"
                and (values % 1 == 0).all()):
            values = values.astype("int64")
        df[target] = values

        n_known = len(df) - n_unknown
        pfi_note = (f"PFI uses only the {n_known} rows with a true value."
                    if n_known else "PFI is skipped: it needs true values.")
        print(f"Target '{target}': {n_unknown} of {len(df)} rows have no true value. "
              f"They get the model's prediction, so every method can run; {pfi_note}")
    return df, known


def prepare_xai_inputs(unified_payload):
    """
    Prepare the inputs required by the XAI methods
    using the unified payload from the framework.
    """

    model = unified_payload["model"]
    reference_dataset = unified_payload["reference_dataset"]
    target = unified_payload["target"]
    query_instance = unified_payload["query_instance"]
    scope = unified_payload["scope"]
    metadata = unified_payload["metadata"]

    # Dynamic metadata extraction
    active_target_col = metadata["active_target"]
    feature_names = list(reference_dataset.columns)
    target_feature = metadata["target_feature"]
    task_type = metadata["task_type"]

    targets = metadata["targets"]

    # PyTorch networks get the sklearn interface (predict / predict_proba /
    # classes_) so every XAI method can use them. sklearn models are unchanged.
    model = prepare_model(
        model,
        task_type=task_type,
        feature_names=feature_names,
        n_targets=len(targets),
        output_type=getattr(config, "torch_output_type", "auto"),
        classes=getattr(config, "torch_class_labels", None),
    )

    # The model must fit the dataset and the task, and have one output per
    # target column
    check_model_matches(model, feature_names, task_type)
    check_output_count(model, query_instance, targets)

    # A model with several targets is explained one target at a time:
    # the methods get a view that returns only the active target's outputs
    model = make_target_view(
        model,
        targets=targets,
        active_target=active_target_col,
        task_type=task_type,
        df=unified_payload["full_dataset"],
    )

    # Classes to explain (classification only): decided here once, every
    # method loops over this list
    classes_to_explain = []
    if task_type == "classification":
        setting = metadata["class_to_explain"]
        if (setting not in (None, "predicted")
                and setting not in list(model.classes_)
                and metadata["class_to_explain_shared"] and len(targets) > 1):
            # One class given for every target: a target without that class
            # uses the default instead
            print(f"Target '{active_target_col}': no class {setting!r} "
                  f"(classes: {list(model.classes_)}), using the default classes.")
            metadata["class_to_explain_skipped"] = True
            setting = None
        classes_to_explain = explained_classes(model, query_instance, setting)

    # Convert reference feature set to a 2D NumPy array
    # Used by LIME, Anchors, ALE, PDP and ICE
    X_np = reference_dataset.to_numpy()

    # Flatten query instance to a 1D NumPy array
    # Used by LIME and Anchors
    x_1d = query_instance.to_numpy().flatten()

    # Reshape query instance to a 2D NumPy array
    # Used by SHAP
    x_2d = query_instance.to_numpy().reshape(1, -1)

    # Ensure query instance is a single-row DataFrame for DiCE
    x_dice_df = query_instance.iloc[[0]]

    # PyTorch Tensor conversion for Integrated Gradients
    try:
        import torch

        x_tensor = torch.tensor(
            query_instance.to_numpy(),
            dtype=torch.float32
        )
        x_tensor_baseline = torch.zeros_like(x_tensor)

    except ImportError:
        x_tensor = None
        x_tensor_baseline = None

    return {
        "model": model,
        "reference_dataset": reference_dataset,
        "target": target,
        "query_instance": query_instance,
        "scope": scope,
        "metadata": metadata,
        "active_target_col": active_target_col,
        "feature_names": feature_names,
        "target_feature": target_feature,
        "task_type": task_type,
        "classes_to_explain": classes_to_explain,
        "X_np": X_np,
        "x_1d": x_1d,
        "x_2d": x_2d,
        "x_dice_df": x_dice_df,
        "x_tensor": x_tensor,
        "x_tensor_baseline": x_tensor_baseline,
    }


