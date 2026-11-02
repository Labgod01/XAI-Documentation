import config
from torch_wrapper import prepare_model
from multi_output import check_output_count, explained_classes, make_target_view

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

    # The model must have one output per target column
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


