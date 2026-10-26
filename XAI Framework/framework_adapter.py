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
        "X_np": X_np,
        "x_1d": x_1d,
        "x_2d": x_2d,
        "x_dice_df": x_dice_df,
        "x_tensor": x_tensor,
        "x_tensor_baseline": x_tensor_baseline,
    }


