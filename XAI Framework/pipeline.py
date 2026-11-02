from multi_output import as_target_list, describe_problem, per_target


def extract_metadata(
    df,
    target_col,
    categorical_number_features=None,
    active_target=None
):
    """
    Extract feature and target metadata from the dataset.
    """

    # Isolate predictor feature names
    targets = as_target_list(target_col)
    features_list = [col for col in df.columns if col not in targets]

    if not features_list:
        raise ValueError("The dataset must contain at least one feature column.")

    # Categorical features: text / "category" columns are found automatically,
    # number columns that hold category codes are added by the user
    categorical_number_features = list(categorical_number_features or [])
    unknown = [col for col in categorical_number_features if col not in features_list]
    if unknown:
        raise ValueError(
            f"categorical_number_features {unknown} are not features of the dataset."
        )
    categorical_features = (
        df[features_list]
        .select_dtypes(include=["object", "category"])
        .columns.tolist()
    )
    categorical_features += [
        col for col in categorical_number_features if col not in categorical_features
    ]

    # Numerical features
    numerical_features = [
        col for col in features_list
        if col not in categorical_features
    ]

    # Default feature for 1D ALE / PDP
    default_target_feature = features_list[0]

    # Resolve the active target
    if isinstance(target_col, list):
        if active_target is None:
            resolved_active_target = target_col[0]
        else:
            if active_target not in target_col:
                raise ValueError(
                    f"'{active_target}' is not in the target columns: {target_col}"
                )
            resolved_active_target = active_target
    else:
        resolved_active_target = target_col

    return {
        "targets": targets,
        "features_list": features_list,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "default_target_feature": default_target_feature,
        "resolved_active_target": resolved_active_target,
    }

def build_unified_payload(
    model,
    df,
    target_col,
    task_type,
    scope="both",
    active_target=None,
    query_index=0,
    class_to_explain=None,
    categorical_number_features=None,
    immutable_features=None,
    scoring_metric=None,
    class_names=None,
    features_to_plot=None,
    random_state=42,
    feature_metadata=None
):
    """Build the unified payload used by the XAI methods."""

    # Extract metadata from the dataset
    metadata_info = extract_metadata(
        df=df,
        target_col=target_col,
        categorical_number_features=categorical_number_features,
        active_target=active_target
    )

    features_list = metadata_info["features_list"]
    resolved_active_target = metadata_info["resolved_active_target"]

    # Validate the query instance index
    if not 0 <= query_index < len(df):
        raise IndexError(
            f"query_index must be between 0 and {len(df) - 1}."
        )

    # class_names / class_to_explain may be one value, or a dict {target: value}
    class_names = per_target(class_names, resolved_active_target)
    # One value for every target: targets without that class use the default
    class_to_explain_shared = not isinstance(class_to_explain, dict)
    class_to_explain = per_target(class_to_explain, resolved_active_target)

    # Build the unified payload
    unified_payload = {
        "model": model,
        "full_dataset": df,
        "reference_dataset": df[features_list],
        "target": df[resolved_active_target],
        "query_instance": df[features_list].iloc[[query_index]],
        "scope": scope,
        "metadata": {
            "task_type": task_type,
            "problem_type": describe_problem(df, metadata_info["targets"], task_type),
            "target_col": target_col,
            "active_target": resolved_active_target,
            "targets": metadata_info["targets"],
            "categorical_features": metadata_info["categorical_features"],
            "immutable_features": immutable_features or [],
            "target_feature": metadata_info["default_target_feature"],
            "scoring_metric": scoring_metric,
            "class_names": class_names,
            "class_to_explain": class_to_explain,
            "class_to_explain_shared": class_to_explain_shared,
            "query_index": query_index,
            "features_to_plot": features_to_plot,
            "random_state": random_state,
            "numerical_features": metadata_info["numerical_features"],
            "feature_metadata": feature_metadata
        }
    }

    return unified_payload

