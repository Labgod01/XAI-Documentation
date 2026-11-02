import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn
import nn_cl
import nn_reg
from torch_wrapper import is_wrapped
from multi_output import class_suffix
import config
from config import ig_model, ig_use_surrogate, ig_n_steps, ig_baseline
from config import ig_global_n_samples, ig_compute_global_importance, ig_global_top_k
from pathlib import Path
from captum.attr import IntegratedGradients
from sklearn.preprocessing import StandardScaler

def run_ig(unified_payload, prepared_inputs, output_dir=None, df=None):
    """Generate Integrated Gradients explanations for the active target."""

    model = prepared_inputs["model"]
    feature_names = prepared_inputs["feature_names"]
    reference_dataset = prepared_inputs["reference_dataset"]
    X_np = prepared_inputs["X_np"]
    x_1d = prepared_inputs["x_1d"]
    task_type = prepared_inputs["task_type"]
    metadata = prepared_inputs["metadata"]
    current_target = metadata["active_target"]
    scope = unified_payload["scope"] 
    # targets = metadata.get("targets", [current_target])
    targets = [current_target]
    if df is None:
        raise ValueError(
            "The original DataFrame must be provided to run_ig()."
        )
    
    output_dir = Path(output_dir or "outputs")
    raw_dir = output_dir / "raw_results"
    plots_dir = output_dir / "plots"

    raw_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if scope == "global":
        print(
            "Integrated Gradients: Generating global feature importance only."
        )
    elif scope == "both":
        print(
            "Integrated Gradients: Generating local explanations "
            "and global feature importance."
        )
    elif scope == "local":
        print(
            "Integrated Gradients: Generating local explanations."
        )
    else:
        raise ValueError(f"Unsupported explanation scope: {scope}")
    
    # Helper 1: Check whether a model is a PyTorch model

    def is_pytorch_model(model):
        return isinstance(model, torch.nn.Module)

    # Helper 2: Generic tabular PyTorch surrogate

    class TabularSurrogate(nn.Module):
        def __init__(self, input_dim, output_dim=1,
                    mean=None, std=None):
            super().__init__()

            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim)
            )
            if mean is None:
                mean = np.zeros((1, input_dim), dtype=np.float32)
            if std is None:
                std = np.ones((1, input_dim), dtype=np.float32)
            self.register_buffer(
                "feature_mean",
                torch.tensor(mean, dtype=torch.float32)
            )
            self.register_buffer(
                "feature_std",
                torch.tensor(std, dtype=torch.float32)
            )

        def forward(self, x):
            x_scaled = (x - self.feature_mean) / self.feature_std
            return self.network(x_scaled)

    # Store IG explanation objects and raw values separately
    integrated_gradients_explanations = {}
    integrated_gradients_raw_values = {}

    # Iterate over targets
    for current_target in targets:
        # Update unified payload
        metadata["active_target"] = current_target
        unified_payload["target"] = df[current_target]

        print(f"Target: {current_target}")
        print(f"Task type: {task_type}")

        
        # 1. Determine the model used for IG
        ig_scaler = None
        is_demo_nn = False

        if ig_model is not None:
            # Explicitly provided IG model has highest priority
            ig_estimator = ig_model
            using_surrogate = False
            print("IG model: User-provided model")

        elif is_wrapped(model):
            # PyTorch model wrapped for the other methods: use the network
            # itself, with its scaler built in, so attributions are in raw
            # feature units
            ig_estimator = model.as_torch_module()
            using_surrogate = False
            print("IG model: Main PyTorch model")

        elif is_pytorch_model(model):
            # Use the main model if it is already a PyTorch model
            ig_estimator = model
            using_surrogate = False
            print("IG model: Main PyTorch model")

        # run_mode is set by main.py ("demo" with --demo, otherwise "custom")
        elif getattr(config, "run_mode", "custom") == "demo" and config.ENABLE_IG_NN:
            # Train a demo neural network only in demo mode
            if task_type == "regression":
                ig_estimator, ig_scaler = (
                    nn_reg.train_regression_nn(
                        df,
                        feature_names,
                        current_target,
                        random_state=config.random_state
                    )
                )
            elif task_type == "classification":
                ig_estimator, ig_scaler = (
                    nn_cl.train_classification_nn(
                        df,
                        feature_names,
                        current_target,
                        random_state=config.random_state
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported task type: {task_type}"
                )

            is_demo_nn = True
            using_surrogate = False
            print("IG model: Demo neural network")

        else:
            if not ig_use_surrogate:
                raise TypeError(
                    "Integrated Gradients requires a differentiable "
                    "PyTorch model. Provide ig_model, use a PyTorch "
                    "main model, or enable ig_use_surrogate=True."
                )

            print("\n[IG NOTICE] A surrogate model will be used.")
            print(
                f"Original model type: {type(model).__name__}"
            )
            print(
                "The attributions explain the surrogate model, "
                "not the original model directly."
            )
            using_surrogate = True

            # 2. Prepare surrogate training data

            X_surrogate = np.asarray(reference_dataset, dtype=np.float32)

            # Normalize features to improve surrogate training
            surrogate_mean = X_surrogate.mean(axis=0, keepdims=True)
            surrogate_std = X_surrogate.std(axis=0, keepdims=True)
            surrogate_std[surrogate_std < 1e-8] = 1.0

            X_surrogate_tensor = torch.tensor(
                X_surrogate,
                dtype=torch.float32
            )

            # 3. Get predictions from original model
            X_surrogate_df = reference_dataset[feature_names]

            if task_type == "classification":
                # Use class predictions for the surrogate.
                y_original = model.predict(X_surrogate_df)

                classes = np.unique(y_original)
                n_classes = len(classes)

                print(f"Original model classes: {classes}")

            elif task_type == "regression":
                y_original = model.predict(X_surrogate_df)
                y_original = np.asarray(y_original).reshape(-1)
                n_classes = None

            else:
                raise ValueError(
                    f"Unsupported task type: {task_type}"
                )

            # 4. Create surrogate
            if task_type == "classification":
                if n_classes == 2:
                    # Binary classification
                    output_dim = 1
                else:
                    # Multiclass classification
                    output_dim = n_classes
            else:
                # Regression
                output_dim = 1
            ig_estimator = TabularSurrogate(
                input_dim=X_surrogate.shape[1],
                output_dim=output_dim,
                mean=surrogate_mean,
                std=surrogate_std
            )

            # 5. Prepare surrogate target
            if task_type == "regression":
                y_tensor = torch.tensor(
                    y_original,
                    dtype=torch.float32
                ).reshape(-1, 1)
                loss_function = nn.MSELoss()
            else:
                # Convert original class labels to indices
                class_to_index = {
                    cls: idx
                    for idx, cls in enumerate(classes)
                }
                y_indices = np.array([

                    class_to_index[y]

                    for y in y_original

                ])
                if n_classes == 2:
                    # Binary classification
                    y_tensor = torch.tensor(
                        y_indices,
                        dtype=torch.float32
                    ).reshape(-1, 1)
                    loss_function = (
                        nn.BCEWithLogitsLoss()
                    )
                else:
                    # Multiclass classification
                    y_tensor = torch.tensor(
                        y_indices,
                        dtype=torch.long
                    )
                    loss_function = (
                        nn.CrossEntropyLoss()
                    )
            # 6. Train surrogate
            optimizer = torch.optim.Adam(
                ig_estimator.parameters(),
                lr=0.001
            )
            ig_estimator.train()
            n_epochs = 300
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                surrogate_output = (
                    ig_estimator(
                        X_surrogate_tensor
                    )
                )
                if task_type == "classification":
                    if n_classes == 2:
                        loss = loss_function(
                            surrogate_output,
                            y_tensor
                        )
                    else:
                        loss = loss_function(
                            surrogate_output,
                            y_tensor
                        )
                else:
                    loss = loss_function(
                        surrogate_output,
                        y_tensor
                    )
                loss.backward()
                optimizer.step()
            ig_estimator.eval()
            print(
                "[OK] Surrogate model trained."
            )

        ig = IntegratedGradients(ig_estimator)

        # Classes to explain (chosen once for every method in framework_adapter),
        # as indices in class_list: the surrogate's outputs follow the classes it
        # was trained on. Regression: one output.
        class_list = None
        ig_classes = [(None, None)]
        if task_type == "classification":
            class_list = list(classes) if using_surrogate else list(model.classes_)
            ig_classes = []
            for label, _ in prepared_inputs["classes_to_explain"]:
                if label not in class_list:
                    print(f"IG: the surrogate never predicts class {label!r}; skipping it.")
                    continue
                ig_classes.append((label, class_list.index(label)))

        for explained_class, explained_index in ig_classes:
            suffix = class_suffix(explained_class, len(ig_classes))
            title_class = f" (class {explained_class})" if explained_class is not None else ""

            # Local explanation is required for local and both scopes
            attribution_df = None
            delta_value = None
            ig_target = None
            attribution_values = None
            baseline_tensor = None
            x_input = None
            x_tensor = None
            if scope in {"local", "both"}:
                # 7. Prepare observation to explain
                x_input = np.asarray(x_1d, dtype=np.float32).reshape(1, -1)

                if is_demo_nn:
                    x_scaled = ig_scaler.transform(x_input).astype(np.float32)
                    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
                else:
                    x_tensor = torch.tensor(x_input, dtype=torch.float32)

                # 8. Define baseline
                if ig_baseline == "zero":
                    baseline_tensor = torch.zeros_like(x_tensor)
                elif ig_baseline == "mean":
                    if is_demo_nn:
                        # Mean of the training data in scaled space
                        baseline_tensor = torch.zeros_like(x_tensor)
                    else:
                        mean_values = np.asarray(reference_dataset[feature_names].mean(axis=0),
                            dtype=np.float32).reshape(1, -1)
                        baseline_tensor = torch.tensor(mean_values, dtype=torch.float32)
                else:
                    raise ValueError("ig_baseline must be 'zero' or 'mean'.")


                # 9. Output of the explained class
                ig_target = None
                sign = 1.0
                if task_type == "classification":
                    with torch.no_grad():
                        model_output = ig_estimator(x_tensor)
                    if model_output.shape[-1] == 1:
                        # One output: the score of class_list[1]. The score of
                        # class_list[0] is 1 - p (or -logit), and IG is linear in the
                        # explained output, so its attributions are the negated ones.
                        sign = -1.0 if explained_index == 0 else 1.0
                    else:
                        ig_target = explained_index
                # 10. Integrated Gradients

                attributions, convergence_delta = (
                    ig.attribute(
                        x_tensor,
                        baselines=baseline_tensor,
                        target=ig_target,
                        n_steps=ig_n_steps,
                        return_convergence_delta=True
                    )
                )
                attributions = attributions * sign
                convergence_delta = convergence_delta * sign
                # 11. Convert attributions to DataFrame
                attribution_values = (
                    attributions
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                )
                attribution_df = pd.DataFrame({
                    "feature":
                        feature_names,
                    "attribution":
                        attribution_values,
                    "absolute_attribution":
                        np.abs(attribution_values)
                })
                attribution_df = (
                    attribution_df
                    .sort_values(
                        by="absolute_attribution",
                        ascending=False
                    )
                    .reset_index(drop=True)
                )
                # 12. Print results
                print("\nTop feature attributions:")
                print(
                    attribution_df
                    .head(10)
                    .to_string(index=False)
                )
                delta_value = (
                    convergence_delta
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()[0]
                )
                print(
                    f"\nConvergence delta: "
                    f"{delta_value:.6f}"
                )
                if task_type == "classification":
                    print(
                        f"Explained class: {class_list[explained_index]} "
                        f"(index {explained_index})"
                    )
                # 13. Visualization
                top_n = min(
                    10,
                    len(attribution_df)
                )
                plot_df = (
                    attribution_df
                    .head(top_n)
                    .sort_values(
                        "attribution"
                    )
                )
                plt.figure(
                    figsize=(9, 6)
                )
                plt.barh(
                    plot_df["feature"],
                    plot_df["attribution"]
                )
                plt.axvline(
                    0,
                    linestyle="--"
                )
                plt.xlabel(
                    "Integrated Gradients Attribution"
                )
                plt.ylabel(
                    "Feature"
                )
                plt.title(
                    f"Integrated Gradients - "
                    f"{current_target}{title_class}"
                )
                plt.tight_layout()
                plt.savefig(plots_dir / f"integrated_gradients_{current_target}{suffix}.png", dpi=300)
                #plt.show()

            # 13b. Global Integrated Gradients Importance
            global_importance_df=None

            if scope in {"global", "both"} and ig_compute_global_importance:
                X_global=np.asarray(reference_dataset[feature_names],dtype=np.float32)
                if ig_global_n_samples is not None and ig_global_n_samples<len(X_global):
                    rng=np.random.default_rng(unified_payload["metadata"]["random_state"])
                    idx=rng.choice(len(X_global),ig_global_n_samples,replace=False)
                    X_global=X_global[idx]

                # Apply the same preprocessing used during NN training
                if is_demo_nn:
                    X_global = ig_scaler.transform(X_global).astype(np.float32)
                X_global_tensor=torch.tensor(X_global,dtype=torch.float32)
                if ig_baseline == "zero":
                    global_baseline = torch.zeros_like(X_global_tensor)
                else:
                    if is_demo_nn:
                        # Zero in scaled space = training-set feature mean
                        global_baseline = torch.zeros_like(X_global_tensor)
                    else:
                        mean_values = np.asarray(reference_dataset[feature_names].mean(axis=0),dtype=np.float32).reshape(1, -1)
                        global_baseline = torch.tensor(mean_values, dtype=torch.float32).repeat(len(X_global_tensor), 1)

                global_target=None
                if task_type=="classification":
                    output=ig_estimator(X_global_tensor)
                    if output.shape[-1]>1:
                        global_target=explained_index

                global_attrs=ig.attribute(X_global_tensor,
                                        baselines=global_baseline,
                                        target=global_target,
                                        n_steps=ig_n_steps).detach().cpu().numpy()

                global_importance_df=pd.DataFrame({
                    "feature":feature_names,
                    "mean_absolute_attribution":np.mean(np.abs(global_attrs),axis=0)
                }).sort_values("mean_absolute_attribution",ascending=False).reset_index(drop=True)

                print("\nGlobal Average Feature Importance:")
                print(global_importance_df.head(ig_global_top_k).to_string(index=False))

                plot_df=global_importance_df.head(ig_global_top_k).sort_values("mean_absolute_attribution")
                plt.figure(figsize=(9,6))
                plt.barh(plot_df["feature"],plot_df["mean_absolute_attribution"])
                plt.xlabel("Mean Absolute Integrated Gradients Attribution")
                plt.ylabel("Feature")
                plt.title(f"Global Integrated Gradients Importance - {current_target}{title_class}")
                plt.tight_layout()
                plt.savefig(plots_dir / f"integrated_gradients_importance_{current_target}{suffix}.png", dpi=300)
                #plt.show()

            # Save separate JSON for global-only scope
            if scope == "global" and global_importance_df is not None:
                global_results = {
                    "target": str(current_target),
                    "task_type": task_type,
                    "global_importance": [
                        {
                            "feature": str(row["feature"]),
                            "mean_absolute_attribution": float(
                                row["mean_absolute_attribution"]
                            )
                        }
                        for _, row in global_importance_df.iterrows()
                    ]
                }

                global_json_path = (
                    raw_dir / f"integrated_gradients_global_{current_target}{suffix}.json"
                )
                with open(
                    global_json_path, "w", encoding="utf-8"
                ) as f:
                    json.dump(
                        global_results, f, indent=4, ensure_ascii=False
                    )
                print(f"Global IG results saved to: {global_json_path}")

            if scope in {"local", "both"}:
                # 13c. IG Raw Values

                # Prepare structured Integrated Gradients raw values
                with torch.no_grad():
                    ig_output = (
                        ig_estimator(x_tensor)
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                        .tolist()
                    )

                ig_raw = {
                    "method": "Integrated Gradients",
                    "target": str(current_target),
                    "task_type": task_type,
                    "input": {
                        str(name): float(value)
                        for name, value in zip(feature_names, x_input.flatten())
                    },
                    "baseline": {
                        str(name): float(value)
                        for name, value in zip(
                            feature_names, baseline_tensor.detach().cpu().numpy().flatten()
                        )
                    },
                    "model_output": [float(v) for v in ig_output],
                    "explained_class": (
                        class_list[explained_index].item()
                        if isinstance(class_list[explained_index], np.generic)
                        else class_list[explained_index]
                    ) if explained_index is not None else None,
                    "explained_class_index": (
                        int(explained_index) if explained_index is not None else None
                    ),
                    "n_steps": int(ig_n_steps),
                    "convergence_delta": float(delta_value),
                    "using_surrogate": bool(using_surrogate),
                    "feature_attributions": [
                        {
                            "feature": str(name),
                            "attribution": float(attr),
                            "absolute_attribution": float(abs(attr))
                        }
                        for name, attr in zip(feature_names, attribution_values)
                    ],
                    "global_importance": (
                        [
                            {
                                "feature": str(row["feature"]),
                                "mean_absolute_attribution": float(
                                    row["mean_absolute_attribution"]
                                )
                            }
                            for _, row in global_importance_df.iterrows()
                        ]
                        if global_importance_df is not None
                        else None
                    )
                }

                integrated_gradients_raw_values[f"{current_target}{suffix}"] = ig_raw

                print("\n--- Integrated Gradients Raw Values Summary ---")
                print("Target:", current_target)
                print("Task type:", task_type)
                print("Number of feature attributions:", len(attribution_values))
                print("Convergence delta:", float(delta_value))
                print("Using surrogate:", using_surrogate)
                print("Global importance available:", global_importance_df is not None)

                with open(
                    raw_dir / f"integrated_gradients_{current_target}{suffix}.json",
                    "w",
                    encoding="utf-8"
                ) as f:
                    json.dump(integrated_gradients_raw_values[f"{current_target}{suffix}"], f, indent=4, ensure_ascii=False)

            # 14. Store explanation
            integrated_gradients_explanations[f"{current_target}{suffix}"] = {
                "attributions": attribution_df,
                "convergence_delta": delta_value,
                "target_class": explained_index,
                "baseline": ig_baseline,
                "n_steps": ig_n_steps,
                "using_surrogate": using_surrogate,
                "ig_model": ig_estimator,
                "global_importance": global_importance_df
            }
        print(
            "\n[OK] Integrated Gradients "
            "explanation completed."
        )
    return integrated_gradients_explanations