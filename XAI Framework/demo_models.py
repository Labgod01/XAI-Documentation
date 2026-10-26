from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _make_dataframe(X, feature_prefix="feature"):
    """Convert a NumPy feature matrix into a named DataFrame."""
    columns = [f"{feature_prefix}_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns)


def create_regression_demo(random_state=42):
    """Create and train a RandomForestRegressor demo."""

    X_array, y_array = make_regression(
        n_samples=600,
        n_features=8,
        n_informative=6,
        noise=15.0,
        random_state=random_state,
    )

    X = _make_dataframe(X_array)
    y = pd.Series(y_array, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "R2": r2_score(y_test, predictions),
    }

    return {
        "model": model,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "target_col": "target",
        "task_type": "regression",
        "model_type": "sklearn",
        "metrics": metrics,
    }


def create_classification_demo(random_state=42):
    """Create and train a RandomForestClassifier demo."""

    X_array, y_array = make_classification(
        n_samples=600,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_classes=2,
        random_state=random_state,
    )

    X = _make_dataframe(X_array)
    y = pd.Series(y_array, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
    }

    return {
        "model": model,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "target_col": "target",
        "task_type": "classification",
        "model_type": "sklearn",
        "metrics": metrics,
    }


class DemoNeuralNetwork(nn.Module):
    """Small differentiable network for the Integrated Gradients demo."""

    def __init__(self, input_dim, task_type="classification"):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.task_type = task_type

    def forward(self, x):
        # Return logits for binary classification or a scalar for regression.
        return self.network(x)


def create_ig_nn_demo(
    task_type="classification",
    random_state=42,
    epochs=60,
    batch_size=32,
):
    """Create and train a PyTorch model for direct Integrated Gradients."""

    if task_type not in ("classification", "regression"):
        raise ValueError(
            "task_type must be 'classification' or 'regression'."
        )

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    if task_type == "classification":
        X_array, y_array = make_classification(
            n_samples=600,
            n_features=8,
            n_informative=5,
            n_redundant=1,
            n_classes=2,
            random_state=random_state,
        )
        y_array = y_array.astype(np.float32).reshape(-1, 1)
    else:
        X_array, y_array = make_regression(
            n_samples=600,
            n_features=8,
            n_informative=6,
            noise=15.0,
            random_state=random_state,
        )
        y_array = y_array.astype(np.float32).reshape(-1, 1)

    X = _make_dataframe(X_array)
    y = pd.Series(y_array.ravel(), name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y if task_type == "classification" else None,
    )

    # Standardize using training data only.
    train_mean = X_train.mean()
    train_std = X_train.std().replace(0, 1)

    X_train_scaled = (X_train - train_mean) / train_std
    X_test_scaled = (X_test - train_mean) / train_std
    X_scaled = (X - train_mean) / train_std

    X_train_tensor = torch.tensor(
        X_train_scaled.to_numpy(dtype=np.float32)
    )
    X_test_tensor = torch.tensor(
        X_test_scaled.to_numpy(dtype=np.float32)
    )
    y_train_tensor = torch.tensor(
        y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
    )
    y_test_tensor = torch.tensor(
        y_test.to_numpy(dtype=np.float32).reshape(-1, 1)
    )

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    generator = torch.Generator().manual_seed(random_state)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    model = DemoNeuralNetwork(
        input_dim=X.shape[1],
        task_type=task_type,
    )

    if task_type == "classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epochs):
        model.train()

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()

    with torch.no_grad():
        predictions = model(X_test_tensor)

        if task_type == "classification":
            probabilities = torch.sigmoid(predictions).numpy().ravel()
            predicted_classes = (probabilities >= 0.5).astype(int)
            metrics = {
                "accuracy": accuracy_score(
                    y_test, predicted_classes
                )
            }
        else:
            predicted_values = predictions.numpy().ravel()
            metrics = {
                "MAE": mean_absolute_error(
                    y_test, predicted_values
                ),
                "R2": r2_score(y_test, predicted_values),
            }

    return {
        "model": model,
        "X": X,
        "y": y,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "target_col": "target",
        "task_type": task_type,
        "model_type": "pytorch",
        "metrics": metrics,
        "ig_use_surrogate": False,
        "preprocessing": {
            "mean": train_mean,
            "std": train_std,
        },
    }


def create_demo(demo_type, task_type=None, random_state=42):
    """Select and create one of the available demo scenarios."""

    if demo_type == "regression":
        if task_type not in (None, "regression"):
            raise ValueError(
                "The regression demo requires task_type='regression'."
            )
        return create_regression_demo(random_state)

    if demo_type == "classification":
        if task_type not in (None, "classification"):
            raise ValueError(
                "The classification demo requires task_type='classification'."
            )
        return create_classification_demo(random_state)

    if demo_type == "ig_nn":
        if task_type is None:
            task_type = "classification"

        return create_ig_nn_demo(
            task_type=task_type,
            random_state=random_state,
        )

    raise ValueError(
        "Unknown demo_type. Choose 'regression', "
        "'classification', or 'ig_nn'."
    )
