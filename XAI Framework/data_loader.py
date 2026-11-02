from pathlib import Path
import joblib
import pandas as pd
import torch
from torch_wrapper import unpack_torch_bundle

def load_dataset(dataset_path, target_col):
    """Load a CSV dataset and separate features (X) from target(s) (y)."""

    dataset_path = Path(dataset_path)

    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}"
        )

    data = pd.read_csv(dataset_path)

    if data.empty:
        raise ValueError("The dataset is empty.")

    # Support one target column or a list of target columns
    targets = [target_col] if isinstance(target_col, str) else target_col

    if not targets:
        raise ValueError("At least one target column must be specified.")

    if len(targets) != len(set(targets)):
        raise ValueError("Target columns must not contain duplicates.")

    missing_targets = [col for col in targets if col not in data.columns]

    if missing_targets:
        raise ValueError(
            f"Target column(s) not found: {missing_targets}. "
            f"Available columns: {list(data.columns)}"
        )

    if data[targets].isna().any().any():
        raise ValueError(
            f"Target column(s) contain missing values: {targets}"
        )

    X = data.drop(columns=targets)
    y = data[targets[0]] if len(targets) == 1 else data[targets]

    if X.shape[1] == 0:
        raise ValueError(
            "The dataset must contain at least one feature column."
        )

    return data, X, y



def load_model(model_path):
    """Load a serialized model from disk."""

    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    try:
        if model_path.suffix.lower() in (".pt", ".pth"):
            # Whole network saved with torch.save(model, path)
            model = torch.load(model_path, weights_only=False, map_location="cpu")
        else:
            model = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model from '{model_path}'."
        ) from exc

    # A state_dict holds only the weights, not the network itself
    if isinstance(model, dict) and "model" not in model:
        raise TypeError(
            "The file contains a dictionary without a 'model' key (a state_dict?). "
            "Save the whole network, or a dict {'model': net, 'scaler': scaler}."
        )

    # A PyTorch network may be saved alone, as (net, scaler) or as a dict
    # with a 'model' key. It is wrapped later, in framework_adapter.py.
    net, torch_options = unpack_torch_bundle(model)
    if torch_options is not None:
        return model

    if not hasattr(model, "predict"):
        raise TypeError(
            "The loaded model must provide a predict() method or be a torch.nn.Module."
        )
    return model


def load_inputs(dataset_path, model_path, target_col):
    """Load the dataset, split features and target, and load the model."""

    data, X, y = load_dataset(dataset_path, target_col)
    model = load_model(model_path)

    return data, X, y, model