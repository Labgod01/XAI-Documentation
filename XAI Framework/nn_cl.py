import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_classification_nn(df, feature_names, target_col, random_state=42):
    """Train a PyTorch NN Model for classification."""

    if df is None:
        raise ValueError(
            "The original DataFrame must be provided to train_classification_nn()."
        )
    # TEST 1: Train a small PyTorch NN for Integrated Gradients
    # Use the existing demo classification dataset
    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32)

    # This first test requires numerical features and binary labels
    assert np.isfinite(X).all(), "Features contain NaN or infinite values."
    assert len(np.unique(y)) == 2, "This test requires binary classification."

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    class ClassificationNN(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.network(x)

    nn_model = ClassificationNN(X.shape[1])

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)

    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    nn_model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        logits = nn_model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()

    nn_model.eval()

    # Basic predictive sanity check
    with torch.no_grad():
        test_logits = nn_model(torch.tensor(X_test_scaled, dtype=torch.float32))
        predictions = (test_logits >= 0).float().numpy().ravel()

    accuracy = np.mean(predictions == y_test)

    print("NN training completed.")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Input features: {X.shape[1]}")
    print(f"Output shape: {tuple(test_logits.shape)}")

    return nn_model, scaler
