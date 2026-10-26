import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_regression_nn(df, feature_names, target_col, random_state=42):
    """Train a PyTorch NN Model for regression."""

    if df is None:
        raise ValueError(
            "The original DataFrame must be provided to train_regression_nn()."
        )
    
    # TEST 2: PyTorch Neural Network for Regression
    # Prepare regression data
    X = df[feature_names].to_numpy(dtype=np.float32)
    y = df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)

    assert np.isfinite(X).all(), "Features contain NaN or infinite values."
    assert np.isfinite(y).all(), "Target contains NaN or infinite values."

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Scale features using training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    class RegressionNN(nn.Module):
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

    nn_model = RegressionNN(X.shape[1])

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    nn_model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        predictions = nn_model(X_train_t)
        loss = loss_fn(predictions, y_train_t)
        loss.backward()
        optimizer.step()

    nn_model.eval()

    with torch.no_grad():
        test_predictions = nn_model(
            torch.tensor(X_test_scaled, dtype=torch.float32)
        ).numpy()

    rmse = np.sqrt(np.mean((test_predictions - y_test) ** 2))

    print("Regression NN training completed.")
    print(f"Test RMSE: {rmse:.4f}")
    print("Output shape:", test_predictions.shape)

    return nn_model, scaler
