import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import optuna
import os
import sys

from torch.utils.data import DataLoader, TensorDataset

from ann import (
    NARXNet,
    create_narx_dataset_multi,
    prepare_data_multi,
    evaluate_model,
    split_data
)

sys.stdout = open("./visuals/narx_bayesian_optimization.log", "w")

def train_narx_model(X, Y, input_size, output_size, hidden_layers, dropout, activation, epochs, batch_size, lr):
    model = NARXNet(input_size, output_size, hidden_layers=hidden_layers, dropout=dropout, activation=activation)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def objective(trial):
    df = pd.read_csv("./visuals/clustered_raw_data.csv")
    train_df, val_df, _ = split_data(df)

    input_cols = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']
    output_cols = ['T_PM', 'c', 'd10', 'd50', 'd90', 'T_TM']

    # Hyperparameters
    lag = trial.suggest_int("lag", 3, 8)
    u_lag = y_lag = lag
    hidden_layers = trial.suggest_categorical("hidden_layers", [[64], [128], [64, 128], [128, 64], [128, 256, 128], [64, 128, 64]])
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    lr = trial.suggest_float("lr", 1e-4, 0.1, log=True)
    epochs = trial.suggest_int("epochs", 50, 150)

    # Preprocessing
    U_train, Y_train, scaler_u, scaler_y = prepare_data_multi(train_df, input_cols, output_cols)
    X_train, Y_train_target = create_narx_dataset_multi(U_train, Y_train, train_df['trajectory_id'].values, u_lag, y_lag)

    U_val, Y_val, _, _ = prepare_data_multi(val_df, input_cols, output_cols, scaler_u, scaler_y)
    X_val, Y_val_target = create_narx_dataset_multi(U_val, Y_val, val_df['trajectory_id'].values, u_lag, y_lag)

    model = train_narx_model(
        X_train, Y_train_target,
        input_size=X_train.shape[1],
        output_size=Y_train_target.shape[1],
        hidden_layers=hidden_layers,
        dropout=dropout,
        activation=activation,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )

    y_true, y_pred = evaluate_model(model, X_val, Y_val_target, scaler_y)
    rmse = compute_rmse(y_true, y_pred)
    print(f"Trial {trial.number}: RMSE = {rmse:.4f} | Params: {trial.params}")
    return rmse

if __name__ == "__main__":
    os.makedirs("./visuals", exist_ok=True)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("\nBest Trial:")
    print(f"Value (RMSE): {study.best_trial.value:.4f}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
