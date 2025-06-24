# scripts/ANN/narx_bayesian_optimization.py

import os
import sys
import torch
import optuna
import numpy as np
import pandas as pd

from scripts.ANN.ann import (
    train_narx_model,
    create_narx_dataset_multi,
    prepare_data_multi,
    evaluate_model,
    split_data
)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Fixed configuration
input_cols = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']
output_cols = ['T_PM', 'c', 'd10', 'd50', 'd90', 'T_TM']

# Load, filter and split data
df = pd.read_csv("./visuals/clustering/clustered_raw_data.csv")
df = df[df['Cluster'] != -1] # Filter out rows where Cluster is -1 (noise in DBSCAN clustering)
df = df[df['Cluster'] != 2] # Filter out rows where Cluster is 2 (outliers in DBSCAN clustering, d10, d50, d90 are all way too high)
df = df[df['Cluster'] != 3] # Filter out rows where Cluster is 3 (outliers in DBSCAN clustering, d10, d50, d90 are all way too high)
train_df, val_df, _ = split_data(df)

# Create log file
log_dir = "./visuals/bayezian_optimization"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "optimization_log.txt")
log_file = open(log_path, "w")

# Objective function for Optuna
def objective(trial):
    u_lag = y_lag = trial.suggest_int("lag", 3, 8)

    # Preprocess
    u_train, y_train, scaler_u, scaler_y = prepare_data_multi(train_df, input_cols, output_cols)
    u_val, y_val, _, _ = prepare_data_multi(val_df, input_cols, output_cols, scaler_u, scaler_y)
    x_train, y_train_target, _ = create_narx_dataset_multi(u_train, y_train, train_df['trajectory_id'].values, u_lag, y_lag)
    x_val, y_val_target, _ = create_narx_dataset_multi(u_val, y_val, val_df['trajectory_id'].values, u_lag, y_lag)

    hidden_layers = trial.suggest_categorical("hidden_layers", [[64], [128], [64, 128], [128, 64], [128, 256, 128], [64, 128, 64]])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    activation = trial.suggest_categorical("activation", ['relu', 'tanh'])
    epochs = trial.suggest_int("epochs", 50, 100, step=10)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    print("\nTraining with hyperparameters:")
    print(f"  lag={u_lag}, hidden_layers={hidden_layers}, dropout={dropout}, activation={activation}, epochs={epochs}, batch_size={batch_size}, lr={lr}")
    print(f"  lag={u_lag}, hidden_layers={hidden_layers}, dropout={dropout}, activation={activation}, epochs={epochs}, batch_size={batch_size}, lr={lr}", file=log_file)

    model = train_narx_model(
        x_train, y_train_target,
        input_size=x_train.shape[1],
        output_size=y_train_target.shape[1],
        hidden_layers=hidden_layers,
        dropout=dropout,
        activation=activation,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )

    y_val_true, y_val_pred = evaluate_model(model, x_val, y_val_target, scaler_y)
    mse = np.mean((y_val_true - y_val_pred) ** 2)

    # Log trial results
    log_msg = f"Trial {trial.number}: MSE={mse:.6f}, Params={trial.params}"
    print(log_msg)
    print(log_msg, file=log_file)
    log_file.flush()

    return mse

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    print(study.best_trial)
    print("Best trial:", file=log_file)
    print(study.best_trial, file=log_file)
    log_file.close()

    best_params = study.best_trial.params
    pd.DataFrame([best_params]).to_csv(os.path.join(log_dir, "best_hyperparameters.csv"), index=False)
