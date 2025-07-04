import optuna
import numpy as np
import torch
import sys
import os

from scripts.ANN.ann import train_narx_model, NARXNet
from MLME_project.scripts.CQR.cqr import pinball_loss

# Redirect stdout and stderr to a log file
log_path = "./visuals/cqr_data/cqr_optimization_log.txt"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
sys.stdout = open(log_path, "w")
sys.stderr = sys.stdout

# Load data
X_train = np.load("./visuals/ann_data/X_train.npy")
y_pred_train = np.load("./visuals/ann_data/y_pred_train.npy")
y_true_train = np.load("./visuals/ann_data/y_true_train.npy")

X_val = np.load("./visuals/ann_data/X_val.npy")
y_pred_val = np.load("./visuals/ann_data/y_pred_val.npy")
y_true_val = np.load("./visuals/ann_data/y_true_val.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial, i, quantile):
    """
    Objective function for Optuna to optimize hyperparameters for NARX model.

    Args:
        trial (optuna.Trial): The trial object containing the hyperparameters to be optimized.
        i (int): The index of the target variable in the output columns.
        quantile (float): The quantile level for the pinball loss function.
    Returns:
        float: The mean absolute error (MAE) of the model on the validation set.
    """
    
    hidden_layers = trial.suggest_categorical("hidden_layers", [[64], [128], [64, 64], [128, 64], [128, 128]])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 50, 100, step=10)

    eps_train = y_pred_train[:, i] - y_true_train[:, i]
    eps_val = y_pred_val[:, i] - y_true_val[:, i]

    model = train_narx_model(
        X_train, eps_train.reshape(-1, 1),
        input_size=X_train.shape[1], output_size=1,
        hidden_layers=hidden_layers,
        dropout=dropout,
        activation=activation,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        loss_fn=lambda pred, tgt: pinball_loss(pred, tgt, quantile)
    ).to(device)

    x_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_pred_q = model(x_val_tensor).cpu().detach().numpy().squeeze()

    val_error = eps_val - y_pred_q
    loss = np.mean(np.abs(val_error))
    print(f"Target index {i}, quantile {quantile}, trial completed with loss: {loss:.6f}")
    print("Trial hyperparameters:")
    print(f"  hidden_layers: {hidden_layers}")
    print(f"  dropout: {dropout}")
    print(f"  activation: {activation}")
    print(f"  lr: {lr}")
    print(f"  batch_size: {batch_size}")
    return loss

if __name__ == "__main__":
    output_cols = ['T_PM', 'c', 'd10', 'd50', 'd90', 'T_TM']
    quantiles = [0.1, 0.9]

    for i, target in enumerate(output_cols):
        for q in quantiles:
            print(f"\nStarting optimization for target '{target}' (index {i}), quantile {q}")
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective(trial, i, q), n_trials=10)

            print(f"Best trial for target '{target}', quantile {q}:")
            trial = study.best_trial
            for key, value in trial.params.items():
                print(f"  {key}: {value}")