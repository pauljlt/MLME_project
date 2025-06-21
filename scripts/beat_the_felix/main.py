import os
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scripts.ANN.ann import NARXNet, prepare_data_multi, create_narx_dataset_multi, evaluate_model

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
input_cols = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']
output_cols = ['T_PM', 'c', 'd10', 'd50', 'd90', 'T_TM']
u_lag = 5
hidden_layers = [64]
dropout = 0.05600712050613521
activation = 'tanh'

ann_dir = "./visuals/ann_data"
cqr_dir = "./visuals/cqr_data"
btf_path = "./release/Beat-the-Felix/file_btf.txt"
output_dir = "./visuals/beat_the_felix"
os.makedirs(output_dir, exist_ok=True)

# Load scalers
scaler_u = joblib.load(os.path.join(ann_dir, "scaler_u.pkl"))
scaler_y = joblib.load(os.path.join(ann_dir, "scaler_y.pkl"))

# Load model
model = NARXNet(
    input_size=(len(input_cols) + len(output_cols)) * u_lag,
    output_size=len(output_cols),
    hidden_layers=hidden_layers,
    dropout=dropout,
    activation=activation
).to(device)
model.load_state_dict(torch.load(os.path.join(ann_dir, "narx_model.pth"), map_location=device))
model.eval()

# Load and prepare BTF data
btf_data = pd.read_csv(btf_path, sep='\t')
btf_data['trajectory_id'] = 0  # Dummy trajectory_id for consistent processing

u_btf, y_btf, _, _ = prepare_data_multi(btf_data, input_cols, output_cols, scaler_u, scaler_y)
x_btf, y_btf_target, _ = create_narx_dataset_multi(u_btf, y_btf, btf_data['trajectory_id'].values, u_lag, u_lag)

# Predict
true, pred = evaluate_model(model, x_btf, y_btf_target, scaler_y)

# Save predictions
np.save(os.path.join(output_dir, "X_btf.npy"), x_btf)
np.save(os.path.join(output_dir, "btf_true.npy"), true)
np.save(os.path.join(output_dir, "btf_pred.npy"), pred)

# Plot results
for i, col in enumerate(output_cols):
    plt.figure(figsize=(12, 4))
    plt.plot(true[:, i], 'o', label="True", alpha=0.2, markersize=2)
    plt.plot(pred[:, i], '-', label="Predicted", linewidth=1)
    plt.title(f"Beat-the-Felix Prediction – {col}")
    plt.xlabel("Timestep")
    plt.ylabel(col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"btf_prediction_{col}.svg"))
    plt.close()

# Compute metrics
metrics = {
    "target": [],
    "mse": [],
    "mae": []
}

for i, col in enumerate(output_cols):
    mse = mean_squared_error(true[:, i], pred[:, i])
    mae = mean_absolute_error(true[:, i], pred[:, i])
    metrics["target"].append(col)
    metrics["mse"].append(mse)
    metrics["mae"].append(mae)

metric_df = pd.DataFrame(metrics)
metric_df.to_csv(os.path.join(output_dir, "btf_performance.csv"), index=False)
print("\nBeat-the-Felix Evaluation Results:")
print(metric_df.to_string(index=False))
