import os
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from scripts.ANN.ann import NARXNet, prepare_data_multi, create_narx_dataset_multi

def evaluate_model(model, x_init, y_target, scaler_y, u_lag=None, output_size=None, mode="open"):
    """
    Evaluate the NARX model in open-loop or closed-loop mode.
    
    Args:
        model (NARXNet): The trained NARX model.
        x_init (np.ndarray): Initial input data for the model.
        y_target (np.ndarray): Target output data for evaluation.
        scaler_y (joblib.ScikitLearn): Scaler for the target output.
        u_lag (int, optional): Lag for the input data in closed-loop mode.
        output_size (int, optional): Size of the output in closed-loop mode.
        mode (str): Evaluation mode, either "open" or "closed".

    Returns:
        np.ndarray: True target values.
        np.ndarray: Predicted values from the model.
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Open-loop mode: use the initial input data directly
    if mode == "open":
        with torch.no_grad():
            x_tensor = torch.tensor(x_init, dtype=torch.float32)
            y_pred = model(x_tensor).detach().numpy()

    # Closed-loop mode: iteratively predict and update input data
    elif mode == "closed":
        if u_lag is None or output_size is None:
            raise ValueError("u_lag and output_size must be provided for closed-loop mode")

        x_current = x_init.copy()
        y_pred_list = []

        # Iterate through the time steps
        for t in range(len(x_current)):
            x_t = torch.tensor(x_current[t:t + 1], dtype=torch.float32) # Get the current input data
            y_pred_t = model(x_t).detach().numpy() # Predict the output for the current input
            y_pred_list.append(y_pred_t[0]) # Store the prediction

            # Update the input data for the next time step
            if t + 1 < len(x_current):
                u_part = x_current[t + 1][: -output_size * u_lag] # Get the input part of the next time step
                y_old = x_current[t + 1][-output_size * u_lag:] # Get the old output part of the next time step

                y_new = np.roll(y_old, -output_size) # Shift the old output part
                
                y_new[-output_size:] = y_pred_t[0] # Replace the last part with the new prediction
                x_current[t + 1] = np.concatenate([u_part, y_new]) # Concatenate the updated input part with the new output part

        y_pred = np.array(y_pred_list)

    else:
        raise ValueError("mode must be either 'open' or 'closed'")

    y_true = scaler_y.inverse_transform(y_target)
    y_pred = scaler_y.inverse_transform(y_pred)

    return y_true, y_pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_cols = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']
    output_cols = ['T_PM', 'c', 'd10', 'd50', 'd90', 'T_TM']
    u_lag = 5
    hidden_layers = [64]
    dropout = 5.3098337188695055e-05
    activation = 'relu'

    ann_dir = "./visuals/ann_data"
    btf_path = "./release/Beat-the-Felix/file_btf.txt"
    output_dir = "./visuals/beat_the_felix"
    os.makedirs(output_dir, exist_ok=True)

    scaler_u = joblib.load(os.path.join(ann_dir, "scaler_u.pkl"))
    scaler_y = joblib.load(os.path.join(ann_dir, "scaler_y.pkl"))

    model = NARXNet(
        input_size=(len(input_cols) + len(output_cols)) * u_lag,
        output_size=len(output_cols),
        hidden_layers=hidden_layers,
        dropout=dropout,
        activation=activation
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(ann_dir, "narx_model.pth"), map_location=device))
    model.eval()

    btf_data = pd.read_csv(btf_path, sep='\t')
    btf_data['trajectory_id'] = 0

    u_btf, y_btf, _, _ = prepare_data_multi(btf_data, input_cols, output_cols, scaler_u, scaler_y)
    x_btf, y_btf_target, _ = create_narx_dataset_multi(u_btf, y_btf, btf_data['trajectory_id'].values, u_lag, u_lag)

    true_open, pred_open = evaluate_model(model, x_btf, y_btf_target, scaler_y, mode="open")
    true_closed, pred_closed = evaluate_model(
        model, x_btf, y_btf_target, scaler_y,
        u_lag=u_lag, output_size=len(output_cols), mode="closed"
    )

    np.save(os.path.join(output_dir, "btf_true.npy"), true_closed)
    np.save(os.path.join(output_dir, "btf_pred_open.npy"), pred_open)
    np.save(os.path.join(output_dir, "btf_pred_closed.npy"), pred_closed)

    for i, col in enumerate(output_cols):
        plt.figure(figsize=(12, 4))
        plt.plot(true_open[:, i], 'o', label="True", alpha=0.2, markersize=2)
        plt.plot(pred_open[:, i], '-', label="Pred Open-Loop", linewidth=1)
        plt.plot(pred_closed[:, i], '--', label="Pred Closed-Loop", linewidth=1)
        plt.title(f"Beat-the-Felix Prediction: {col}")
        plt.xlabel("Timestep")
        plt.ylabel(col)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"btf_prediction_{col}.svg"))
        plt.close()

    metrics = {
        "target": [],
        "mse_open": [],
        "mae_open": [],
        "mse_closed": [],
        "mae_closed": []
    }

    for i, col in enumerate(output_cols):
        metrics["target"].append(col)
        metrics["mse_open"].append(mean_squared_error(true_open[:, i], pred_open[:, i]))
        metrics["mae_open"].append(mean_absolute_error(true_open[:, i], pred_open[:, i]))
        metrics["mse_closed"].append(mean_squared_error(true_closed[:, i], pred_closed[:, i]))
        metrics["mae_closed"].append(mean_absolute_error(true_closed[:, i], pred_closed[:, i]))

    metric_df = pd.DataFrame(metrics)
    metric_df.to_csv(os.path.join(output_dir, "btf_performance.csv"), index=False)
    print("\nBeat-the-Felix Evaluation Results:")
    print(metric_df.to_string(index=False))


if __name__ == "__main__":
    main()
