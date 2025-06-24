import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from scripts.ANN.data_management import analyze_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class NARXNet(nn.Module):
    """
    Flexible feedforward neural network for NARX modeling.

    Args:
        input_size (int): Size of the input vector (from lags of inputs and outputs).
        output_size (int): Size of the output vector.
        hidden_layers (list): List of integers, each specifying the number of neurons in a hidden layer.
        dropout (float): Dropout rate applied after each hidden layer.
        activation (str): Activation function, either 'relu' or 'tanh'.
    """
    def __init__(self, input_size, output_size, hidden_layers=[64, 64], dropout=0.2, activation='relu'):
        super(NARXNet, self).__init__()

        layers = []
        current_size = input_size

        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_size = hidden_size

        layers.append(nn.Linear(current_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        return self.model(x)

def split_data(df):
    """
    Split the DataFrame into train, validation, and test sets based on trajectory_id and Cluster.

    Args:
        df (pd.DataFrame): Input DataFrame containing trajectory data with 'trajectory_id', 'Cluster' and 'timestamp' columns.

    Returns:
        train_df (pd.DataFrame): DataFrame for training.
        val_df (pd.DataFrame): DataFrame for validation.
        test_df (pd.DataFrame): DataFrame for testing.
    """

    # Filter
    df = df[df['Cluster'] != -1] # Filter out rows where Cluster is -1 (noise in DBSCAN clustering)
    df = df[df['Cluster'] != 2] # Filter out rows where Cluster is 2 (outliers in DBSCAN clustering, d10, d50, d90 are all way too high)
    df = df[df['Cluster'] != 3] # Filter out rows where Cluster is 3 (outliers in DBSCAN clustering, d10, d50, d90 are all way too high)
    

    # Split the data into train, validation, and test sets with respect to the trajectory_id and the Cluster
    trajectory_clusters = df.groupby('trajectory_id')['Cluster'].last() # Get the cluster for each trajectory_id

    train_ids, val_ids, test_ids = [], [], []

    # Iterate over unique clusters and split trajectory_ids accordingly, thus ensuring every cluster is represented in all sets
    for cluster in trajectory_clusters.unique():
        if len(trajectory_clusters[trajectory_clusters == cluster]) > 3: # If there are more than 3 trajectory_ids in the cluster, split them into train, validation, and test sets
            traj_ids = trajectory_clusters[trajectory_clusters == cluster].index.tolist() # Get all trajectory_ids for the current cluster

            traj_train, traj_temp = train_test_split(traj_ids, test_size=0.3, random_state=42) # Split into 70% train and 30% temp
            traj_val, traj_test = train_test_split(traj_temp, test_size=1/3, random_state=42) # Split temp into 20% validation and 10% test overall

            train_ids.extend(traj_train) # Collect all trajectory_ids for training
            val_ids.extend(traj_val) # Collect all trajectory_ids for validation
            test_ids.extend(traj_test) # Collect all trajectory_ids for testing
        else: # If there are 3 or fewer trajectory_ids in the cluster, assign them directly to train, validation, and test sets
            traj_ids = trajectory_clusters[trajectory_clusters == cluster].index.tolist()
            if len(traj_ids) == 1: # If there is only one trajectory_id, assign it to the train set
                train_ids.append(traj_ids[0])
            elif len(traj_ids) == 2: # If there are two trajectory_ids, assign one to the train set and one to the validation set
                train_ids.append(traj_ids[0])
                val_ids.append(traj_ids[1])
            elif len(traj_ids) == 3: # If there are three trajectory_ids, assign one to the train set, one to the validation set, and one to the test set
                train_ids.append(traj_ids[0])
                val_ids.append(traj_ids[1])
                test_ids.append(traj_ids[2])

    train_df = df[df['trajectory_id'].isin(train_ids)].copy() # Create DataFrame for training based on trajectory_ids
    val_df = df[df['trajectory_id'].isin(val_ids)].copy() # Create DataFrame for validation based on trajectory_ids
    test_df = df[df['trajectory_id'].isin(test_ids)].copy() # Create DataFrame for testing based on trajectory_ids

    # Print dataframes to csv
    output_dir = "./visuals/ann_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_df.to_csv(os.path.join(output_dir, "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_data.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_data.csv"), index=False)

    return train_df, val_df, test_df


def prepare_data_multi(df, input_cols, output_cols, scaler_input=None, scaler_output=None):
    """
    Scale input and output columns using MinMaxScaler for training or inference.

    Args:
        df (pd.DataFrame): Input DataFrame containing time series data.
        input_cols (list): Names of input columns.
        output_cols (list): Names of output columns.
        scaler_input (MinMaxScaler, optional): Existing scaler for inputs.
        scaler_output (MinMaxScaler, optional): Existing scaler for outputs.

    Returns:
        u_scaled (np.ndarray): Scaled input data.
        y_scaled (np.ndarray): Scaled output data.
        scaler_input (MinMaxScaler): Fitted or used input scaler.
        scaler_output (MinMaxScaler): Fitted or used output scaler.
    """
    u = df[input_cols].values
    y = df[output_cols].values

    if scaler_input is None:
        scaler_input = MinMaxScaler()
        u_scaled = scaler_input.fit_transform(u)
    else:
        u_scaled = scaler_input.transform(u)

    if scaler_output is None:
        scaler_output = MinMaxScaler()
        y_scaled = scaler_output.fit_transform(y)
    else:
        y_scaled = scaler_output.transform(y)

    return u_scaled, y_scaled, scaler_input, scaler_output


def create_narx_dataset_multi(u, y, trajectory_ids, u_lag=3, y_lag=3):
    """
    Create lagged input-output pairs for NARX modeling from multiple trajectories.

    Args:
        u (np.ndarray): Input data of shape (n_samples, n_input_features).
        y (np.ndarray): Output data of shape (n_samples, n_output_features).
        trajectory_ids (np.ndarray): Array of trajectory identifiers.
        u_lag (int): Number of past input steps to use.
        y_lag (int): Number of past output steps to use.

    Returns:
        x (np.ndarray): Lagged input matrix.
        y_out (np.ndarray): Corresponding target outputs.
    """
    x, y_out, meta_indices = [], [], []
    unique_ids = np.unique(trajectory_ids)

    for traj_id in unique_ids:
        indices = np.where(trajectory_ids == traj_id)[0]
        indices = np.sort(indices)
        u_traj = u[indices] # Here only the inputs of the current trajectory are selected
        y_traj = y[indices] # Here only the outputs of the current trajectory are selected

        # Loop through the current trajectory data to create lagged inputs and outputs
        for t in range(max(u_lag, y_lag), len(y_traj)): # Like this we ensure that we have enough data for the lags and don't start at the beginning, thus only using data from the current trajectory
            x_t = []
            for i in range(1, y_lag + 1):
                x_t.extend(y_traj[t - i]) # Append past outputs based on y_lag
            for i in range(1, u_lag + 1):
                x_t.extend(u_traj[t - i]) # Append past inputs based on u_lag
            x.append(x_t)
            y_out.append(y_traj[t])
            meta_indices.append(indices[t])  # Store the index for metadata

    return np.array(x), np.array(y_out), np.array(meta_indices)


def train_narx_model(X, Y, input_size, output_size, hidden_layers, dropout, activation,
                     epochs=100, batch_size=32, lr=0.01, loss_fn=None):
    """
    Train a NARX neural network with the given architecture and parameters.

    Args:
        X (np.ndarray): Training inputs.
        Y (np.ndarray): Training targets.
        input_size (int): Input vector size.
        output_size (int): Output vector size.
        hidden_layers (list): Sizes of hidden layers.
        dropout (float): Dropout rate.
        activation (str): Activation function ('relu' or 'tanh').
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.
        loss_fn (callable, optional): Custom loss function (default: MSELoss).

    Returns:
        model (NARXNet): Trained PyTorch model.
    """

    model = NARXNet(input_size, output_size, hidden_layers=hidden_layers,
                    dropout=dropout, activation=activation).to(device)
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.10f}")

    return model


def evaluate_model(model, X, Y_true_scaled, scaler_output):
    """
    Evaluate the NARX model on scaled validation/test data and inverse-transform output.

    Args:
        model (NARXNet): Trained model.
        X (np.ndarray): Input features.
        Y_true_scaled (np.ndarray): Scaled true outputs.
        scaler_output (MinMaxScaler): Output scaler for inverse transformation.

    Returns:
        Y_true (np.ndarray): Original scale true outputs.
        Y_pred (np.ndarray): Original scale predicted outputs.
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        Y_pred_scaled = model(X_tensor).numpy()
    Y_pred = scaler_output.inverse_transform(Y_pred_scaled)
    Y_true = scaler_output.inverse_transform(Y_true_scaled)

    return Y_true, Y_pred


def plot_predictions(y_true, y_pred, feature_names, title="NARX Model Prediction"):
    """
    Plot predicted vs true values for each output feature.

    Args:
        y_true (np.ndarray): Ground truth outputs.
        y_pred (np.ndarray): Model predictions.
        feature_names (list): List of feature names.
        title (str): Plot title.
    """

    output_dir = "./visuals/ann_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # All in one plot
    plt.figure(figsize=(16, 9))
    for i, name in enumerate(feature_names):
        plt.subplot(len(feature_names), 1, i + 1)
        plt.plot(y_true[:, i], label='True', marker='o', markersize=2, linestyle='None', alpha=0.5)
        plt.plot(y_pred[:, i], label='Predicted')
        plt.ylabel(name)
        plt.xlabel("Timestep")
        plt.legend()
        plt.grid(True)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title}.svg"))
    plt.close()

    # Sigle plots for each feature
    for i, name in enumerate(feature_names):
        plt.figure(figsize=(16, 9))
        plt.plot(y_true[:, i], label='True', marker='o', markersize=2, linestyle='None', alpha=0.5)
        plt.plot(y_pred[:, i], label='Predicted')
        plt.ylabel(name)
        plt.xlabel("Timestep")
        plt.legend()
        plt.grid(True)
        plt.title(f"{title} – {name}")
        filename = f"{title}_{name}.svg".replace("/", "_").replace(" ", "_")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

def safe_residuals(y_true, y_pred, output_cols, meta=None, path="residuals.csv"):
    """
    Save residuals of predictions to a CSV file.

    Args:
        y_true (np.ndarray): True output values.
        y_pred (np.ndarray): Predicted output values.
        output_cols (list): List of output feature names.
        meta (pd.DataFrame, optional): Additional metadata to include in the CSV.
        path (str): Path to save the residuals CSV file.

    Returns:
        None: Saves the residuals to a CSV file.
    """
    residuals = y_pred - y_true
    df = pd.DataFrame(residuals, columns=[f"ε_{col}" for col in output_cols])
    if meta is not None:
        df = pd.concat([df.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)
    df.to_csv("./visuals/ann_data/" + path, index=False)


def main():
    # Load the clustered raw data
    df = pd.read_csv("./visuals/clustering/clustered_raw_data.csv")

    # Output directory for visuals
    output_dir = "./visuals/ann_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Split the data into train, validation, and test sets
    train_df, val_df, test_df = split_data(df)

    # Analyze the splitted data sets
    print("Analyzing splitted data sets...")
    print(f"Train set:")
    analyze_data(train_df)
    print(f"Validation set:")
    analyze_data(val_df)
    print(f"Test set:")
    analyze_data(test_df)

    # Define input and output columns as well as the lag
    input_cols = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in'] # u
    output_cols = ['T_PM', 'c', 'd10', 'd50', 'd90', 'T_TM'] # y
    lag = 5  # Number of lags for inputs and outputs
    u_lag, y_lag = lag, lag

    # Prepare the data for training and validation
    U_train, Y_train, scaler_u, scaler_y = prepare_data_multi(train_df, input_cols, output_cols)
    X_train, Y_train_target, train_indices = create_narx_dataset_multi(U_train, Y_train, train_df['trajectory_id'].values, u_lag, y_lag)

    U_val, Y_val, _, _ = prepare_data_multi(val_df, input_cols, output_cols, scaler_u, scaler_y)
    X_val, Y_val_target, val_indices = create_narx_dataset_multi(U_val, Y_val, val_df['trajectory_id'].values, u_lag, y_lag)

    # Train the NARX model
    model = train_narx_model(
        X_train, Y_train_target,
        input_size=X_train.shape[1],
        output_size=Y_train_target.shape[1],
        hidden_layers=[64],
        dropout=5.3098337188695055e-05,
        activation='relu',
        epochs=60,
        batch_size=16,
        lr=7.023694002448674e-05,
        loss_fn=None  # Use default MSELoss
    )

    # Validate the model on the validation set
    y_true, y_pred = evaluate_model(model, X_val, Y_val_target, scaler_y)
    plot_predictions(y_true, y_pred, output_cols, title="NARX Model Validation Predictions")

    # Test the model on the test set
    U_test, Y_test, _, _ = prepare_data_multi(test_df, input_cols, output_cols, scaler_u, scaler_y)
    X_test, Y_test_target, test_indices = create_narx_dataset_multi(U_test, Y_test, test_df['trajectory_id'].values, u_lag, y_lag)

    y_test_true, y_test_pred = evaluate_model(model, X_test, Y_test_target, scaler_y)
    plot_predictions(y_test_true, y_test_pred, output_cols, title="NARX Model Test Predictions")

    # Eavaluate the model on the train and val set for CQR analysis
    y_train_true, y_train_pred = evaluate_model(model, X_train, Y_train_target, scaler_y)
    y_val_true, y_val_pred = evaluate_model(model, X_val, Y_val_target, scaler_y)

    # Save residuals to CSV for CQR analysis
    safe_residuals(y_train_true, y_train_pred, output_cols, meta=train_df.iloc[train_indices][['trajectory_id', 'timestamp','Cluster']], path="residuals_train.csv")
    safe_residuals(y_val_true, y_val_pred, output_cols, meta=val_df.iloc[val_indices][['trajectory_id', 'timestamp','Cluster']], path="residuals_val.csv")
    safe_residuals(y_test_true, y_test_pred, output_cols, meta=test_df.iloc[test_indices][['trajectory_id', 'timestamp','Cluster']], path="residuals_test.csv")

    # Save the model
    torch.save(model.state_dict(), os.path.join(output_dir, "narx_model.pth"))
    
    joblib.dump(scaler_u, os.path.join(output_dir, "scaler_u.pkl"))
    joblib.dump(scaler_y, os.path.join(output_dir, "scaler_y.pkl"))

    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)

    np.save(os.path.join(output_dir, "y_true_train.npy"), y_train_true)
    np.save(os.path.join(output_dir, "y_pred_train.npy"), y_train_pred)

    np.save(os.path.join(output_dir, "y_true_val.npy"), y_val_true)
    np.save(os.path.join(output_dir, "y_pred_val.npy"), y_val_pred)

    np.save(os.path.join(output_dir, "y_true_test.npy"), y_test_true)
    np.save(os.path.join(output_dir, "y_pred_test.npy"), y_test_pred)


if __name__ == "__main__":
    main()