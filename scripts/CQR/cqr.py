import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from scripts.ANN.ann import train_narx_model, NARXNet

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
input_cols = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']
output_cols = ['T_PM', 'c', 'd10', 'd50', 'd90', 'T_TM']

alpha = 0.2  # For 80% prediction interval
alpha_lo = alpha / 2
alpha_hi = 1 - alpha / 2
quantiles = [alpha_lo, alpha_hi]

batch_size = 32
epochs = 100
lr = 1e-4
hidden_layers = [128, 64]
dropout = 0.1
activation = 'tanh'

visuals_dir = "./visuals/ann_data"
output_dir = "./visuals/cqr_data"
btf_dir = "./visuals/beat_the_felix"
os.makedirs(output_dir, exist_ok=True)

# Load data
X_train = np.load(os.path.join(visuals_dir, "X_train.npy"))
X_val = np.load(os.path.join(visuals_dir, "X_val.npy"))
X_test = np.load(os.path.join(visuals_dir, "X_test.npy")) # X_test = np.load(os.path.join(btf_dir, "X_btf.npy")) # BTF data

# Prediction & true values
y_pred_train = np.load(os.path.join(visuals_dir, "y_pred_train.npy"))
y_true_train = np.load(os.path.join(visuals_dir, "y_true_train.npy"))

y_pred_val = np.load(os.path.join(visuals_dir, "y_pred_val.npy"))
y_true_val = np.load(os.path.join(visuals_dir, "y_true_val.npy"))

y_pred_test = np.load(os.path.join(visuals_dir, "y_pred_test.npy")) # y_pred_test = np.load(os.path.join(btf_dir, "btf_pred.npy")) # BTF data
y_true_test = np.load(os.path.join(visuals_dir, "y_true_test.npy")) # y_true_test = np.load(os.path.join(btf_dir, "btf_true.npy"))  # BTF data

# Pinball loss
def pinball_loss(pred, target, alpha):
    error = target - pred
    return torch.mean(torch.where(error >= 0, alpha * error, (alpha - 1) * error))

def evaluate_cqr_on_errors(train=True):
    results = []
    conformity_vals = {}

    for i, target in enumerate(output_cols):
        print(f"\nProcessing {target}…")

        eps_train = y_pred_train[:, i] - y_true_train[:, i]
        eps_val = y_pred_val[:, i] - y_true_val[:, i]
        eps_test = y_pred_test[:, i] - y_true_test[:, i]

        models = {}
        for q in quantiles:
            model_path = os.path.join(output_dir, f"qr_model_{target}_{int(100*q)}.pth")
            if not train:
                print(f"Loading existing model for q={q:.2f} from {model_path}")
                model = NARXNet(
                    input_size=X_train.shape[1],
                    output_size=1,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                    activation=activation
                ).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                models[q] = model
            else:
                print(f"Training q={q}")
                model = train_narx_model(
                    X_train, eps_train.reshape(-1, 1),
                    input_size=X_train.shape[1], output_size=1,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                    activation=activation,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    loss_fn=lambda pred, tgt: pinball_loss(pred, tgt, q)
                ).to(device)
                models[q] = model
                torch.save(model.state_dict(), model_path)

        x_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        eps_q_val = {q: models[q](x_val_tensor).cpu().detach().numpy().squeeze() for q in quantiles}
        E = np.maximum(eps_q_val[alpha_lo] - eps_val, eps_val - eps_q_val[alpha_hi])
        Q = max(np.quantile(E, 1 - alpha), 0.0)
        conformity_vals[target] = Q
        print(f"Conformity Q={Q:.5f}")

        # Evaluate on test set
        x_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        eps_q_test = {q: models[q](x_test_tensor).cpu().detach().numpy().squeeze() for q in quantiles}
        
        # Calculate the spread of the quantile predictions
        spread = eps_q_test[0.9] - eps_q_test[0.1]
        print(f"Mean spread for {target}: {np.mean(spread):.4f}")

        # Print ranges of quantile predictions
        print(f"q0.1 range: {eps_q_test[alpha_lo].min():.3f} to {eps_q_test[alpha_lo].max():.3f}")
        print(f"q0.9 range: {eps_q_test[alpha_hi].min():.3f} to {eps_q_test[alpha_hi].max():.3f}")

        # Print ranges of true values
        print("c min/max:", y_true_train[:, 1].min(), y_true_train[:, 1].max())
        print("T_TM min/max:", y_true_train[:, 5].min(), y_true_train[:, 5].max())



        # OPTIONAL: Visualisierung der Quantilmodelle
        plt.figure(figsize=(8, 5))
        # plt.scatter(np.arange(len(eps_q_test[alpha_lo])), eps_q_test[alpha_lo]-Q, label=f"q{alpha_lo:.2f}", alpha=0.5)
        # plt.scatter(np.arange(len(eps_q_test[alpha_hi])), eps_q_test[alpha_hi]+Q, label=f"q{alpha_hi:.2f}", alpha=0.5)
        plt.plot(eps_q_test[alpha_lo]-Q, label=f"q{alpha_lo:.2f} - Q", color='gray', linewidth=1.0, alpha=0.5)
        plt.plot(eps_q_test[alpha_hi]+Q, label=f"q{alpha_hi:.2f} + Q", color='gray', linewidth=1.0, alpha=0.5)
        plt.fill_between(np.arange(len(eps_q_test[alpha_lo]-Q)), eps_q_test[alpha_lo]-Q, eps_q_test[alpha_hi]+Q, color='gray', alpha=0.3, label='Prediction Interval')

        plt.scatter(np.arange(len(eps_test)), eps_test, label="true error", alpha=0.5, color="black", s=10)
        plt.title(f"Quantilvorhersage für Fehler: {target}")
        plt.xlabel("Testdatenpunkt")
        plt.ylabel("Vorhergesagter Fehler")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"quantile_predictions_{target}.svg"))
        plt.close()

        lower = y_pred_test[:, i] + eps_q_test[alpha_lo] - Q
        upper = y_pred_test[:, i] + eps_q_test[alpha_hi] + Q
        cover = ((y_true_test[:, i] >= lower) & (y_true_test[:, i] <= upper)).mean()
        width = (upper - lower).mean()

        print(f"Coverage: {cover:.3f}, Width: {width:.3f}")
        results.append({
            "target": target,
            "coverage": round(cover, 3),
            "interval_width": round(width, 3)
        })

        # Plot
        plt.figure(figsize=(12, 4))
        plt.plot(y_true_test[:, i], 'o', label="True", alpha=0.2, markersize=1)
        plt.plot(y_pred_test[:, i], '-', label="Pred", linewidth=0.5)
        plt.fill_between(np.arange(len(lower)), lower, upper, color='gray', alpha=0.5,
                         label=f"{int((1-alpha)*100)}% PI")
        plt.title(f"CQR (error-based): {target}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"cqr_interval_{target}.svg"))
        plt.close()

    pd.DataFrame(results).to_csv(os.path.join(output_dir, "cqr_results.csv"), index=False)
    np.save(os.path.join(output_dir, "conformity_values.npy"), conformity_vals)
    print("\nCQR on errors done:\n", pd.DataFrame(results))

if __name__ == "__main__":
    evaluate_cqr_on_errors(train=False)  # Set to True to train models, False to evaluate existing ones
