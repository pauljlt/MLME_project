import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
# Path to folder with .txt files
folder_path = "/Users/user/Downloads/release/Data"  # UPDATE this to your real path

# List to hold dataframes
data_list = []

# Loop through all .txt files
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        print(f"Loading: {file_name}")

        # Load each file as a DataFrame
        df = pd.read_csv(file_path, delimiter="\t")  # Tab-separated
        data_list.append(df)

# Combine all into one DataFrame
data = pd.concat(data_list, ignore_index=True)
#######################################heat map
inputs = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']
outputs = ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']
# Select input and output columns
df_subset = data[inputs + outputs].dropna()

corr_matrix = df_subset.corr()

# Extract correlation between inputs and outputs only
corr_inputs_outputs = corr_matrix.loc[outputs, inputs]

plt.figure(figsize=(10, 6))
sns.heatmap(corr_inputs_outputs, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation: Outputs vs Inputs")
plt.show()
#######################################visualize

# Loop through each input-output pair and plot
for out_col in outputs:
    plt.figure(figsize=(16, 10))
    for i, in_col in enumerate(inputs, 1):
        plt.subplot(3, 3, i)  # 3x3 grid for up to 9 plots
        sns.scatterplot(data=data, x=in_col, y=out_col, alpha=0.6)
        plt.title(f"{out_col} vs {in_col}")
    plt.tight_layout()
    plt.show()
    ##############################cluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
features = data[inputs + outputs]
# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)
kmeans = KMeans(n_clusters=3, random_state=1)
cluster_labels = kmeans.fit_predict(pca_features)

# Add cluster labels back to original data
features['Cluster'] = cluster_labels
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1],
                hue=cluster_labels, palette='Set2', s=60)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering (PCA-reduced Features)')
plt.legend(title='Cluster')
plt.show()
############################## dbscan cluster
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=2, min_samples=5)  # You can tune eps and min_samples
db_labels = dbscan.fit_predict(scaled_features)

# Convert scaled_features to DataFrame for visualization
scaled_features_df = pd.DataFrame(scaled_features)
scaled_features_df['Cluster'] = db_labels

# Visualize DBSCAN clustering using PCA components
pca_features_dbscan = pca.transform(scaled_features)  # Use the same PCA transformation
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_features_dbscan[:, 0], y=pca_features_dbscan[:, 1],
                hue=db_labels, palette='tab10', legend='full', s=70)
plt.title("DBSCAN Clustering (based on Inputs)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster')
plt.show()
############################## ANN
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_multivariate_narx_dataset(Y, U, n_lags=3):
    """
    Y: ndarray of shape (T, m) -- outputs
    U: ndarray of shape (T, d) -- inputs
    Returns: (X, Y_target)
    X shape: (T - n_lags - 1, (n_lags + 1) * (m + d))
    Y_target shape: (T - n_lags - 1, m)
    """
    T = len(Y)
    X, Y_target = [], []
    for i in range(n_lags, T - 1):
        y_lags = Y[i - n_lags:i + 1].flatten()
        u_lags = U[i - n_lags:i + 1].flatten()
        x_input = np.concatenate([y_lags, u_lags])
        X.append(x_input)
        Y_target.append(Y[i + 1])  # predict next output vector
    return np.array(X), np.array(Y_target)

# Load your data and select input/output columns
inputs = ['mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']
outputs = ['c', 'T_PM', 'd50', 'd90', 'd10', 'T_TM']

U_raw = data[inputs].values
Y_raw = data[outputs].values

# Normalize
scaler_U = StandardScaler()
scaler_Y = StandardScaler()
U_scaled = scaler_U.fit_transform(U_raw)
Y_scaled = scaler_Y.fit_transform(Y_raw)

# Create lagged dataset
n_lags = 3
X, Y_target = create_multivariate_narx_dataset(Y_scaled, U_scaled, n_lags=n_lags)

# Define MLP model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(Y_target.shape[1])  # same number of outputs as target vector
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, Y_target, epochs=100, batch_size=32, validation_split=0.2)

# Predict and inverse transform
Y_pred_scaled = model.predict(X)
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)

# Optional: compare prediction vs actual
import matplotlib.pyplot as plt
plt.plot(Y_raw[n_lags+1:, 0], label='Actual c')
plt.plot(Y_pred[:, 0], label='Predicted c')
plt.legend()
plt.title("Prediction of 'c'")
plt.show()

