import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    """
    Load data from a txt-file into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the txt-file.
    
    Returns:
    pd.DataFrame: The loaded data as a DataFrame.
    """
    
    try:
        data = pd.read_csv(file_path, sep='\t')
        data['trajectory_id'] = os.path.basename(file_path).split('.')[0]
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


def load_all_data(directory):
    """
    Load all txt-files from a directory into a single pandas DataFrame.
    
    Parameters:
    directory (str): The path to the directory containing txt-files.
    
    Returns:
    pd.DataFrame: A DataFrame containing all loaded data.
    """

    all_data = []
    for file_path in glob.glob(os.path.join(directory, "*.txt")):
        data = load_data(file_path)
        if data is not None:
            all_data.append(data)
    
    if all_data:
        #save the data to a csv file
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data.to_csv(os.path.join("C:/Users/paulj/Documents/Uni/12. Semester/MLME/MLME_project/visuals", "combined_data.csv"), index=False)
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no data was loaded
    

def analyze_data(data):
    """
    Analyze the loaded data.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the loaded data.
    
    Returns:
    None: Prints basic statistics of the data.
    """

    if not data.empty:
        print("Data Summary:")
        print(data.describe(percentiles=[0.1, 0.5, 0.9]))
        print("\nData Types:")
        print(data.dtypes)
        print("\nNumber of unique trajectories:", data['trajectory_id'].nunique())
        print("\nAre there any missing values?")
        print(data.isnull().any())
        if data.isnull().any().any(): # Check for missing values
            print("Missing values found in the dataset.")
            print("\nMissing values per column:")
            print(data.isnull().sum())
    else:
        print("No data to analyze.")


def plot_data(data):
    """
    Plot the data for visual analysis.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the loaded data.
    
    Returns:
    None: Displays plots of the data.
    """

    visuals_dir = "C:/Users/paulj/Documents/Uni/12. Semester/MLME/MLME_project/visuals"

    # histogram of the of the measurements
    data[['c', 'T_PM', 'd10', 'd50', 'd90', 'T_TM', 'mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']].hist(bins=50, figsize=(14,8))
    plt.suptitle("Histograms of the Measurements")
    plt.savefig(os.path.join(visuals_dir, "raw_data_histograms.png"))

    # boxplot of the measurements
    cols = ['c', 'T_PM', 'd10', 'd50', 'd90', 'T_TM', 'mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']
    plt.figure(figsize=(14,8))
    for i, col in enumerate(cols, 1):
        plt.subplot(4, 4, i)
        data.boxplot(column=col)
        # plt.title(col)
    plt.suptitle("Boxplots of the Measurements")
    plt.savefig(os.path.join(visuals_dir, "raw_data_boxplots.png"))

    # correlation matrix
    corr = data[['c', 'T_PM', 'd10', 'd50', 'd90', 'T_TM', 'mf_PM', 'mf_TM', 'Q_g', 'w_crystal', 'c_in', 'T_PM_in', 'T_TM_in']].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr, cmap='viridis')

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix of the Measurements", y=1.15)
    plt.colorbar(cax)
    
    for (i, j), val in np.ndenumerate(corr.values): # Annotate the correlation matrix
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
    
    plt.savefig(os.path.join(visuals_dir, "raw_data_correlation_matrix.png"))

    # scatter plot of the measurements
    scatter_pairs = [
        ('c', 'T_PM'),
        ('c', 'T_TM'),
        ('c', 'mf_TM'),
        ('c', 'c_in'),
        ('c', 'T_PM_in'),
        ('c', 'T_TM_in'),
        ('T_PM', 'T_TM'),
        ('T_PM', 'mf_TM'),
        ('d10', 'd50'),
        ('d10', 'd90'),
        ('d50', 'd90'),
        ('T_PM', 'mf_PM'),
        ('c_in', 'T_PM_in'),
        ('c_in', 'T_TM_in'),
        ('T_PM_in', 'T_TM_in'),
    ]

    plt.figure(figsize=(14, 8))

    for i, (x, y) in enumerate(scatter_pairs, 1):
        plt.subplot(3, 5, i)
        plt.scatter(data[x], data[y], alpha=0.25)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f'{y} vs {x}')
        plt.grid(True, which='major', axis='both', linestyle='--', alpha=0.6)

    plt.suptitle("Scatterplots of interesting feature pairs")
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, "raw_data_scatter_plots.png"))
    plt.show()



if __name__ == "__main__":
    # Example usage
    file_path = "./release/Data"
    data = load_all_data(file_path)
    if not data.empty:
        print("Data loaded successfully.")
        analyze_data(data)
        plot_data(data)
    else:
        print("No data found or loaded.")