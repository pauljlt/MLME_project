# MLME_project

First, create a folder on your machine where you want to install the repository. Then open a shell and run:

```bash
git clone https://github.com/pauljlt/MLME_project.git
```

## Setup

Please make sure you're using **Python 3.11.9**. This version is not strictly required, but other versions might cause errors.

To use the project, set up a virtual environment by running:

- On Windows:
  ```bash
  python -m venv ..\MLME_venv
  ```
- On macOS/Linux:
  ```bash
  python3 -m venv ../MLME_venv
  ```

To activate the environment:

- On Windows:
  ```bash
  ..\MLME_venv\Scripts\activate.bat
  ```
- On macOS/Linux (or Bash):
  ```bash
  source ../MLME_venv/bin/activate
  ```

## Install Required Packages

Once the virtual environment is activated, upgrade `pip`:

- On Windows:
  ```bash
  python -m pip install --upgrade pip
  ```
- On macOS/Linux:
  ```bash
  python3 -m pip install --upgrade pip
  ```

Then install the required packages using:

```bash
pip install -r requirements.txt
```

## File Structure

```text
MLME_project/
├── release/
│   ├── Beat-the-Felix/
│   ├── Literature/
│   ├── CrysID_MLME25
│   └── Project report template-20250630.zip
├── scripts/
│   ├── __pycache__/
│   ├── ANN/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── ann.py
│   │   ├── clustering.py
│   │   ├── data_management.py
│   │   └── narx_bayesian_optimization.py
│   ├── beat_the_felix/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── CQR/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── cqr.py
│   │   └── cqr_bayesian_optimization.py
│   └── __init__.py
├── submission_files/
│   ├── ai_disclosure/
│   │   ├── MLME_project_AI_disclosure.pdf
│   │   └── MLME_project_AI_disclosure.zip
│   ├── report/
│   │   ├── MLME_project_report_G2.pdf
│   │   └── MLME_project_report_G2.zip
│   ├── screencast/
│   │   ├── MLME_project_final_presentation_G2.mp4
│   └── └── MLME_project_final_presentation_G2.pptx
├── visuals/
│   ├── ann_data/
│   ├── bayezian_optimization/
│   ├── beat_the_felix/
│   ├── clustering/
│   ├── cqr_data/
│   ├── data_management/
│   └── specials/
├── LICENSE.md
├── README.md
└── requirements.txt
MLME_venv/
```


## Git-Workflow

1. **Create a branch with your name.** This branch is where you’ll work without affecting the main branch. This avoids conflicts and keeps everyone’s work isolated.

   ```bash
   git checkout -b <YOUR_NAME>
   ```

   If your branch already exists, switch to it using:

   ```bash
   git checkout <YOUR_NAME>
   ```

2. **Work on the code and save files locally.**  
   If you're using VS Code, you should see your branch name in the bottom-left corner.

3. **When you’ve made successful changes**, go to the `MLME_project` folder and run the following commands:

   ```bash
   git add .
   git commit -m "<DESCRIPTION_OF_CHANGES>"
   git push -u origin <YOUR_NAME>   # Use just `git push` if the branch already exists
   git pull origin main
   ```

4. **Open GitHub**, navigate to your branch, and click **"Compare & pull request"** under the "Contribute" section.

5. **Fill in the PR details** (what you've done and why), and assign **pauljlt** as a reviewer.

## Git Help

For more details on using Git, check out this helpful video:  
👉 https://www.youtube.com/watch?v=8JJ101D3knE


# Running the project

First one needs to setup the venv and install the requirements like discribed above. To run any script please navigate to the 'MLME_project'-directory in your terminal and then execute the respective script.

**Make sure to execut using following command structure, since functions get imported from other scripts in different folders within the project.***

```bash
python -m path.to.script
```

1. **Analyze and visualize** the given data by executing **'data_management.py'**.
2. **Cluster** the data by executing **'clustering.py'**.
3. **Setup and train the ANN** by running **'ann.py'**.
4. If you want to **optimize your hyperparameters** by bayesian optimization, you can do so by running **'narx_bayesian_optimization.py'**. Be carefull, since this needs a lot of computational power. The standard hyperparameters (used if one changes nothing) are already optimized. See 'MLME_project/visuals/bayesian_optimization/optimization_log.txt' for logging.
5. Run **'cqr.py'** for **Conformalized Quantile Regression (CQR)**. The hyperparameters of the models for each target value and each quantile respectively can be optimized by running 'cqr_bayesian_optimization.py'.


## Beat-the-Felix

1. Store your file in **MLME_projet/release/Beat-the-Felix** and rename it as **'beat_the_felix.txt'**.
2. Just run **'python -m scripts.beat_the_felix.main'** out of the './MLME_project'.