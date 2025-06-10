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
  python -m venv venv
  ```
- On macOS/Linux:
  ```bash
  python3 -m venv venv
  ```

To activate the environment:

- On Windows:
  ```bash
  venv\Scripts\activate.bat
  ```
- On macOS/Linux (or Bash):
  ```bash
  source venv/bin/activate
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

## Workflow

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
