# MLME_project

# Setup
First please check if you are on python 3.11.9. This version is not necessary, but other versions may lead to errors.

For using the project one needs to set up a virtual environment by running "python -m venv <<VENV-NAME>>" on windows or "python3 -m venv <<VENV-NAME>>" on macOS. To activate this environment please run "<<VENV-NAME>>\Scipts\activate.bat" on windows or "source env/bin/activate" on macOS. For bash one can use the macOS commands.

# Install nescessery requirements in the virtual environment
Once the venv is activated, upgrade pip first by running "python -m pip install --upgrade pip" on windows or "python3 -m pip install --upgrade pip". Following that, run "pip install -r requirements.txt".

# Workflow
1. Create a branch with your name. This is the branch you will be working on without changing the main script. This is for safety reasons and to not hinder each other.
    "git checkout -b <<YOUR_NAME>>"
2. Work on the code and save your files on your local machine. If you use vs-code you should see your branchname all the time in the bottom left corner of your screen.
3. As soon as you have a running and succesfull change, run following commands:
    "git pull origin main"
    "git add -a"
    "git commit -m <<COMMIT_MESSAGE_ON_WHAT_YOU_HAVE_DONE>>"
    "git push -u origin <<YOUR_NAME>>"
    "git pull origin main"
4. Go to github.com and navigate to your branch.
5. Create a new pull request. Add as much details as possible on what you did in the message section. Add the repo owner (pauljlt) as reviewer of the code.

# How to use git
For detailed informtion on how to use git please visit "https://www.youtube.com/watch?v=8JJ101D3knE".