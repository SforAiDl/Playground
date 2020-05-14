# Contributing Guidelines

## Getting started
- To start contributing, first fork the repository by clicking on **Fork**. This will create your own copy of the repository at github.com/{github_username}/Playground.
- Now, clone your fork using:
    
        $ git clone github.com/{github_username}/Playground.git

    and set remotes:

        $ git remote add origin github.com/{github_username}/Playground.git

        $ git remote add upstream github.com/SforAiDl/Playground.git
  
Setting up environment :
- Create an Anaconda environment for the project to ensure that you're working with the right versions of libraries that `Playground` is dependent on.

        $ conda create -n playground_env python=3.6 pip

        $ conda activate ts_env

OR

- Run 'sh setup_pip.sh' or 'sh setup_conda.sh' in terminal

## Installation
Will be added once the project is registered on PyPi

## Ways to contribute
### 1. Solving issues

- At a given point of time, all the open (unsolved) issues will be displayed on the **[GitHub Issues page](https://github.com/SforAiDl/Playground/issues)**. To start working on an issue, open the issue and request to take it up. One of the repository maintainers will then assign it to you. 

- Once you've taken up an issue, you can start working on it locally. Before starting update your local version using:

        $ git pull upstream master

    and create a new branch where you will be making your changes.

        $ git checkout -b {branch_name}

    Now you are ready to start making your changes.

- Once you are done, do the following:

        $ git add .

        $ git commit -m "{commit message}"

        $ git push origin {branch_name}

    This will update your `origin` branch with the changes. Now you are ready to submit a **Pull Request**.

- Once you have pushed to `origin`, click on **New Pull Request** to raise a PR to `upstream/master`. You can view pending PRs on the **[GitHub Pull Requests page](https://github.com/SforAiDl/Playground/pulls)**.

### 2. Opening issues
If you have new ideas or find issues while using the latest version of `Playground`, you can add new issues from the GitHub Issues page. Issues could be any of the following:

    - Fixing bugs
    - Adding features
    - New ideas

### 3. Pull Requests
You're encouraged to submit PRs for any of the following

    - Fixing issues
    - Adding tests to pre-existing code. This will help improve code coverage. [TODO]
    - Adding docs for existing code. To help users get started easily. [TODO]
