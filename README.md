<br />
<p align="center">
  <h1 align="center">Assignment 3 Reinforcement Learning</h1>

  <p align="center">
  </p>
</p>

## About
This project contains skeleton code and a virtual environment to help you get started on the programming of reinforcement learning agents.
The project contains one module `cliffwalksim` in which you can add your code. You are free to add more modules but for the programming of the simulation that should not be necessary.
You are also free to change the project structure to use a convention you prefer. The one provided is the default project structure from `poetry`. 

## Getting started

### Prerequisites

- [Poetry](https://python-poetry.org/).

## Running
<!--
-->

You can also setup a virtual environment using Poetry. Poetry can  be installed using `pip`:
```
pip install poetry
```
Then initiate the virtual environment with the required dependencies (see `poetry.lock`, `pyproject.toml`):
```
poetry config virtualenvs.in-project true    # ensures virtual environment is in project
poetry install
```
The virtual environment can be accessed from the shell using:
```
poetry shell
```
IDEs like Pycharm will be able to detect the interpreter of this virtual environment (after `Add new interpreter`). The interpreter that Pycharm should use is `./.venv/bin/python3.10`.

If you want to add dependencies to the project then you can simply do
```
poetry add <package_name>
```
## Usage
You can add here some description on how to run the project (which file to run for example).

## Information on provided code

### Agent Creation
You can add RL agents by inheriting from the `TabularAgent` abstract base class and adjusting the `AgentFactory` class.

### metrickstracker

`MetricsTracker` is an object that you can use to record metrics of the reinforcement learning agents. Currently it provides functionality for keeping track of the sample mean and variance of the rewards over time for each agent. There is also a `plot` function you can use to plot the saved data.
You can use `MetricsTracker` as follows: Suppose an agent with name `agent_id` received `return` after the current time step. Then you can record the reward as follows:
```
tracker = MetricksTracker()
tracker.record_return(agent_id, return)
```
At the end you can plot the results using:
```
tracker.plot()
```
which will give a plot showing the average reward over time (with variance) for each agent.

You are free to modify `MetricksTracker` to suit your needs.