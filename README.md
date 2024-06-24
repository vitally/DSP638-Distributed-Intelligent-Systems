# Distributed Intelligent Systems - Homework Assignment

## Overview

This repository contains the solution for the homework assignment in the course DSP638 - Distributed Intelligent Systems. The solution includes the implementation of a stochastic agent operating in a 7x6 environment, performing various tasks such as value iteration, state transition sequence generation, and passive learning using different methods. The code is organized into multiple modules, each responsible for specific aspects of the solution.

## Repository Structure

- `config.py`: Configuration settings for the environment and learning parameters.
- `environment.py`: Functions related to the environment, such as updating rewards and checking terminal states.
- `learning_methods.py`: Implementation of passive learning methods (ADP, TD, DUE).
- `main.py`: Main program that executes the tasks outlined in the assignment.
- `policy_utils.py`: Utility functions for policy derivation and analysis.
- `sequence_generation.py`: Functions to generate state transition sequences.
- `value_iteration.py`: Implementation of the value iteration algorithm.
- `visualization.py`: Functions for plotting and visualizing the results.
- `validation.py`: Unit tests to validate the solution.

## Prerequisites

Ensure you have Python installed along with the necessary libraries. You can install the required libraries using:
```bash
pip install -r requirements.txt
```
## Instructions

### Running the Main Program

To execute the main program and observe the results of the value iteration, sequence generation, and passive learning methods, run:

```bash
python main.py
```

This will perform the following tasks:

1.	Value Iteration (Tasks 1-3): Perform value iteration to derive the initial policy and value table.
2.	Sequence Generation (Task 4): Generate sequences of state transitions based on the derived policy.
3.	Passive Learning (Task 5): Conduct ADP, TD, and DUE training using the generated sequences.
4.	Path Generation and Visualization: Generate and visualize paths based on the trained policies.
5.	Policy Analysis: Analyze and print the policy changes for different cost values.

### Running the Tests

To validate the solution, run the unit tests provided in validation.py:

```bash
python validation.py
```

The tests will check:

1.	If the value iteration produces a valid policy and value table.
2.	If the sequence generation works as expected.
3.	If the passive learning methods (ADP, TD, DUE) function correctly and update the policies.
4.	If paths are generated correctly based on the policies.
5.	If policies change appropriately with different cost values.