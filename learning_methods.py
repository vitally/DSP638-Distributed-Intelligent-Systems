import numpy as np
import logging
from config import grid_size, discount_factor, theta, alpha
from environment import is_terminal


# Task 5: Passive Learning Training

# Implementing ADP, TD, and DUE methods

# ADP (Adaptive Dynamic Programming): This method uses a model of the environment to update the values of the cells.
def adp_training(sequences, rewards):
    value_table = np.zeros(grid_size)
    model = {}

    # Initialize model with state transitions from sequences    
    for sequence in sequences:
        for (state, action, next_state) in sequence:
            if state not in model:
                model[state] = {}
            if action not in model[state]:
                model[state][action] = []
            model[state][action].append(next_state)
    
    iteration = 0
    while True:
        delta = 0
        iteration += 1
        logging.info(f"ADP Iteration: {iteration}")
        for state in model:
            if is_terminal(state):
                continue
            v = value_table[state]
            max_value = float('-inf')
            for action in model[state]:
                new_value = 0
                for next_state in model[state][action]:
                    reward = rewards.get(next_state, 0)
                    new_value += (reward + discount_factor * value_table[next_state])
                new_value /= len(model[state][action])
                max_value = max(max_value, new_value)
            value_table[state] = max_value
            delta = max(delta, abs(v - value_table[state]))
        logging.info(f"ADP Delta: {delta}")
        if delta < theta:
            break
    return value_table

# TD (Temporal Difference): This method updates the cell values based on the robot’s actual experiences and movements.
def td_training(sequences, rewards):
    value_table = np.zeros(grid_size)
    
    for i, sequence in enumerate(sequences):
        for (state, action, next_state) in sequence:
            reward = rewards.get(next_state, 0)
            value_table[state] += alpha * (reward + discount_factor * value_table[next_state] - value_table[state])
        logging.info(f"TD Training on Sequence {i + 1}/{len(sequences)}")
    return value_table

# DUE (Direct Utility Estimation): This method estimates the cell values directly from the rewards the robot collects.
def due_training(sequences, rewards):
    returns = {state: [] for state in np.ndindex(grid_size)}
    value_table = np.zeros(grid_size)

    for i, sequence in enumerate(sequences):
        G = 0
        for (state, action, next_state) in reversed(sequence):
            reward = rewards.get(next_state, 0)
            G = discount_factor * G + reward
            if state not in [x[0] for x in sequence[:-1]]:
                returns[state].append(G)
                value_table[state] = np.mean(returns[state])
        logging.info(f"DUE Training on Sequence {i + 1}/{len(sequences)}")
    return value_table
