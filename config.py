import numpy as np
# Task 1: Environment Configuration

# Define the environment size
grid_size = (7, 6)
# Define obstacles positions in the grid
obstacles = [(0, 3), (1, 1), (3, 5), (5, 5)]
# Define rewards for specific states in the grid
rewards = {(6, 5): 1, (0, 0): -1}

# Task 2: Stochastic Model for Agent Actions

# Define movement probabilities for the agent
transition_probs = {
    'up': 0.6,
    'down': 0.1,
    'left': 0.13,
    'right': 0.17
}

# Learning parameters
discount_factor = 0.9
theta = 0.0001
alpha = 0.1

# Cost values for analysis
cost_values = [-1.2, -0.5, -0.06]
