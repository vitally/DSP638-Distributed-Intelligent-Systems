import numpy as np
import logging
from config import grid_size, obstacles, rewards, transition_probs, discount_factor, theta
from environment import is_terminal, get_transitions

# Task 3: Fixed Policy Selection using Value Iteration

def value_iteration():
    # Initialize value table with zeros
    value_table = np.zeros(grid_size)
    # Initialize policy with 'up' action for all states
    policy = np.full(grid_size, 'up', dtype=object)
    
    iteration = 0
    while True:
        delta = 0
        iteration += 1
        logging.info(f"Value Iteration: {iteration}")
        for state in np.ndindex(grid_size):
            if is_terminal(state) or state in obstacles:
                continue
            v = value_table[state]
            max_value = float('-inf')
            best_action = None
            # Evaluate each action to find the best one
            for action in ['up', 'down', 'left', 'right']:
                new_value = 0
                for prob, next_state in get_transitions(state, action):
                    reward = rewards.get(next_state, 0)
                    new_value += prob * (reward + discount_factor * value_table[next_state])
                if new_value > max_value:
                    max_value = new_value
                    best_action = action
            value_table[state] = max_value
            policy[state] = best_action
            delta = max(delta, abs(v - value_table[state]))
        logging.info(f"Delta: {delta}")
        if delta < theta:
            break
    return policy, value_table
