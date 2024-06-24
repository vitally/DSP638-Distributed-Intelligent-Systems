import numpy as np
from config import grid_size, obstacles, rewards, transition_probs

# Task 1: Environment Configuration

# Helper function to check if a state is terminal
def is_terminal(state):
    return state in rewards

# Helper function to get possible transitions from a given state and action
def get_transitions(state, action):
    transitions = []
    x, y = state
    if action == 'up':
        intended = (x-1, y)
        transitions.append((transition_probs['up'], intended))
        transitions.append((transition_probs['down'], (x+1, y)))
        transitions.append((transition_probs['left'], (x, y-1)))
        transitions.append((transition_probs['right'], (x, y+1)))
    elif action == 'down':
        intended = (x+1, y)
        transitions.append((transition_probs['down'], intended))
        transitions.append((transition_probs['up'], (x-1, y)))
        transitions.append((transition_probs['left'], (x, y-1)))
        transitions.append((transition_probs['right'], (x, y+1)))
    elif action == 'left':
        intended = (x, y-1)
        transitions.append((transition_probs['left'], intended))
        transitions.append((transition_probs['right'], (x, y+1)))
        transitions.append((transition_probs['up'], (x-1, y)))
        transitions.append((transition_probs['down'], (x+1, y)))
    elif action == 'right':
        intended = (x, y+1)
        transitions.append((transition_probs['right'], intended))
        transitions.append((transition_probs['left'], (x, y-1)))
        transitions.append((transition_probs['up'], (x-1, y)))
        transitions.append((transition_probs['down'], (x+1, y)))
    
    valid_transitions = []
    for prob, next_state in transitions:
        next_state = (max(0, min(next_state[0], grid_size[0] - 1)),
                        max(0, min(next_state[1], grid_size[1] - 1)))
        if next_state in obstacles:
            next_state = state
        valid_transitions.append((prob, next_state))
    
    return valid_transitions

# Function to update rewards based on different cost values
def update_rewards(cost_value):
    new_rewards = rewards.copy()
    for state in np.ndindex(grid_size):
        if state not in rewards:
            new_rewards[state] = cost_value
    return new_rewards
