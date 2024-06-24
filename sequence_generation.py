import random
import logging
from collections import defaultdict
from config import grid_size, obstacles, rewards
from environment import is_terminal

# Task 4: Generating State Transition Sequences
def generate_sequences(policy, num_sequences=20):
    sequences = []
    for i in range(num_sequences):
        sequence = []
        state = (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))

        # Ensure the initial state is not an obstacle
        while state in obstacles:
            state = (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))

        state_count = defaultdict(int)
        steps = 0 # Counter to prevent infinite loops
        max_steps = 100 # Maximum steps to take before breaking the loop
        while state not in rewards and steps < max_steps:
            state_count[state] += 1
            if state_count[state] > 4: # Detecting cycles
                logging.info(f"Detected cycle at state {state}, breaking cycle.")
                break
            action = policy[state]
            if action == 'up':
                next_state = (state[0] - 1, state[1])
            elif action == 'down':
                next_state = (state[0] + 1, state[1])
            elif action == 'left':
                next_state = (state[0], state[1] - 1)
            elif action == 'right':
                next_state = (state[0], state[1] + 1)
            next_state = (max(0, min(next_state[0], grid_size[0] - 1)), 
                          max(0, min(next_state[1], grid_size[1] - 1)))
            if next_state in obstacles or state == next_state:
                logging.info(f"Detected obstacle or no movement at state {state}, breaking.")
                break
            if is_terminal(next_state):
                break
            sequence.append((state, action, next_state))
            state = next_state
            steps += 1
        if steps >= max_steps:
            logging.info(f"Sequence {i + 1} reached max steps limit, breaking the loop.")
        sequences.append(sequence)
        logging.info(f"Generated Sequence {i + 1}/{num_sequences}")
    return sequences
