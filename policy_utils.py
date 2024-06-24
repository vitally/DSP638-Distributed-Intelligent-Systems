import numpy as np
import random
from config import grid_size, obstacles, rewards, discount_factor
from environment import is_terminal, get_transitions

def derive_policy_from_values(value_table, rewards):
    policy = np.full(grid_size, 'up', dtype=object)
    for state in np.ndindex(grid_size):
        if is_terminal(state) or state in obstacles:
            continue
        max_value = float('-inf')
        best_action = None
        for action in ['up', 'down', 'left', 'right']:
            new_value = 0
            for prob, next_state in get_transitions(state, action):
                reward = rewards.get(next_state, 0)
                new_value += prob * (reward + discount_factor * value_table[next_state])
            if new_value > max_value:
                max_value = new_value
                best_action = action
        policy[state] = best_action
    return policy

def generate_path(policy, start_state):
    path = []
    state = start_state
    while not is_terminal(state):
        action = policy[state]
        path.append((state, action))
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
            break
        state = next_state
    return path

def generate_random_start_state():
    while True:
        start_state = (random.randint(0, grid_size[0] - 1), random.randint(0, grid_size[1] - 1))
        if start_state not in obstacles and not is_terminal(start_state):
            return start_state

def compare_policies(adp_policy, td_policy, due_policy, cost_value):
    print(f"\nPolicy Comparison for Cost Value: {cost_value}")
    print("State\t\tADP\tTD\tDUE")
    for state in np.ndindex(grid_size):
        if state not in obstacles and not is_terminal(state):
            print(f"{state}\t\t{adp_policy[state]}\t{td_policy[state]}\t{due_policy[state]}")
    print("\n")

def analyze_policy_changes(policies, cost_values):
    summary = []
    for i, cost in enumerate(cost_values):
        adp_policy, td_policy, due_policy = policies[i]
        
        diff_adp_td = sum(1 for state in np.ndindex(grid_size) if adp_policy[state] != td_policy[state])
        diff_adp_due = sum(1 for state in np.ndindex(grid_size) if adp_policy[state] != due_policy[state])
        diff_td_due = sum(1 for state in np.ndindex(grid_size) if td_policy[state] != due_policy[state])
        
        adp_most_common = max(set(adp_policy.flatten()), key=list(adp_policy.flatten()).count)
        td_most_common = max(set(td_policy.flatten()), key=list(td_policy.flatten()).count)
        due_most_common = max(set(due_policy.flatten()), key=list(due_policy.flatten()).count)
        
        summary.append(f"Cost value {cost}:\n"
                       f"- Differences: ADP-TD: {diff_adp_td}, ADP-DUE: {diff_adp_due}, TD-DUE: {diff_td_due}\n"
                       f"- Most common actions: ADP: {adp_most_common}, TD: {td_most_common}, DUE: {due_most_common}\n")
    
    return "\n".join(summary)
