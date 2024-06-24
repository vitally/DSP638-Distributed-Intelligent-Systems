import matplotlib.pyplot as plt
import numpy as np
from config import obstacles, rewards

# Display the value tables for comparison
def plot_values(values, title, ax):
    cax = ax.imshow(values, cmap='coolwarm', interpolation='none', aspect='equal')
    ax.set_title(title)
    plt.colorbar(cax, ax=ax)
     # Contour obstacles and rewards
    for (x, y) in obstacles:
        ax.scatter(y, x, marker='x', color='black', s=100, label='Obstacle' if (x, y) == obstacles[0] else "")
    for (x, y), reward in rewards.items():
        color = 'green' if reward > 0 else 'red'
        marker = 'o' if reward > 0 else 's'
        ax.scatter(y, x, marker=marker, color=color, s=100, label='Reward' if (x, y) == list(rewards.keys())[0] else "")

def plot_policy_with_path(grid_size, policy, path, title, ax):
    grid = np.full(grid_size, ' ')
    
    for state in np.ndindex(grid_size):
        if state in obstacles:
            grid[state] = 'X'
        elif state in rewards:
            grid[state] = 'R' if rewards[state] > 0 else 'P'  # R for positive reward, P for penalty
        else:
            grid[state] = policy[state][0].upper()
    
    for state, action in path:
        grid[state] = '.'
    
    ax.imshow(grid != ' ', cmap='gray', interpolation='none')
    for (i, j), val in np.ndenumerate(grid):
        if grid[i, j] != ' ':
            ax.text(j, i, val, ha='center', va='center', color='white' if grid[i, j] in ['X', 'R', 'P'] else 'yellow')
    
    ax.set_xticks(np.arange(grid_size[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid_size[0]) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    ax.set_title(title)

def visualize_policy(policy, grid_size, cost_value, method_name, ax):
    grid = np.full(grid_size, ' ', dtype=object)
    for state in np.ndindex(grid_size):
        if state in obstacles:
            grid[state] = 'X'
        elif state in rewards:
            grid[state] = 'R' if rewards[state] > 0 else 'P'
        else:
            grid[state] = policy[state][0].upper()
    
    ax.imshow(grid != ' ', cmap='cool', interpolation='none')
    for (i, j), val in np.ndenumerate(grid):
        ax.text(j, i, val, ha='center', va='center', color='red' if val in ['X', 'R', 'P'] else 'blue')
    
    ax.set_xticks(np.arange(grid_size[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid_size[0]) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    ax.set_title(f"{method_name} Policy (Cost: {cost_value})")
