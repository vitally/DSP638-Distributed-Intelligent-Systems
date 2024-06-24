import logging
import matplotlib.pyplot as plt
from config import grid_size, obstacles, rewards, cost_values
from environment import update_rewards
from value_iteration import value_iteration
from sequence_generation import generate_sequences
from learning_methods import adp_training, td_training, due_training
from policy_utils import derive_policy_from_values, generate_random_start_state, generate_path, compare_policies, analyze_policy_changes
from visualization import plot_values, plot_policy_with_path, visualize_policy

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

def main():
    # Task 1-3: Perform value iteration to determine the optimal policy and value table
    policy, value_table = value_iteration()
    logging.info("Value Iteration Completed")

    # Task 4: Generate state transition sequences based on the derived policy
    sequences = generate_sequences(policy)
    logging.info("Sequence Generation Completed")

    # Task 5: Perform passive learning using ADP, TD, and DUE methods
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    all_policies = []

    for i, cost_value in enumerate(cost_values):
        updated_rewards = update_rewards(cost_value)
        
        # ADP Training
        adp_values = adp_training(sequences, updated_rewards)
        logging.info(f"ADP Training Completed with cost value {cost_value}")
        
        # TD Training
        td_values = td_training(sequences, updated_rewards)
        logging.info(f"TD Training Completed with cost value {cost_value}")
        
        # DUE Training
        due_values = due_training(sequences, updated_rewards)
        logging.info(f"DUE Training Completed with cost value {cost_value}")
        
        # Plot the value tables for ADP, TD, and DUE methods
        plot_values(adp_values, f'ADP Value Table with cost value {cost_value}', axes[i, 0])
        plot_values(td_values, f'TD Value Table with cost value {cost_value}', axes[i, 1])
        plot_values(due_values, f'DUE Value Table with cost value {cost_value}', axes[i, 2])

        # Derive policies from the trained value tables
        adp_policy = derive_policy_from_values(adp_values, updated_rewards)
        td_policy = derive_policy_from_values(td_values, updated_rewards)
        due_policy = derive_policy_from_values(due_values, updated_rewards)

        # Compare the derived policies
        compare_policies(adp_policy, td_policy, due_policy, cost_value)

        all_policies.append((adp_policy, td_policy, due_policy))

    plt.tight_layout()
    plt.show()

    # Generate random start state for path generation
    start_state = generate_random_start_state()

    # Derive policies for path generation
    adp_policy = derive_policy_from_values(adp_values, rewards)
    td_policy = derive_policy_from_values(td_values, rewards)
    due_policy = derive_policy_from_values(due_values, rewards)

    # Generate paths based on the derived policies
    path_adp = generate_path(adp_policy, start_state)
    path_td = generate_path(td_policy, start_state)
    path_due = generate_path(due_policy, start_state)

    print("Random Start State:", start_state)
    print("Path using ADP values:")
    print(path_adp)
    print("Path using TD values:")
    print(path_td)
    print("Path using DUE values:")
    print(path_due)

    # Plot the policies and paths
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    plot_policy_with_path(grid_size, adp_policy, path_adp, "ADP Policy with Path", axs[0])
    plot_policy_with_path(grid_size, td_policy, path_td, "TD Policy with Path", axs[1])
    plot_policy_with_path(grid_size, due_policy, path_due, "DUE Policy with Path", axs[2])

    plt.tight_layout()
    plt.show()

    # Analyze and print policy changes across different cost values
    policy_analysis = analyze_policy_changes(all_policies, cost_values)
    print("\nPolicy Change Analysis:")
    print(policy_analysis)

if __name__ == "__main__":
    main()
