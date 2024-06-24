import unittest
from config import grid_size, obstacles, rewards, cost_values
from environment import update_rewards
from value_iteration import value_iteration
from sequence_generation import generate_sequences
from learning_methods import adp_training, td_training, due_training
from policy_utils import derive_policy_from_values, generate_random_start_state, generate_path, compare_policies, analyze_policy_changes

class TestSolution(unittest.TestCase):

    def setUp(self):
        self.policy, self.value_table = value_iteration()
        self.sequences = generate_sequences(self.policy)
        self.cost_value = cost_values[0]
        self.updated_rewards = update_rewards(self.cost_value)
    
    def test_value_iteration(self):
        # Ensure value iteration generates valid policy and value table
        self.assertIsNotNone(self.policy, "Policy should not be None")
        self.assertIsNotNone(self.value_table, "Value table should not be None")
        self.assertEqual(self.policy.shape, grid_size, "Policy should cover the entire grid")
    
    def test_sequence_generation(self):
        # Ensure sequences are generated correctly
        self.assertGreater(len(self.sequences), 0, "Sequences should be generated")
    
    def test_adp_training(self):
        adp_values = adp_training(self.sequences, self.updated_rewards)
        adp_policy = derive_policy_from_values(adp_values, self.updated_rewards)
        self.assertIsNotNone(adp_policy, "ADP policy should not be None")
    
    def test_td_training(self):
        td_values = td_training(self.sequences, self.updated_rewards)
        td_policy = derive_policy_from_values(td_values, self.updated_rewards)
        self.assertIsNotNone(td_policy, "TD policy should not be None")
    
    def test_due_training(self):
        due_values = due_training(self.sequences, self.updated_rewards)
        due_policy = derive_policy_from_values(due_values, self.updated_rewards)
        self.assertIsNotNone(due_policy, "DUE policy should not be None")
    
    def test_path_generation(self):
        adp_values = adp_training(self.sequences, self.updated_rewards)
        adp_policy = derive_policy_from_values(adp_values, rewards)
        start_state = generate_random_start_state()
        path_adp = generate_path(adp_policy, start_state)
        self.assertGreater(len(path_adp), 0, "Path should be generated using ADP policy")
    
    def test_policy_changes(self):
        all_policies = []
        for cost_value in cost_values:
            updated_rewards = update_rewards(cost_value)
            adp_values = adp_training(self.sequences, updated_rewards)
            td_values = td_training(self.sequences, updated_rewards)
            due_values = due_training(self.sequences, updated_rewards)
            adp_policy = derive_policy_from_values(adp_values, updated_rewards)
            td_policy = derive_policy_from_values(td_values, updated_rewards)
            due_policy = derive_policy_from_values(due_values, updated_rewards)
            all_policies.append((adp_policy, td_policy, due_policy))
        policy_analysis = analyze_policy_changes(all_policies, cost_values)
        self.assertIsNotNone(policy_analysis, "Policy analysis should not be None")

if __name__ == "__main__":
    unittest.main()