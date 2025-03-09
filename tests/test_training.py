# test_training.py
import unittest
import numpy as np
import sys
import os

# Add the src directory to sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, src_path)

from force_training_rate_network.models import GeneratorNetwork
from force_training_rate_network.training import FORCETrainer

class TestFORCETrainer(unittest.TestCase):

    def setUp(self):
        # Create a mock network and target for testing
        self.N_network = 100
        self.N_readout = 1
        self.network = GeneratorNetwork(N_network=self.N_network, N_readout=self.N_readout)
        self.target = np.sin(np.linspace(0, 2*np.pi, 1000)).reshape(1, -1)  # Simple sine wave target
        self.stop_period = 5
        self.dt = 0.1
        self.l_steps = 10

    def test_initialization(self):
        trainer = FORCETrainer(self.network, self.target, self.stop_period, self.dt, self.l_steps)

        self.assertEqual(trainer.period, self.target.shape[1])
        self.assertEqual(trainer.T_prior_learning, 2 * trainer.period)
        self.assertEqual(trainer.T_training, self.stop_period * trainer.period)
        self.assertEqual(trainer.T_post_learning, 2 * trainer.period)

        expected_total_sim = trainer.T_prior_learning + trainer.T_training + trainer.T_post_learning
        self.assertEqual(trainer.total_sim, expected_total_sim)

    def test_array_initialization(self):
        trainer = FORCETrainer(self.network, self.target, self.stop_period, self.dt, self.l_steps)

        expected_rate_all_shape = (4, int(trainer.total_sim / 10))
        self.assertEqual(trainer.rate_all.shape, expected_rate_all_shape)

        expected_Z_all_shape = (self.N_readout, trainer.total_sim)
        self.assertEqual(trainer.Z_all.shape, expected_Z_all_shape)

        expected_W_all_shape = (3, self.N_network, 1)
        self.assertEqual(trainer.W_all.shape, expected_W_all_shape)

        expected_W_dot_shape = (self.N_readout, int(trainer.T_training / int(self.l_steps / self.dt)))
        self.assertEqual(trainer.W_dot.shape, expected_W_dot_shape)

    def test_train(self):
        trainer = FORCETrainer(self.network, self.target, self.stop_period, self.dt, self.l_steps)
        trainer.train()

        # Check that arrays are filled after training
        self.assertFalse(np.all(trainer.rate_all == 0))
        self.assertFalse(np.all(trainer.Z_all == 0))
        self.assertFalse(np.all(trainer.W_all == 0))
        self.assertFalse(np.all(trainer.W_dot == 0))

    def test_get_results(self):
        trainer = FORCETrainer(self.network, self.target, self.stop_period, self.dt, self.l_steps)
        trainer.train()
        results = trainer.get_results()

        self.assertEqual(len(results), 6)
        self.assertIsInstance(results[0], np.ndarray)  # J_GG_initial
        self.assertIsInstance(results[1], np.ndarray)  # J_GG final
        self.assertIsInstance(results[2], np.ndarray)  # rate_all
        self.assertIsInstance(results[3], np.ndarray)  # Z_all
        self.assertIsInstance(results[4], np.ndarray)  # W_all
        self.assertIsInstance(results[5], np.ndarray)  # W_dot

    def test_weight_changes(self):
        trainer = FORCETrainer(self.network, self.target, self.stop_period, self.dt, self.l_steps)
        initial_W = trainer.network.W.copy()
        trainer.train()
        final_W = trainer.network.W

        # Check that weights have changed during training
        self.assertFalse(np.array_equal(initial_W, final_W))

if __name__ == '__main__':
    unittest.main()
