# tests/test_models.py
import unittest
import numpy as np
import sys
import os

# Add the src directory to sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, src_path)

from force_training_rate_network.models import GeneratorNetwork
from force_training_rate_network.lyapunov import LyapunovExponentCalculator

class TestLyapunovExponentCalculator(unittest.TestCase):

    def setUp(self):
        # Create a mock network for testing
        self.mock_network = GeneratorNetwork(N_network=100, N_readout=1)
        self.simulation_time = 1000
        self.dt = 0.1
        self.renorm_interval = 50
        self.delta_separation = 1e-10

    def test_initialization(self):
        calculator = LyapunovExponentCalculator(self.mock_network, self.simulation_time, self.dt, self.renorm_interval, self.delta_separation)
        self.assertEqual(calculator.simulation_time, self.simulation_time)
        self.assertEqual(calculator.dt, self.dt)
        self.assertEqual(calculator.renorm_interval, self.renorm_interval)
        self.assertEqual(calculator.delta_separation, self.delta_separation)
        self.assertEqual(calculator.num_renorms, int(self.simulation_time / self.renorm_interval))
        self.assertEqual(calculator.steps_per_renorm, int(self.renorm_interval / self.dt))

    def test_initialize_perturbation(self):
        calculator = LyapunovExponentCalculator(self.mock_network, self.simulation_time, self.dt, self.renorm_interval, self.delta_separation)
        calculator.initialize_perturbation()
        self.assertIsNotNone(calculator.network_perturbed)
        self.assertFalse(np.array_equal(calculator.network.rate, calculator.network_perturbed.rate))
        self.assertAlmostEqual(np.linalg.norm(calculator.network.rate - calculator.network_perturbed.rate), self.delta_separation, places=12)

    def test_compute(self):
        calculator = LyapunovExponentCalculator(self.mock_network, self.simulation_time, self.dt, self.renorm_interval, self.delta_separation)
        calculator.initialize_perturbation()
        calculator.compute()
        self.assertIsNotNone(calculator.lyapunov_exponents)
        self.assertEqual(len(calculator.lyapunov_exponents), calculator.num_renorms)
        self.assertEqual(len(calculator.diff_over_time), int(self.simulation_time / self.dt))

    def test_get_results_without_trajectories(self):
        calculator = LyapunovExponentCalculator(self.mock_network, self.simulation_time, self.dt, self.renorm_interval, self.delta_separation, store_trajectories=False)
        calculator.initialize_perturbation()
        calculator.compute()
        results = calculator.get_results()

        self.assertEqual(len(results), 5)
        self.assertIsInstance(results[0], float)  # lyapunov_exponent
        self.assertIsInstance(results[1], np.ndarray)  # diff_over_time
        self.assertIsNone(results[2])  # rates_original
        self.assertIsNone(results[3])  # rates_perturbed
        self.assertIsInstance(results[4], np.ndarray)  # true_diff_over_time
        self.assertEqual(len(results[4]), 0)  # true_diff_over_time should be empty

    def test_get_results_with_trajectories(self):
        calculator = LyapunovExponentCalculator(self.mock_network, self.simulation_time, self.dt, self.renorm_interval, self.delta_separation, store_trajectories=True)
        calculator.initialize_perturbation()
        calculator.compute()
        results = calculator.get_results()

        self.assertEqual(len(results), 5)
        self.assertIsInstance(results[0], float)  # lyapunov_exponent
        self.assertIsInstance(results[1], np.ndarray)  # diff_over_time
        self.assertIsInstance(results[2], np.ndarray)  # rates_original
        self.assertIsInstance(results[3], np.ndarray)  # rates_perturbed
        self.assertIsInstance(results[4], np.ndarray)  # true_diff_over_time

        # Check shapes of arrays
        self.assertEqual(results[2].shape[0], 4)  # rates_original should have 4 rows
        self.assertEqual(results[3].shape[0], 4)  # rates_perturbed should have 4 rows
        self.assertEqual(len(results[4]), int(self.simulation_time / self.dt))  # true_diff_over_time

    def test_lyapunov_exponent_range(self):
        calculator = LyapunovExponentCalculator(self.mock_network, self.simulation_time, self.dt, self.renorm_interval, self.delta_separation)
        calculator.initialize_perturbation()
        calculator.compute()
        lyapunov_exponent = calculator.get_results()[0]

        # The Lyapunov exponent should typically be a non-zero value
        # This is a very loose check and might need adjustment based on your specific system
        self.assertNotEqual(lyapunov_exponent, 0)
        self.assertIsInstance(lyapunov_exponent, float)

if __name__ == '__main__':
    unittest.main()
