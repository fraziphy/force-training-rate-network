# tests/test_models.py
import unittest
import sys
import os
import numpy as np

# Add the src directory to sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, src_path)

from force_training_rate_network.models import GeneratorNetwork, GeneratorNetworkFeedback

class TestNetworks(unittest.TestCase):
    def test_generator_network(self):
        # Initialize the network
        N = 100
        N_readout = 1
        g_GG = 1.5
        network = GeneratorNetwork(N_network=N, N_readout=N_readout, g_GG=g_GG)

        # Test state update
        dt = 0.1
        initial_rate = network.rate.copy()
        network.state_update(dt)
        self.assertFalse(np.allclose(initial_rate, network.rate), "State should change after update")

        # Test weight update
        target = np.random.rand(N_readout, 1)
        initial_W = network.W.copy()
        initial_J_GG = network.J_GG.copy()
        network.weight_update(target)
        self.assertFalse(np.allclose(initial_W, network.W), "W should change after weight update")
        self.assertFalse(np.allclose(initial_J_GG, network.J_GG), "J_GG should change after weight update")

    def test_generator_network_feedback(self):
        # Initialize the network
        N = 100
        N_readout = 1
        g_GG = 1.5
        g_GZ = 1.0

        # Create RNG_conn with at least 4 random number generators
        RNG_conn = [np.random.default_rng(seed) for seed in range(4)]

        network = GeneratorNetworkFeedback(N_network=N, N_readout=N_readout, g_GG=g_GG, g_GZ=g_GZ, RNG_conn=RNG_conn)

        # Test state update
        dt = 0.1
        initial_rate = network.rate.copy()
        network.state_update(dt)
        self.assertFalse(np.allclose(initial_rate, network.rate), "State should change after update")

        # Test weight update
        target = np.random.rand(N_readout, 1)
        initial_W = network.W.copy()
        initial_J_GG = network.J_GG.copy()
        network.weight_update(target)
        self.assertFalse(np.allclose(initial_W, network.W), "W should change after weight update")
        self.assertTrue(np.allclose(initial_J_GG, network.J_GG), "J_GG should not change after weight update in feedback network")

if __name__ == '__main__':
    unittest.main()
