# test_simulation.py
import unittest
import numpy as np
import sys
import os

# Add the src directory to sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, src_path)

from force_training_rate_network.models import GeneratorNetwork, GeneratorNetworkFeedback
from force_training_rate_network.simulation import SimulationEngine

class TestSimulationEngine(unittest.TestCase):

    def setUp(self):
        # Create mock networks for testing
        self.gen_network = GeneratorNetwork(N_network=100, N_readout=1)
        self.gen_network_feedback = GeneratorNetworkFeedback(N_network=100, N_readout=1)
        self.simulation_time = 200
        self.dt = 0.1

    def test_initialization(self):
        engine = SimulationEngine(self.gen_network, self.simulation_time, self.dt)
        self.assertEqual(engine.simulation_time, self.simulation_time)
        self.assertEqual(engine.dt, self.dt)
        self.assertEqual(engine.time_steps, int(self.simulation_time / self.dt))
        self.assertEqual(len(engine.time), engine.time_steps)

    def test_run_generator_network(self):
        engine = SimulationEngine(self.gen_network, self.simulation_time, self.dt)
        rates, readouts = engine.run()

        # Check shapes of output
        self.assertEqual(rates.shape, (self.gen_network.N_network, engine.time_steps))
        self.assertEqual(readouts.shape, (self.gen_network.N_readout, engine.time_steps))

        # Check that rates and readouts are not all zero
        self.assertFalse(np.all(rates == 0))
        self.assertFalse(np.all(readouts == 0))

    def test_run_generator_network_feedback(self):
        engine = SimulationEngine(self.gen_network_feedback, self.simulation_time, self.dt)
        rates, readouts = engine.run()

        # Check shapes of output
        self.assertEqual(rates.shape, (self.gen_network_feedback.N_network, engine.time_steps))
        self.assertEqual(readouts.shape, (self.gen_network_feedback.N_readout, engine.time_steps))

        # Check that rates and readouts are not all zero
        self.assertFalse(np.all(rates == 0))
        self.assertFalse(np.all(readouts == 0))

    def test_time_array(self):
        engine = SimulationEngine(self.gen_network, self.simulation_time, self.dt)
        self.assertEqual(len(engine.time), engine.time_steps)
        self.assertAlmostEqual(engine.time[1] - engine.time[0], self.dt)
        self.assertAlmostEqual(engine.time[-1], self.simulation_time - self.dt)

if __name__ == '__main__':
    unittest.main()
