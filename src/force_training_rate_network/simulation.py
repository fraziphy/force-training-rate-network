# src/force_training_rate_network/simulation.py

import numpy as np

class SimulationEngine:
    def __init__(self, network, simulation_time=200, dt=0.1):
        """
        Initialize the simulation engine.

        Args:
            network: An instance of GeneratorNetwork or GeneratorNetworkFeedback.
            simulation_time (float): Total simulation time in milliseconds.
            dt (float): Simulation time step in milliseconds.
        """
        self.network = network
        self.simulation_time = simulation_time
        self.dt = dt
        self.time_steps = int(simulation_time / dt)
        self.time = np.arange(0, simulation_time, dt)

    def run(self):
        """
        Run the simulation for the specified time.
        """

        # Initialize arrays to store results
        rates = np.zeros((self.network.N_network, self.time_steps))
        readouts = np.zeros((self.network.N_readout, self.time_steps))

        # Simulate the network
        for t_idx, t in enumerate(self.time):
            self.network.state_update(self.dt)

            # Store the rates and readouts
            rates[:, t_idx] = self.network.rate.flatten()
            readouts[:, t_idx] = self.network.Z.flatten()

        # Return results for further analysis or visualization
        return rates, readouts
