# src/force_training_rate_network/simulation.py

import numpy as np

class SimulationEngine:
    _usage_guide = """
    SimulationEngine Usage Guide:

    1. Initialize the simulation engine:
    engine = SimulationEngine(network, simulation_time=200, dt=0.1)

    Parameters:
    - network: An instance of GeneratorNetwork or GeneratorNetworkFeedback.
    - simulation_time (float): Total simulation time in milliseconds (default: 200).
    - dt (float): Simulation time step in milliseconds (default: 0.1).

    2. Run the simulation:
    rates, readouts = engine.run()

    This method simulates the network dynamics for the specified simulation time and returns:
    - rates (numpy.ndarray): The activity of all neurons in the network over time. Shape: (N_network, time_steps).
    - readouts (numpy.ndarray): The activity of readout neurons over time. Shape: (N_readout, time_steps).

    Example usage:

    import numpy as np
    from your_module import GeneratorNetwork, SimulationEngine

    # Initialize a generator network
    network = GeneratorNetwork(N_network=1000, N_readout=1, g_GG=1.5, p_GG=0.1, p_z=0.1, tau=10.0)

    # Create a simulation engine
    engine = SimulationEngine(network, simulation_time=500, dt=0.1)

    # Run the simulation
    rates, readouts = engine.run()

    # Analyze or visualize results
    import matplotlib.pyplot as plt

    # Plot rates of a few neurons
    plt.figure(figsize=(10, 6))
    for neuron_idx in range(5):  # Plot activity of first 5 neurons
        plt.plot(engine.time, rates[neuron_idx], label=f'Neuron {neuron_idx}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Rate')
    plt.title('Neuron Activity Over Time')
    plt.legend()
    plt.show()

    # Plot readout activity
    plt.figure(figsize=(10, 6))
    plt.plot(engine.time, readouts[0], label='Readout Neuron')
    plt.xlabel('Time (ms)')
    plt.ylabel('Readout Activity')
    plt.title('Readout Neuron Activity Over Time')
    plt.legend()
    plt.show()

    Important notes:
    - Ensure that the `network` passed to the `SimulationEngine` is properly initialized and configured.
    - The `dt` parameter should be chosen carefully to ensure numerical stability during simulation.
    - The `simulation_time` should be long enough to capture meaningful dynamics but not excessively large to avoid memory issues.
    """

    @property
    def help(self):
        print(self._usage_guide)

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

            # Store the rates and readouts
            rates[:, t_idx] = self.network.rate.flatten()
            readouts[:, t_idx] = self.network.Z.flatten()


            self.network.state_update(self.dt)

        # Return results for further analysis or visualization
        return rates, readouts
