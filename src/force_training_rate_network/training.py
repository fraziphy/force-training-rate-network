# src/force_training_rate_network/training.py
import numpy as np

class FORCETrainer:
    _usage_guide = """
    FORCETrainer Usage Guide:

    1. Initialize the FORCE trainer:
    trainer = FORCETrainer(network, target, stop_period, dt, l_steps)

    Parameters:
    - network: An instance of GeneratorNetwork or GeneratorNetworkFeedback.
    - target (numpy.ndarray): Target output for the network to learn. Shape: (N_readout, period).
    - stop_period (int): Number of periods to train for.
    - dt (float): Simulation time step in milliseconds.
    - l_steps (int): Number of time steps between weight updates.

    2. Run the training:
    trainer.train()

    This method runs the entire training process, including prior learning, training, and post-learning phases.

    3. Get the results:
    results = trainer.get_results()

    Returns a tuple containing:
    - J_GG_initial: Initial internal connectivity matrix.
    - J_GG_final: Final internal connectivity matrix.
    - rate_all: Activity of selected neurons over time.
    - Z_all: Readout activity over time.
    - W_all: Readout weights at start, after first update, and end of training.
    - W_dot: Norm of weight changes during training.

    Example usage:

    import numpy as np
    from your_module import GeneratorNetwork, FORCETrainer

    # Initialize a generator network
    network = GeneratorNetwork(N_network=1000, N_readout=1, g_GG=1.5, p_GG=0.1, p_z=0.1, tau=10.0)

    # Define target function (e.g., sine wave)
    period = 1000
    time = np.arange(0, period, 0.1)
    target = np.sin(2 * np.pi * time / period).reshape(1, -1)

    # Set training parameters
    stop_period = 10
    dt = 0.1
    l_steps = 10

    # Create and run FORCE trainer
    trainer = FORCETrainer(network, target, stop_period, dt, l_steps)
    trainer.train()

    # Get results
    J_GG_initial, J_GG_final, rate_all, Z_all, W_all, W_dot = trainer.get_results()

    # Analyze or visualize results
    import matplotlib.pyplot as plt

    # Plot readout activity vs target
    plt.figure(figsize=(12, 6))
    plt.plot(Z_all[0], label='Network output')
    plt.plot(np.tile(target[0], stop_period + 4), label='Target')
    plt.xlabel('Time step')
    plt.ylabel('Activity')
    plt.title('Network Output vs Target')
    plt.legend()
    plt.show()

    # Plot weight changes during training
    plt.figure(figsize=(10, 6))
    plt.plot(W_dot[0])
    plt.xlabel('Update step')
    plt.ylabel('Weight change (norm)')
    plt.title('Weight Changes During Training')
    plt.show()

    Important notes:
    - Ensure that the `network` passed to the `FORCETrainer` is properly initialized.
    - The `target` shape should match the number of readout neurons in the network.
    - Choose `stop_period`, `dt`, and `l_steps` carefully to balance training time and performance.
    - The `train()` method may take significant time to run for large networks or long training periods.
    """

    @property
    def help(self):
        print(self._usage_guide)

    def __init__(self, network, target=np.array([1, 2, 3, 4, 5, 6 , 7, 8, 9
                                                 1, 2, 3, 4, 5, 6 , 7, 9, 9
                                                 1, 2, 3, 4, 5, 6 , 7, 8, 9]), stop_period=10, dt=0.1, l_steps=1):
        self.network = network
        self.target = target
        self.stop_period = stop_period
        self.dt = dt
        self.l_steps = l_steps

        self.period = target.shape[1]
        self.T_prior_learning = 2 * self.period
        self.T_training = stop_period * self.period
        self.T_post_learning = 2 * self.period
        self.total_sim = self.T_prior_learning + self.T_training + self.T_post_learning

        self.initialize_arrays()

    def initialize_arrays(self):
        self.rate_all = np.empty((4, int(self.total_sim / 10)), dtype=float)
        self.Z_all = np.empty((self.network.N_readout, self.total_sim), dtype=float)
        self.W_all = np.empty((3, self.network.N_network, 1), dtype=float)

        # Correctly calculate size based on number of updates during training
        num_updates = int(self.T_training / int(self.l_steps / self.dt))
        self.W_dot = np.empty((self.network.N_readout, num_updates), dtype=float)

        self.J_GG_initial = self.network.J_GG.copy()

    def run_prior_learning(self):
        for i in range(self.T_prior_learning):
            self.update_state(i)

    def run_training(self):
        index_W_dot = 0  # Explicit index tracker for W_dot
        self.W_all[0] = self.network.W

        for i in range(self.T_training):
            if i % int(self.l_steps / self.dt) == 0:
                W_dot_aux = self.network.W.copy()
                self.network.weight_update(self.target[:, i % self.period])

                # Use explicit index tracking to avoid out-of-bounds errors
                if index_W_dot < self.W_dot.shape[1]:
                    self.W_dot[:, index_W_dot] = np.linalg.norm(self.network.W - W_dot_aux, axis=0)
                    index_W_dot += 1

                if i == 0:
                    self.W_all[1] = self.network.W

            self.update_state(i + self.T_prior_learning)

    def run_post_learning(self):
        for i in range(self.T_post_learning):
            self.update_state(i + self.T_prior_learning + self.T_training)

    def update_state(self, i):
        if i % 10 == 0:
            rate_index = i // 10
            if rate_index < self.rate_all.shape[1]:  # Ensure bounds are respected
                self.rate_all[:, rate_index] = self.network.rate[:4, 0]

        if i < self.Z_all.shape[1]:  # Ensure bounds are respected
            self.Z_all[:, [i]] = self.network.Z

        # Update network state
        self.network.state_update(self.dt)

    def train(self):
        # Run all phases of training
        self.run_prior_learning()
        self.run_training()
        self.W_all[2] = self.network.W
        self.run_post_learning()

    def get_results(self):
        return (self.J_GG_initial,
                self.network.J_GG,
                self.rate_all,
                self.Z_all,
                self.W_all,
                self.W_dot)


# Usage
# trainer = FORCETrainer(network, target, stop_period, dt, l_steps)
# trainer.train()
# results = trainer.get_results()
