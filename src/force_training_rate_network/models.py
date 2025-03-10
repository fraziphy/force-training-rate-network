# src/force_training_rate_network/models.py
import numpy as np
from .utils import connectivity_matrix_internal, connectivity_matrix_input_to_network
from .internal_weight_update_functions import (
    update_internal_weights_standard,
    update_internal_weights_scalar_g,
    update_internal_weights_vector_g
)

class GeneratorNetwork:
    _usage_guide = """
    GeneratorNetwork and GeneratorNetworkFeedback Usage Guide:

    1. Initialize the network:
    For GeneratorNetwork:
    network = GeneratorNetwork(N_network=1000, N_readout=1, g_GG=1.5, p_GG=0.1, p_z=0.1, tau=10.0, internal_weight_update_method='standard')

    For GeneratorNetworkFeedback:
    feedback_network = GeneratorNetworkFeedback(N_network=1000, N_readout=1, g_GG=1.5, g_GZ=1.0, p_GG=0.1, p_z=0.1, tau=10.0)

    Parameters:
    - N_network (int): Number of neurons in the generator network (default: 1000).
    - N_readout (int): Number of readout neurons (default: 1).
    - g_GG (float or np.ndarray): Synaptic strength in the generator network (default: 1.5). Can be scalar or array-like for GeneratorNetwork.
    - g_GZ (float): Feedback strength from readout to network (default: 1.0, only for GeneratorNetworkFeedback). Must be scalar.
    - p_GG (float): Connection probability for internal connections (default: 0.1).
    - p_z (float): Connection probability for readout connections (default: 0.1).
    - tau (float): Time constant for synaptic decay in milliseconds (default: 10.0).
    - internal_weight_update_method (str): Method for updating internal weights ('standard', 'scalar_g', or 'vector_g', only for GeneratorNetwork).
    - RNG_conn (list): List of random number generators for connectivity (optional).
    - RNG_init (numpy.random.Generator): Random number generator for initialization (optional).

    2. Update the network state:
    network.state_update(dt)

    Parameters:
    - dt (float): Time step for updating the neuron's state in milliseconds.

    3. Perform weight updates:
    network.weight_update(target_point)

    Parameters:
    - target_point (numpy.ndarray): Target output for readout neurons.

    4. Use child classes for specialized networks:
    - For a simple generator network without feedback, use `GeneratorNetwork`.
        Example:
        generator_network = GeneratorNetwork(N_network=1000, N_readout=1)

    - For a generator network with feedback from readout neurons to the network, use `GeneratorNetworkFeedback`.
        Example:
        feedback_network = GeneratorNetworkFeedback(N_network=1000, N_readout=1, g_GZ=1.0)

    Example usage:

    import numpy as np
    from your_module import GeneratorNetwork, GeneratorNetworkFeedback

    # Initialize networks
    network = GeneratorNetwork(N_network=1000, N_readout=1, g_GG=1.5, p_GG=0.1, p_z=0.1, tau=10.0)
    feedback_network = GeneratorNetworkFeedback(N_network=1000, N_readout=1, g_GG=1.5, g_GZ=1.0, p_GG=0.1, p_z=0.1, tau=10.0)

    # Simulation loop
    dt = 0.1
    simulation_time = 1000
    target_function = lambda t: np.sin(2 * np.pi * t / 1000)

    for t in range(int(simulation_time / dt)):
        # Update network states
        network.state_update(dt)
        feedback_network.state_update(dt)

        # Generate target point
        target = target_function(t * dt)
        target_point = np.array([[target]])

        # Perform weight updates
        network.weight_update(target_point)
        feedback_network.weight_update(target_point)

        # Here you can record or analyze network output
        # network_output = network.Z
        # feedback_network_output = feedback_network.Z

    Important notes:
    - Ensure that the dimensions of `target_point` match `N_readout`.
    - The time step `dt` must be non-zero to avoid ValueError.
    - For `GeneratorNetwork`, `g_GG` can be a scalar or an array of size `N_network`.
    - For `GeneratorNetworkFeedback`, `g_GZ` must be a scalar.
    - The internal weight update method is only applicable to `GeneratorNetwork`, not `GeneratorNetworkFeedback`.
    """

    @property
    def help(self):
        print(self._usage_guide)

    """
    Generator Network for FORCE training.
    """

    def __init__(self,
                 N_network=1000,
                 N_readout=1,
                 g_GG=1.5,  # Can be scalar or array-like
                 p_GG=0.1,
                 p_z=0.1,
                 tau=10.0,
                 internal_weight_update_method='standard',
                 RNG_conn=None,
                 RNG_init=None):
        """
        Initialize the GeneratorNetwork with default or user-defined parameters.

        Args:
            N_network (int): Number of neurons in the generator network.
            N_readout (int): Number of readout neurons.
            g_GG (float or np.ndarray): Synaptic strength in the generator network. Can be scalar or array-like.
            p_GG (float): Connection probability for internal connections.
            p_z (float): Connection probability for readout connections.
            tau (float): Time constant for synaptic decay.
            RNG_conn (list): List of random number generators for connectivity.
            RNG_init (numpy.random.Generator): Random number generator for initialization.
        """
        # Initialize parameters
        self.N_network = N_network
        self.N_readout = N_readout
        self.g_GG = g_GG  # Can be scalar or array-like
        self.p_GG = p_GG
        self.p_z = p_z
        self.tau = tau

        # Random number generators
        if RNG_conn is None:
            RNG_conn = [np.random.default_rng(seed) for seed in range(4)]
        if RNG_init is None:
            RNG_init = np.random.default_rng(0)
        self.RNG_conn = RNG_conn
        self.RNG_init = RNG_init

        # Initialize network state
        self.rate = 2 * self.RNG_init.random((self.N_network, 1)) - 1  # Neuron states in [-1, 1]

        # Generate connectivity matrices
        J_GG_raw = connectivity_matrix_internal(self.N_network, (self.N_network, self.N_network), self.p_GG, self.RNG_conn[0])

        # Handle g_GG being scalar or array-like
        if np.isscalar(self.g_GG):
            self.J_GG = self.g_GG * J_GG_raw
        else:
            if len(self.g_GG) != self.N_network:
                raise ValueError("g_GG must have the same size as N_network if it is an array.")
            self.J_GG = J_GG_raw * self.g_GG[:, np.newaxis]

        self.W = connectivity_matrix_internal(self.N_network, (self.N_network, self.N_readout), self.p_z, self.RNG_conn[1])

        self.I = np.zeros((1, 1), dtype=int)
        self.J_GI = connectivity_matrix_input_to_network(self.N_network, self.I.shape[0], self.RNG_conn[2])

        # Other state variables
        self.Z = self.W.T @ self.rate
        self.total_current = np.zeros_like(self.rate)
        self.P = np.eye(self.N_network)

        self.internal_weight_update_method = internal_weight_update_method
        self.internal_weight_update_functions = {
            'standard': update_internal_weights_standard,
            'scalar_g': update_internal_weights_scalar_g,
            'vector_g': update_internal_weights_vector_g
        }

    def state_update(self, dt):
        """
        Update the state of the generator network.

        Args:
            dt (float): Time step for updating the neuron's state.

        Raises:
            ValueError: If dt is zero.
        """
        if dt == 0:
            raise ValueError("dt cannot be zero.")

        # Update total current and neuron rates
        self.total_current += (dt / self.tau) * (-self.total_current +
                                                 self.J_GG @ self.rate +
                                                 self.J_GI @ self.I)
        self.rate = np.tanh(self.total_current)

        # Update readout neuron activity
        self.Z = self.W.T @ self.rate

    def weight_update(self, target_point):
        err_n = self.Z - target_point
        P_Rate = self.P @ self.rate
        normalizer = 1 / (1 + self.rate.T @ P_Rate)[0, 0]
        delta_W = -normalizer * P_Rate @ err_n.T
        self.P -= normalizer * P_Rate @ P_Rate.T
        self.W += delta_W

        # Update internal weights using the selected method
        self.internal_weight_update_functions[self.internal_weight_update_method](self, delta_W)


class GeneratorNetworkFeedback(GeneratorNetwork):
    """
    Generator Network with feedback from readout neurons to the network.
    """

    def __init__(self,
                 N_network=1000,
                 N_readout=1,
                 g_GG=1.5,
                 g_GZ=1.0,
                 p_GG=0.1,
                 p_z=0.1,
                 tau=10.0,
                 RNG_conn=None,
                 RNG_init=None):

        super().__init__(N_network=N_network,
                         N_readout=N_readout,
                         g_GG=g_GG,
                         p_GG=p_GG,
                         p_z=p_z,
                         tau=tau,
                         RNG_conn=RNG_conn,
                         RNG_init=RNG_init)

        # Random number generators
        if RNG_conn is None:
            RNG_conn = [np.random.default_rng(seed) for seed in range(4)]
        if RNG_init is None:
            RNG_init = np.random.default_rng(0)

        J_Z_raw = 2 * RNG_conn[3].random((self.N_network, N_readout)) - 1
        if np.isscalar(g_GZ):
            self.J_Z = g_GZ * J_Z_raw
        else:
            raise ValueError("g_GZ must be a scalar.")

    def state_update(self, dt):
        """
        Update the state of the feedback network.

        Args:
            dt (float): Time step for updating the neuron's state.

        Raises:
            ValueError: If dt is zero.
        """
        if dt == 0:
            raise ValueError("dt cannot be zero.")

        # Update total current and neuron rates with feedback term included
        self.total_current += (dt / self.tau) * (-self.total_current +
                                                 self.J_GG @ self.rate +
                                                 self.J_Z @ self.Z +
                                                 self.J_GI @ self.I)
        self.rate = np.tanh(self.total_current)

        # Update readout neuron activity
        self.Z = self.W.T @ self.rate

    def weight_update(self, target_point):
        """
        Update weights using FORCE learning for the feedback network.

        Args:
            target_point (numpy.ndarray): Target output for readout neurons.
        """
        err_n = self.Z - target_point
        P_Rate = self.P @ self.rate
        normalizer = 1 / (1 + self.rate.T @ P_Rate)[0, 0]

        # Update inverse correlation matrix P and weights W
        delta_W = -normalizer * P_Rate @ err_n.T
        self.P -= normalizer * P_Rate @ P_Rate.T
        self.W += delta_W

        # Note: In the feedback network, we don't update J_GG
        # The internal connections remain fixed

