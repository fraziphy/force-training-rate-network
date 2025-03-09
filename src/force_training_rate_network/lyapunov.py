# src/force_training_rate_network/lyapunov.py
import numpy as np
import copy

class LyapunovExponentCalculator:
    """
    A class to compute the maximal Lyapunov exponent for a given network.
    This class simulates two trajectories of the network with an infinitesimal
    difference in their initial conditions and evaluates their divergence over time.

    Attributes:
        network (object): The original network under study.
        simulation_time (float): Total simulation time in milliseconds.
        dt (float): Simulation time step in milliseconds.
        renorm_interval (float): Interval (in ms) at which renormalization occurs.
        delta_separation (float): Initial infinitesimal separation between trajectories.
        store_trajectories (bool): Whether to store the rates of the original and perturbed networks.
    """

    def __init__(self, network, simulation_time=30000, dt=0.1, renorm_interval=50, delta_separation=1e-10, store_trajectories=False):
        self.network = network
        self.simulation_time = simulation_time
        self.dt = dt
        self.renorm_interval = renorm_interval
        self.delta_separation = delta_separation
        self.store_trajectories = store_trajectories

        # Derived parameters
        self.num_renorms = int(simulation_time / renorm_interval)
        self.steps_per_renorm = int(renorm_interval / dt)
        self.scale_to_seconds = 1000 / renorm_interval

        # Initialize copies of the network for perturbed trajectories
        self.network_perturbed = copy.deepcopy(network)

        if store_trajectories:
            # Only create a third trajectory if storing trajectories is enabled
            self.network_true_diff = copy.deepcopy(network)

        # Initialize arrays for results
        self.lyapunov_exponents = np.empty(self.num_renorms, dtype=float)
        self.diff_over_time = np.empty(int(simulation_time / dt), dtype=float)

        if store_trajectories:
            self.true_diff_over_time = np.empty(int(simulation_time / dt), dtype=float)
            # Example rates for identical neurons in both trajectories
            self.rates_original = []
            self.rates_perturbed = []

    def initialize_perturbation(self):
        """
        Initialize the perturbed trajectory with an infinitesimal difference.
        """
        direction = (self.network_perturbed.rate + self.delta_separation - self.network.rate)
        direction /= np.linalg.norm(direction)

        # Apply perturbation to the perturbed network
        self.network_perturbed.rate += self.delta_separation * direction
        self.network_perturbed.Z = self.network_perturbed.W.T @ self.network_perturbed.rate

        if self.store_trajectories:
            # Apply perturbation to the true difference network if storing trajectories
            self.network_true_diff.rate += self.delta_separation * direction
            self.network_true_diff.Z = self.network_true_diff.W.T @ self.network_true_diff.rate

    def compute(self):
        """
        Compute the maximal Lyapunov exponent by evolving two trajectories over time,
        periodically renormalizing their separation.

        Returns:
            dict: A dictionary containing results such as Lyapunov exponent,
                  trajectory differences, and optionally stored rates.
        """

        # Initialize perturbation
        diff_aux = self.network_perturbed.rate - self.network.rate
        separation = np.linalg.norm(diff_aux)

        direction = diff_aux / separation

        for i in range(self.num_renorms):
            # Evolve both trajectories for one renormalization interval
            for j in range(self.steps_per_renorm):
                index = i * self.steps_per_renorm + j

                # Store differences over time
                self.diff_over_time[index] = np.linalg.norm(diff_aux)

                if self.store_trajectories:
                    # Only compute true differences if storing trajectories
                    true_diff_aux = np.linalg.norm(self.network_true_diff.rate - self.network.rate)
                    self.true_diff_over_time[index] = true_diff_aux

                # Update states of both networks
                self.network.state_update(self.dt)
                self.network_perturbed.state_update(self.dt)

                if self.store_trajectories:
                    # Update state of the true difference network only if needed
                    self.network_true_diff.state_update(self.dt)

                diff_aux = self.network_perturbed.rate - self.network.rate

                if self.store_trajectories and index % 10 == 0:
                    # Store example rates for identical neurons
                    rate_original_example = copy.deepcopy(self.network.rate[:4])
                    rate_neighbour_example = copy.deepcopy(self.network_true_diff.rate[:4])

                    # Append rates to storage arrays
                    self.rates_original.append(rate_original_example)
                    self.rates_perturbed.append(rate_neighbour_example)

            # Calculate new separation after renormalization interval
            new_separation = np.linalg.norm(diff_aux)

            # Accumulate the log of the separation ratio for Lyapunov exponent calculation
            self.lyapunov_exponents[i] = np.log(new_separation / separation)

            # Renormalize the perturbed trajectory to keep it close to the original trajectory
            new_direction = diff_aux / new_separation
            self.network_perturbed.rate = self.network.rate + self.delta_separation * new_direction

            diff_aux = self.network_perturbed.rate - self.network.rate
            separation = np.linalg.norm(diff_aux)

            direction = new_direction


    def get_results(self):
        """
        Return computed results as a tuple of numpy arrays.

        Returns:
            tuple: A tuple containing:
                - lyapunov_exponent (float): The computed Lyapunov exponent.
                - diff_over_time (np.ndarray): Differences over time.
                - rates_original (np.ndarray or None): Original rates if stored, else None.
                - rates_neighbour (np.ndarray or None): Perturbed rates if stored, else None.
                - true_diff_over_time (np.ndarray or None): True differences if stored, else None.
        """
        lyapunov_exponent = np.mean(self.lyapunov_exponents) * self.scale_to_seconds

        # Convert rates_original and rates_perturbed to numpy arrays if they exist
        rates_original = (
            np.array(self.rates_original).T.reshape(4,-1) if hasattr(self, "rates_original") and self.rates_original else None
        )
        rates_perturbed = (
            np.array(self.rates_perturbed).T.reshape(4,-1) if hasattr(self, "rates_perturbed") and self.rates_perturbed else None
        )

        return (
            lyapunov_exponent,
            np.array(self.diff_over_time),
            rates_original,
            rates_perturbed,
            np.array(getattr(self, "true_diff_over_time", []))  # Convert true_diff_over_time to np.array
        )


