import argparse
import numpy as np
import pickle
import os
from .models import GeneratorNetwork, GeneratorNetworkFeedback
from .simulation import SimulationEngine
from .training import FORCETrainer
from .lyapunov import LyapunovExponentCalculator

def main():
    parser = argparse.ArgumentParser(description="Simulate, train, or compute Lyapunov exponent for FORCE networks.")
    parser.add_argument("--network", type=str, choices=["generator_network", "generator_network_feedback"], required=True, help="Type of network to simulate.")
    parser.add_argument("--mode", type=str, choices=["spontaneous", "force_training", "lyapunov"], required=True, help="Mode: spontaneous activity, FORCE training, or Lyapunov exponent computation.")
    parser.add_argument("-o", "--output", type=str, default="data.pkl", help="Output data file (pickle format).")
    parser.add_argument("--simulation_time", type=int, default=200, help="Simulation time in ms (for spontaneous mode or Lyapunov mode).")
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation time step in milliseconds.")
    parser.add_argument("--N_network", type=int, default=1000, help="Number of neurons in the network.")
    parser.add_argument("--N_readout", type=int, default=1, help="Number of readout neurons.")
    parser.add_argument("--g_GG", type=float, default=1.5, help="Synaptic strength in the generator network.")
    parser.add_argument("--p_GG", type=float, default=0.1, help="Connection probability for internal connections.")
    parser.add_argument("--p_z", type=float, default=0.1, help="Connection probability for readout connections.")
    parser.add_argument("--tau", type=float, default=10.0, help="Time constant for synaptic decay.")
    parser.add_argument("--g_GZ", type=float, default=1.0, help="Feedback connection strength (only for feedback network).")

    # Arguments specific to FORCE training
    parser.add_argument(
                "--target_signal",
                type=str,
                default=os.path.join(os.path.dirname(__file__), "../sample_target_signal/target.npy"),
                help="Path to target signal file (for FORCE training). Default: src/sample_target_signal/target.npy")
    parser.add_argument("--training_periods", type=int, default=10, help="Number of periods for training (for FORCE training).")
    parser.add_argument("--update_step", type=int, default=2, help="Step in ms at which weights are updated (for FORCE training).")

    # Arguments specific to Lyapunov computation
    parser.add_argument("--renorm_interval", type=float, default=50.0, help="Renormalization interval in ms (for Lyapunov mode).")
    parser.add_argument("--delta_separation", type=float, default=1e-10, help="Initial infinitesimal separation between trajectories (for Lyapunov mode).")
    parser.add_argument("--store_trajectories", action="store_true", help="Store trajectories of original and perturbed networks (for Lyapunov mode).")

    args = parser.parse_args()

    # Initialize network
    if args.network == "generator_network":
        net = GeneratorNetwork(N_network=args.N_network,
                               N_readout=args.N_readout,
                               g_GG=args.g_GG,
                               p_GG=args.p_GG,
                               p_z=args.p_z,
                               tau=args.tau)
    else:
        net = GeneratorNetworkFeedback(N_network=args.N_network,
                                       N_readout=args.N_readout,
                                       g_GG=args.g_GG,
                                       g_GZ=args.g_GZ,
                                       p_GG=args.p_GG,
                                       p_z=args.p_z,
                                       tau=args.tau)

    if args.mode == "spontaneous":
        # Spontaneous activity mode
        engine = SimulationEngine(net, args.simulation_time)
        results = engine.run()
        save_data({
            "rates": results[0],
            "readouts": results[1]
        }, args.output)

    elif args.mode == "lyapunov":
        # Lyapunov exponent computation mode
        calculator = LyapunovExponentCalculator(
            network=net,
            simulation_time=args.simulation_time,
            dt=args.dt,
            renorm_interval=args.renorm_interval,
            delta_separation=args.delta_separation,
            store_trajectories=args.store_trajectories
        )

        # Initialize perturbation and compute Lyapunov exponent
        calculator.initialize_perturbation()
        calculator.compute()

        # Get results and save them
        results = calculator.get_results()

        save_data({
            "lyapunov_exponent": results[0],
            "diff_over_time": results[1],
            "rates_original": results[2],
            "rates_neighbour": results[3],
            "true_diff_over_time": results[4]
        }, args.output)


    elif args.mode == "force_training":
        # FORCE training mode
        if not args.target_signal:
            raise ValueError("Target signal file must be provided for FORCE training mode.")

        # Load target signal
        target_signal = np.load(args.target_signal)

        # Initialize trainer
        trainer = FORCETrainer(network=net,
                               target=target_signal,
                               stop_period=args.training_periods,
                               dt=args.dt,
                               l_steps=args.update_step)

        # Train network
        trainer.train()

        # Save results
        results = trainer.get_results()
        save_data({
            "J_GG_initial": results[0],
            "J_GG_final": results[1],
            "rate_all": results[2],
            "Z_all": results[3],
            "W_all": results[4],
            "W_dot": results[5]
        }, args.output)


def save_data(data, filename):
    """Save data to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()
