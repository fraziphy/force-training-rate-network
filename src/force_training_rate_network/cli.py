import argparse
import numpy as np
import pickle
from .models import GeneratorNetwork, GeneratorNetworkFeedback
from .simulation import SimulationEngine
from .training import FORCETrainer

def main():
    parser = argparse.ArgumentParser(description="Simulate or train FORCE networks.")
    parser.add_argument("--network", type=str, choices=["generator_network", "generator_network_feedback"], required=True, help="Type of network to simulate.")
    parser.add_argument("--mode", type=str, choices=["spontaneous", "force_training"], required=True, help="Mode: spontaneous activity or FORCE training.")
    parser.add_argument("-o", "--output", type=str, default="data.pkl", help="Output data file (pickle format).")
    parser.add_argument("--simulation_time", type=int, default=200, help="Simulation time in ms (for spontaneous mode).")
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation time step in milliseconds.")
    parser.add_argument("--N_network", type=int, default=1000, help="Number of neurons in the network.")
    parser.add_argument("--N_readout", type=int, default=1, help="Number of readout neurons.")
    parser.add_argument("--g_GG", type=float, default=1.5, help="Synaptic strength in the generator network.")
    parser.add_argument("--p_GG", type=float, default=0.1, help="Connection probability for internal connections.")
    parser.add_argument("--p_z", type=float, default=0.1, help="Connection probability for readout connections.")
    parser.add_argument("--tau", type=float, default=10.0, help="Time constant for synaptic decay.")
    parser.add_argument("--g_GZ", type=float, default=1.0, help="Feedback connection strength (only for feedback network).")

    # Arguments specific to FORCE training
    parser.add_argument("--target_signal", type=str, help="Path to target signal file (for FORCE training).")
    parser.add_argument("--training_periods", type=int, default=10, help="Number of periods for training (for FORCE training).")
    parser.add_argument("--update_step", type=int, default=2, help="Step in ms at which weights are updated (for FORCE training).")

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
        engine = SimulationEngine(net)
        rates, readouts = engine.run_spontaneous(args.simulation_time)
        save_data({"rates": rates, "readouts": readouts}, args.output)

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
        save_data({"results": results}, args.output)

def save_data(data, filename):
    """Save data to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()
