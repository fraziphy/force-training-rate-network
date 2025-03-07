import argparse
from .models import GeneratorNetwork, GeneratorNetworkFeedback
from .simulation import SimulationEngine
import pickle

def main():
    parser = argparse.ArgumentParser(description="Simulate FORCE training networks.")
    parser.add_argument("--network", type=str, choices=["generator_network", "generator_network_feedback"], required=True, help="Type of network to simulate.")
    parser.add_argument("-o", "--output", type=str, default="data.pkl", help="Output spike data file (pickle format)")
    parser.add_argument("--simulation_time", type=int, default=200, help="Simulation time in ms")
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation time step in milliseconds.")
    parser.add_argument("--N_network", type=int, default=1000, help="Number of neurons in the network")
    parser.add_argument("--N_readout", type=int, default=1, help="Number of readout neurons")
    parser.add_argument("--g_GG", type=float, default=1.5, help="Synaptic strength in the generator network")
    parser.add_argument("--p_GG", type=float, default=0.1, help="Connection probability for internal connections")
    parser.add_argument("--p_z", type=float, default=0.1, help="Connection probability for readout connections")
    parser.add_argument("--tau", type=float, default=10.0, help="Time constant for synaptic decay")
    parser.add_argument("--g_GZ", type=float, default=1.0, help="Feedback connection strength (only for feedback network)")

    args = parser.parse_args()

    if args.network == "generator_network":
        net = GeneratorNetwork(N_network=args.N_network, N_readout=args.N_readout, g_GG=args.g_GG, p_GG=args.p_GG, p_z=args.p_z, tau=args.tau)
    else:
        net = GeneratorNetworkFeedback(N_network=args.N_network, N_readout=args.N_readout, g_GG=args.g_GG, g_GZ=args.g_GZ, p_GG=args.p_GG, p_z=args.p_z, tau=args.tau)

    engine = SimulationEngine(net, args.simulation_time, args.dt)
    rates, readouts = engine.run()

    data_to_save = {"rates":rates,
                    "readouts":readouts}

    # Save results
    save_data(data_to_save, args.output)

def save_data(data, filename):
    """Save spike times to a pickle file."""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()
