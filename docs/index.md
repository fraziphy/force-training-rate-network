# _**force_training_rate_network**_
=====================================

## Overview
-----------------------------------------------------------------

Welcome to the comprehensive documentation for the `force-training-rate-network` module. This module implements the FORCE (First-Order Reduced and Controlled Error) training algorithm for recurrent neural networks. This method enables networks to learn complex dynamical tasks.

### Methodology
-----------------------------------------------------------------

The FORCE training algorithm utilizes the following key principles:

1.  **High-Dimensional Chaotic Dynamics**: The network starts with random connectivity that induces chaotic activity, providing a rich reservoir of dynamics.
2.  **Error-Driven Learning**: A supervisor provides an error signal to guide the network toward desired outputs.
3.  **Recursive Least Squares (RLS)**: The RLS algorithm dynamically adjusts the output weights to minimize the error between the network's output and the target signal.

This implementation supports:

*   Training rate-based recurrent networks using FORCE.
*   Simulation of learned dynamics.
*   Analysis of Lyapunov exponents to study stability.

For further details on the methodology, refer to Abbott and Sussillo's original papers.

### References

*   Sussillo, D., & Abbott, L. F. (2009). Generating coherent patterns of activity from chaotic neural networks. *Neuron*, 63(4), 544–557.
*   Sussillo, D., & Abbott, L. F. (2012). Transferring learning from rate to spiking networks. *Nature Neuroscience*, 15(4), 478–486.
=====================================

## Usage
-----------------------------------------------------------------

There are two primary ways to use the `force_training_rate_network` module:

### 1. Direct Import in Python Scripts or Jupyter Notebooks

You can import and use the module directly in your Python scripts or Jupyter notebooks for more flexible and interactive use.

#### Example:  Simulating and Training a Rate Neural Network

```
from force_training_rate_network.models import GeneratorNetwork
from force_training_rate_network.training import FORCETrainer
from force_training_rate_network.simulation import SimulationEngine
from force_training_rate_network.lyapunov import LyapunovExponentCalculator
import numpy as np
```

Initialize a generator network:

```
net = GeneratorNetwork(
			      N_network,
                               N_readout,
                               g_GG,
                               p_GG,
                               p_z,
                               tau)
```

Or,

```
net = GeneratorNetworkFeedback(
			        N_network,
                                 N_readout,
                                 g_GG,
                                 g_GZ,
                                 p_GG,
                                 p_z,
                                 tau)
```

To simulate the spontaneous activity of the network

```
engine = SimulationEngine(net, simulation_time, dt)
rates, readouts= engine.run()
```

To compute the Maximal Lyapunov Exponent (MLE)

```
calculator = LyapunovExponentCalculator(
            net,
            simulation_time,
            dt,
            renorm_interval,
            delta_separation,
            store_trajectories=True
        )

# Initialize perturbation and compute Lyapunov exponent
calculator.initialize_perturbation()
calculator.compute()

# Get results
lyapunov_exponent, diff_over_time, rates_original, rates_neighbour, true_diff_over_time = calculator.get_results()
```

To train the network using FORCE

```
target_signal = np.load(/path/to/the/np.array)

# Initialize trainer
trainer = FORCETrainer(net,
                        target_signal,
                        training_periods,
                        dt,
                        update_step)

# Train network
trainer.train()

# Get results

J_GG_initial, J_GG_final, rate_all, Z_all, W_all, W_dot = trainer.get_results()
```

### 2. Command Line Interface (CLI)

The module also provides a command-line interface for easy execution of common tasks.

To simulate spontaneous activity of a network:

```
force-training-rate-network --network generator_network --mode spontaneous --simulation_time 500 --output spontaneous.pkl
```

To compute MLE:

```
force-training-rate-network --network generator_network --mode lyapunov --simulation_time 30000 --renorm_interval 50 --delta_separation 1e-10 --store_trajectories --output lyapunov.pkl
```

To perform FORCE training on the network:

```
force-training-rate-network --network generator_network_feedback --mode force_training --target_signal target.npy --training_periods 10 --update_step 5 --output training.pkl
```
=====================================

## Accessing Help
-----------------------------------------------------------------

You can access the guide within your python scripts or Jupyter Notebooks by using:
```
net = GeneratorNetwork()
net.help

engine = SimulationEngine(net)
engine.help

calculator = LyapunovExponentCalculator(net)
calculator.help

trainer = FORCETrainer(net)
trainer.help
```
These will print the usage instructions directly in your Python environment.

Additionally, you can access the guid for the command-line interface using:

```
force-training-rate-network --extended-help
```
=====================================

## Module Structure
-----------------------------------------------------------------

This section provides an overview of the directory structure and contents of the **force_training_rate_network** module. The structure of the project is as follows:
```
force-training-rate-network/
├── src/
│   ├── force_training_rate_network/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── internal_weight_update_functions.py
│   │   ├── simulation.py
│   │   ├── lyapunov.py
│   │   ├── training.py
│   │   └── cli.py
│   └── sample_target_signal/
│       └── target.npy
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_simulation.py
│   ├── test_lyapunov.py
│   └── test_training.py
├── docs/
│   └── index.md
├── README.md
├── CHANGELOG.md
├── LICENSE
└── pyproject.toml
```

-**src/**: This directory holds the core functionality of the **force_training_rate_network** module.

    - **force_training_rate_network/**: This directory contains all the main classes and functions.
        - **__init__.py**: Marks the directory as a Python package and may contain package-level imports.
        - **models.py**: Defines the neural network models used in the project, such as `GeneratorNetwork` and `GeneratorNetworkFeedback`.
        - **internal_weight_update_functions.py**: Contains functions for updating internal weights of the networks during training.
        - **simulation.py**: Implements the `SimulationEngine` class for running network simulations.
        - **lyapunov.py**: Provides functionality for calculating Lyapunov exponents to analyze network stability.
        - **training.py**: Implements the `FORCETrainer` class for training networks using the FORCE algorithm.
        - **cli.py**: Defines the command-line interface for interacting with the module's functionality.
        
    - **sample_target_signal/**: This directory provides a default target signal (`target.npy`) for use with CLI-based FORCE training.  
        - **target.npy**: A sample target signal file that ensures CLI functionality even if no custom target signal is provided by the user.

-**tests/**: This directory contains unit tests for the **force_training_rate_network** module.
    - **__init__.py**: Marks the directory as a Python package, allowing for test discovery.
    - **test_models.py**: Contains unit tests for the network models defined in `models.py`.
    - **test_simulation.py**: Contains unit tests for the `SimulationEngine` class in `simulation.py`.
    - **test_lyapunov.py**: Contains unit tests for the Lyapunov exponent calculations in `lyapunov.py`.
    - **test_training.py**: Contains unit tests for the `FORCETrainer` class in `training.py`.

-**docs/**: This directory is for documentation files.

    - **index.md**: The main entry point for the module's documentation, providing an overview and links to other documentation sections.

- **README.md**: This file provides an overview of the project, its purpose, installation instructions, and basic usage examples.
- **CHANGELOG.md**: Documents the version history of the project, listing notable changes, additions, and fixes for each release.
- **LICENSE**: The license file specifying the terms under which the project can be used, modified, and distributed.
- **pyproject.toml**: The configuration file for building the project, managing dependencies, and setting up tools like pytest and black.
=====================================

## Development and Testing
-----------------------------------------------------------------

### Running Tests

To run the unit tests, execute the following command from the project root directory:

```
python -m unittest discover -s tests -v
```

### Developing the CLI

When developing the `cli.py` file, you can test it by running commands from the `src` directory. Here are some example commands:

1. To simulate spontaneous activity of a network:

```
python -m force_training_rate_network.cli --network generator_network --mode spontaneous --simulation_time 500 --output spontaneous.pkl
```

2. To compute Maximum Lyapunov Exponent (MLE):

```
python -m force_training_rate_network.cli --network generator_network --mode lyapunov --simulation_time 30000 --renorm_interval 50 --delta_separation 1e-10 --store_trajectories --output lyapunov.pkl
```

3. To perform FORCE training on the network:

```
python -m force_training_rate_network.cli --network generator_network_feedback --mode force_training --target_signal target.npy --training_periods 10 --update_step 5 --output training.pkl
```

These commands allow you to test different functionalities of the CLI during development.
=====================================

## Contributing
-----------------------------------------------------------------

Thank you for considering contributing to our project! We welcome contributions from the community to help improve our project and make it even better. To ensure a smooth contribution process, please follow these guidelines:

1.  **Fork the Repository**: Fork our repository to your GitHub account and clone it to your local machine.
2.  **Branching Strategy**: Create a new branch for your contribution. Use a descriptive branch name that reflects the purpose of your changes.
3.  **Code Style**: Follow our coding standards and style guidelines. Make sure your code adheres to the existing conventions to maintain consistency across the project.

### Pull Request Process

1.  Before starting work, check the issue tracker to see if your contribution aligns with any existing issues or feature requests.
2.  Create a new branch for your contribution and make your changes.
3.  Commit your changes with clear and descriptive messages explaining the purpose of each commit.
4.  Once you are ready to submit your changes, push your branch to your forked repository.
5.  Submit a pull request to the main repository's `develop` branch. Provide a detailed description of your changes and reference any relevant issues or pull requests.

*   **Code Review**: Expect feedback and review from our maintainers or contributors. Address any comments or suggestions provided during the review process.
*   **Testing**: Ensure that your contribution is properly tested. Write unit tests or integration tests as necessary to validate your changes. Make sure all tests pass before submitting your pull request.
*   **Documentation**: Update the project's documentation to reflect your changes. Include any necessary documentation updates, such as code comments, README modifications, or user guides.
*   **License Agreement**: By contributing to our project, you agree to license your contributions under the terms of the project's license (GNU General Public License v3.0).
*   **Be Respectful**: Respect the opinions and efforts of other contributors. Maintain a positive and collaborative attitude throughout the contribution process.

We appreciate your contributions and look forward to working with you to improve our project! If you have any questions or need further assistance, please don't hesitate to reach out to us.

=====================================

## Credits
-----------------------------------------------------------------

- **Author:** [Farhad Razi](https://github.com/fraziphy)

=====================================

## License
-----------------------------------------------------------------

This project is licensed under the [GNU General Public License v3.0](LICENSE)

=====================================

## Contact
-----------------------------------------------------------------

- **Contact information:** [email](farhad.razi.1988@gmail.com)

=====================================

## Acknowledgments
-----------------------------------------------------------------

This work was supported by the Dutch Research Council (NWO Vidi grant VI.Vidi.213.137) awarded to Dr. [Fleur Zeldenrust](https://fleurzeldenrust.nl/).

=====================================
# _**force_training_rate_network**_