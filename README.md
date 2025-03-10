# _**force-training-rate-network**_
=====================================

## Overview
---------------

The `force-training-rate-network` module implements the FORCE (First-Order Reduced and Controlled Error) training algorithm for recurrent neural networks. This method enables networks to learn complex dynamical tasks by leveraging high-dimensional chaotic dynamics and supervised learning through Recursive Least Squares (RLS). The FORCE method is particularly useful for training rate-based neural networks and has applications in neuroscience, machine learning, and control systems.

This implementation is inspired by the seminal work of Abbott and Sussillo, 2009, [doi:10.1016/j.neuron.2009.07.018](https://doi.org/10.1016/j.neuron.2009.07.018),, which introduced FORCE training as a robust method for learning in recurrent networks.

### References
- Sussillo, D., & Abbott, L. F. (2009). Generating coherent patterns of activity from chaotic neural networks. *Neuron*, 63(4), 544–557.
- Sussillo, D., & Abbott, L. F. (2012). Transferring learning from rate to spiking networks. *Nature Neuroscience*, 15(4), 478–486.
=====================================

## Methodology
-------------

The FORCE training algorithm utilizes the following key principles:
1. **High-Dimensional Chaotic Dynamics**: The network starts with random connectivity that induces chaotic activity, providing a rich reservoir of dynamics.
2. **Error-Driven Learning**: A supervisor provides an error signal to guide the network toward desired outputs.
3. **Recursive Least Squares (RLS)**: The RLS algorithm dynamically adjusts the output weights to minimize the error between the network's output and the target signal.

This implementation supports:
- Training rate-based recurrent networks using FORCE.
- Simulation of learned dynamics.
- Analysis of Lyapunov exponents to study stability.

For further details on the methodology, refer to Abbott and Sussillo's original papers.
=====================================

## Installation
--------------

To install the `force-training-rate-network` module:

1. Clone the repository:

```
git clone https://github.com/fraziphy/force-training-rate-network.git
cd force-training-rate-network
```

2. Install the package using pip:

```
pip install .
```

Alternatively, install directly from GitHub:

```
pip install git+https://github.com/fraziphy/force-training-rate-network.git@v1.0.0
```

To uninstall the module:

```
pip uninstall force-training-rate-network.git@v1.0.0
```
=====================================

## Usage
-----

There are two primary ways to use the `force-training-rate-network` module:

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

## Example Notebook

A Jupyter notebook has been included to demonstrate how to use the _**LinearDecoder**_ class with dummy data. The notebook verifies that the defined class can perform decoding tasks effectively.
=====================================

## The structure of the project is as follows:
```
force-training-rate-network/
├── LICENSE
├── README.md
├── CHANGELOG.md
├── pyproject.toml
├── src/
│   └── force_training_rate_network/
│       ├── __init__.py
│       ├── models.py
│       ├── internal_weight_update_functions.py
│       ├── simulation.py
│       ├── lyapunov.py
│       ├── training.py
│       ├──utils.py
│       └── subpackage/
│           ├── __init__.py
│           └── submodule.py
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   └── test_module2.py
└── docs/
    └── index.md



```

- src: This directory holds the directory that contains the _**linear_decoder**_ module.
    -lineardecoder: This directory holds the core functionality of the _**linear_decoder**_ module.
    	- __init__.py: Marks the directory as a Python package.
    	- linear_decoder.py: Includes functions and classes defining the main functionalities of the _**linear_decoder**_ module.

- test_script: This directory contains unit tests for the _**linear_decoder**_ module.
    - test_dependencies.py: Unit tests for the functions module.
   
- examples: This directory contains the directories of notebooks and scripts, which are defined for manifesting example usage of the module.
    - notebook: This directory contains the Jupyter notebook for exploratory analysis and demonstrations related to the _**linear_decoder**_ module.
   	 - linear_decoder.ipynb: Jupyter notebook for _**linear_decoder**_ module usage and demonstrations.
	 
    - scripts: This directory contains Python scripts for generating dummy data and plotting the raster plot in the linear_decoder.ipynb.
        - generate_data.py: Script for generating dummy data.
        - plots.py: Script for plotting figures.

- LICENSE: The license file for the project.

- README.md: The README file providing an overview of the project, its purpose, and how to use it.

- setup.py: The setup script for installing the _**linear_decoder**_ module as a Python package.

=====================================

## Contributing

Thank you for considering contributing to our project! We welcome contributions from the community to help improve our project and make it even better. To ensure a smooth contribution process, please follow these guidelines:

1. **Fork the Repository**: Fork our repository to your GitHub account and clone it to your local machine.

2. **Branching Strategy**: Create a new branch for your contribution. Use a descriptive branch name that reflects the purpose of your changes.

3. **Code Style**: Follow our coding standards and style guidelines. Make sure your code adheres to the existing conventions to maintain consistency across the project.

4. **Pull Request Process**:
    Before starting work, check the issue tracker to see if your contribution aligns with any existing issues or feature requests.
    Create a new branch for your contribution and make your changes.
    Commit your changes with clear and descriptive messages explaining the purpose of each commit.
    Once you are ready to submit your changes, push your branch to your forked repository.
    Submit a pull request to the main repository's develop branch. Provide a detailed description of your changes and reference any relevant issues or pull requests.

5. **Code Review**: Expect feedback and review from our maintainers or contributors. Address any comments or suggestions provided during the review process.

6. **Testing**: Ensure that your contribution is properly tested. Write unit tests or integration tests as necessary to validate your changes. Make sure all tests pass before submitting your pull request.

7. **Documentation**: Update the project's documentation to reflect your changes. Include any necessary documentation updates, such as code comments, README modifications, or user guides.

8. **License Agreement**: By contributing to our project, you agree to license your contributions under the terms of the project's license (GNU General Public License v3.0).

9. **Be Respectful**: Respect the opinions and efforts of other contributors. Maintain a positive and collaborative attitude throughout the contribution process.

We appreciate your contributions and look forward to working with you to improve our project! If you have any questions or need further assistance, please don't hesitate to reach out to us.

=====================================

## Credits

- **Author:** [Farhad Razi](https://github.com/fraziphy)

=====================================

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE)

=====================================

## Contact

- **Contact information:** [email](farhad.razi.1988@gmail.com)

=====================================

## Acknowledgments

This work was supported by the Dutch Research Council (NWO Vidi grant VI.Vidi.213.137).

=====================================
# _**force-training-rate-network**_