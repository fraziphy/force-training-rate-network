# src/force_training_rate_network/internal_weight_update_functions.py
import numpy as np

def update_internal_weights_standard(network, delta_W):
    network.J_GG += np.ones((network.N_network, network.N_readout)) @ delta_W.T

def update_internal_weights_scalar_g(network, delta_W):
    network.J_GG += network.g_GG * np.ones((network.N_network, network.N_readout)) @ delta_W.T

def update_internal_weights_vector_g(network, delta_W):
    g_matrix = np.repeat(network.g_GG[:, np.newaxis], network.N_readout, axis=1)
    network.J_GG += g_matrix @ delta_W.T
