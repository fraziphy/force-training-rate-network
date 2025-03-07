# src/force_training_rate_network/training.py
import numpy as np

class FORCETrainer:
    def __init__(self, network, target, stop_period, dt, l_steps):
        self.network = network
        self.target = target
        self.stop_period = stop_period
        self.dt = dt
        self.l_steps = l_steps

        self.period = target.shape[1]
        self.T_prior_learning = 2 * self.period
        self.T_training = stop_period * self.period
        self.T_post_learning = stop_period * self.period
        self.total_sim = self.T_prior_learning + self.T_training + self.T_post_learning

        self.initialize_arrays()

    def initialize_arrays(self):
        self.rate_all = np.empty((4, int(self.total_sim/10)), dtype=float)
        self.Z_all = np.empty((self.network.N_readout, self.total_sim), dtype=float)
        self.W_all = np.empty((3, self.network.N_network, 1), dtype=float)
        self.W_dot = np.empty((self.network.N_readout, int(self.T_training/int(self.l_steps/self.dt))), dtype=float)

        self.J_GG_initial = self.network.J_GG.copy()

    def run_prior_learning(self):
        for i in range(self.T_prior_learning):
            self.update_state(i)

    def run_training(self):
        self.W_all[0] = self.network.W
        for i in range(self.T_training):
            if i % int(self.l_steps/self.dt) == 0:
                self.update_weights(i)
            self.update_state(i + self.T_prior_learning)

    def run_post_learning(self):
        for i in range(self.T_post_learning):
            self.update_state(i + self.T_prior_learning + self.T_training)

    def update_weights(self, i):
        W_dot_aux = self.network.W.copy()
        self.network.Weight_Update(self.target[:, i % self.period])
        self.W_dot[:, i // int(self.l_steps/self.dt)] = np.linalg.norm(self.network.W - W_dot_aux, axis=0)
        if i == 0:
            self.W_all[1] = self.network.W

    def update_state(self, i):
        if i % 10 == 0:
            self.rate_all[:, i // 10] = self.network.rate[:4, 0]
        self.Z_all[:, [i]] = self.network.Z
        self.network.State_Update(self.dt)

    def train(self):
        self.run_prior_learning()
        self.run_training()
        self.W_all[2] = self.network.W
        self.run_post_learning()

    def get_results(self):
        return (self.J_GG_initial, self.network.J_GG, self.rate_all,
                self.Z_all, self.W_all, self.W_dot)

# Usage
# trainer = FORCETrainer(network, target, stop_period, dt, l_steps)
# trainer.train()
# results = trainer.get_results()
