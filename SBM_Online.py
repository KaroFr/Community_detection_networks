import numpy as np
import pandas as pd
from Helpers import getMembershipMatrix
# from numpy.linalg import eigh

"""
Class to simulate the SBM
    1) Connectivity matrix: Planted partition model
    2) Membership matrix: For DSBM simulated by Markov chain
    3) Transition matrix: See Keriven, Vaiter (2022) with parameter kappa (=epsilon in their work)
    4) Initial distribution: uniformly distributed over all clusters
"""


class SBM_Online:
    probability_matrix = []
    current_states = []

    # lambda_min_B_0 = 0
    # n_min = 0
    # n_max = 0

    def __init__(self, n_clusters, n_nodes, rho, alpha, kappa=0, initial_distribution=None):
        """
        :param n_clusters: number of clusters (k)
        :param n_nodes: number of nodes (n)
        :param rho: parameter of the planted partition model;
                    prob. matrix B_0 has 1 on diagonal and rho for every other entry
        :param alpha: sparsity parameter; B = alpha*B_0
        :param kappa: parameter of the transition matrix;
                        (1-kappa) on diagonal and kappa/(n_clusters-1) for every other entry
        :param initial_distribution: defines how the nodes are distributed among the clusters
        """
        self.time_step = 0
        self.n_clusters = n_clusters
        self.state_space = np.arange(n_clusters)
        self.n_nodes = n_nodes
        self.rho = rho
        self.alpha = alpha
        self.kappa = kappa
        if initial_distribution is None:
            self.initial_distribution = self.get_initial_distribution()
        else:
            self.initial_distribution = initial_distribution
        self.connectivity_matrix = self.get_connectivity_matrix()
        self.transition_matrix = self.get_transition_matrix()

    def get_values(self):
        """
        get import values as dictionary
        """
        var_df = pd.DataFrame([{'n_clusters': self.n_clusters,
                                'n_nodes': self.n_nodes,
                                'rho': self.rho,
                                'alpha': self.alpha,
                                'kappa': self.kappa,
                                # 'lambda_min_B_0': self.lambda_min_B_0,
                                # 'n_min': self.n_min,
                                # 'n_max': self.n_max,
                                'time_step_SBM': self.time_step,
                                }])
        return var_df

    def get_initial_distribution(self):
        """
        if no initial distribution is given, use uniform distribution over all clusters
        :return: initial uniform distribution
        """
        n_clusters = self.n_clusters
        return np.full(n_clusters, (1 / n_clusters))

    def get_connectivity_matrix(self):
        """
        Construct the connectivity matrix from the input parameters rho, alpha and n_clusters.
        Is fixed for all time steps.
        :return: the connectivity matrix of the DSBM
        """
        n_clusters = self.n_clusters
        rho = self.rho
        alpha = self.alpha
        full_matrix = np.full((n_clusters, n_clusters), rho)
        diag_matrix = np.diag(np.array([1 - rho] * n_clusters))
        B_0 = diag_matrix + full_matrix
        # get minimal eigenvalue of B_0
        # evalues, _ = eigh(B_0)
        # self.lambda_min_B_0 = min(evalues)
        return alpha * B_0

    def get_transition_matrix(self):
        """
        Construct the transition matrix from the input parameters kappa and n_clusters.
        Is fixed for all time steps.
        :return: the transition matrix of the DSBM
        """
        n_clusters = self.n_clusters
        kappa = self.kappa
        transition_matrix = np.full((n_clusters, n_clusters), (kappa / (n_clusters - 1)))
        np.fill_diagonal(transition_matrix, (1 - kappa))
        return transition_matrix

    def get_initial_states(self):
        """
        Randomly generate initial states (membership to one of the clusters) for all the nodes
        Safe the initial states as 'current_states' which will be updated for every timestep
        """
        n = self.n_nodes
        state_space = self.state_space
        initial_distribution = self.initial_distribution
        initial_states = np.random.choice(state_space, size=n, p=initial_distribution)
        # self.n_min = min(np.bincount(initial_states))
        # self.n_max = max(np.bincount(initial_states))
        self.current_states = initial_states

    def get_next_states(self):
        """
        Randomly generate next states (membership to one of the clusters) for all the nodes
        Use 'current_states' and update them for this timestep
        """
        state_space = self.state_space
        current_states = self.current_states
        transition_matrix = self.transition_matrix

        next_states = []
        for state in current_states:
            next_state = np.random.choice(state_space, p=transition_matrix[state])
            next_states.append(next_state)

        # self.n_min = min(np.bincount(next_states))
        # self.n_max = max(np.bincount(next_states))
        self.current_states = next_states

    def get_probability_matrix(self):
        """
        Calculate the probability matrix of the DSBM (at the current time step)
        P = Theta B Theta^T (Theta = membership matrix; B = connectivity matrix)
        :return: probability matrix
        """
        B = self.connectivity_matrix
        n_clusters = self.n_clusters
        n = self.n_nodes

        current_states = self.current_states
        membership_matrix = getMembershipMatrix(current_states)
        # check if the membership has enough columns
        K_memb = len(membership_matrix[0])
        if n_clusters != K_memb:
            diff = n_clusters - K_memb
            print('The membership matrix had only ', K_memb, ' clusters. Added zero columns.')
            zero_cols = np.zeros((n, diff))
            membership_matrix = np.hstack((membership_matrix, zero_cols))
        probability_matrix = membership_matrix @ B @ membership_matrix.transpose()
        return probability_matrix

    def get_adjacency_matrix(self):
        """
        Calculate the adjacency matrix of the DSBM at the current time step
        :return: adjacency matrix
        """
        n = self.n_nodes

        probability_matrix = self.get_probability_matrix()

        # construct adjacency matrix
        adj_matrix_full = np.random.binomial(1, probability_matrix, size=(n, n))

        # make it symmetric
        adj_matrix_lower_triangle = np.tril(adj_matrix_full, k=-1)
        adj_matrix = adj_matrix_lower_triangle + adj_matrix_lower_triangle.T
        return adj_matrix

    def simulate_next(self):
        """
        After initiating an object of class 'SBM_Online', this is used to simulate the first and any further SBM.
        :return: current_states (memberships) and adjacency matrix of the next time step
        """
        # time_step counts at what time step we are
        time_step = self.time_step
        if time_step == 0:
            self.get_initial_states()
        else:
            self.get_next_states()
        time_step += 1
        self.time_step = time_step
        adj_matrix = self.get_adjacency_matrix()
        current_states = self.current_states
        print('Simulated the SBM for time step t=', time_step)
        return current_states, adj_matrix
