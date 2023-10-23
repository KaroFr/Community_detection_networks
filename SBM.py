import numpy as np
import pandas as pd
from Helpers import getMembershipMatrix
from numpy.linalg import eigh

"""
Class to simulate the SBM
    1) Connectivity matrix: Planted partition model
    2) Membership matrix: For DSBM simulated by Markov chain
    3) Transition matrix: See Keriven, Vaiter (2022) with parameter epsilon
    4) Initial distribution: uniformly distributed over the K clusters
"""


class SBM:
    SBMs = pd.DataFrame([])
    lambda_min_B_0 = 0

    def __init__(self, n_clusters, n_nodes, rho, alpha, n_time_steps=1, epsilon=0):
        self.K = n_clusters
        self.state_space = np.arange(n_clusters)
        self.n_nodes = n_nodes
        self.rho = rho
        self.alpha = alpha
        if n_time_steps > 1:
            self.dynamic = True
        else:
            self.dynamic = False
        self.T = n_time_steps
        self.eps = epsilon

    """
    get import values as dictionary
    """

    def get_values(self):
        var_df = pd.DataFrame([{'n_clusters': self.K,
                                'n_nodes': self.n_nodes,
                                'rho': self.rho,
                                'alpha': self.alpha,
                                'dynamic': self.dynamic,
                                'n_time_steps': self.T,
                                'epsilon': self.eps,
                                'lambda_min_B_0': self.lambda_min_B_0,
                                }])
        return var_df

    def get_connectivity_matrix(self):
        K = self.K
        rho = self.rho
        alpha = self.alpha
        full_matrix = np.full((K, K), rho)
        diag_matrix = np.diag(np.array([1 - rho] * K))
        B_0 = diag_matrix + full_matrix
        # get minimal eigenvalue of B_0
        evalues, _ = eigh(B_0)
        self.lambda_min_B_0 = min(evalues)
        return alpha * B_0

    def get_transition_matrix(self):
        K = self.K
        eps = self.eps
        transition_matrix = np.full((K, K), (eps / (K - 1)))
        np.fill_diagonal(transition_matrix, (1 - eps))
        return transition_matrix

    def get_initial_distribution(self):
        K = self.K
        return np.full(K, (1 / K))

    def get_initial_states(self):
        n = self.n_nodes
        state_space = self.state_space
        initial_distribution = self.get_initial_distribution()
        return np.random.choice(state_space, size=n, p=initial_distribution)

    def get_T_states(self):
        T = self.T
        state_space = self.state_space
        CurrentState = self.get_initial_states()
        transition_matrix = self.get_transition_matrix()

        # initiate the output
        data = [{
            'labels': CurrentState
        }]
        for _ in np.arange(T - 1):
            NextState = []
            for state in CurrentState:
                next_state = np.random.choice(state_space, p=transition_matrix[state])
                NextState.append(next_state)
            CurrentState = NextState
            data.append(
                {
                    'labels': CurrentState
                }
            )

        # load data into a DataFrame object:
        self.SBMs = pd.DataFrame(data)

    def get_probability_matrices(self):
        B = self.get_connectivity_matrix()
        K = self.K
        n = self.n_nodes

        SBMs = self.SBMs
        probability_matrices = []
        for _, graph in SBMs.iterrows():
            labels = graph['labels']
            memb_matrix = getMembershipMatrix(labels)
            # check if the membership has enough columns
            K_memb = len(memb_matrix[0])
            if K != K_memb:
                diff = K - K_memb
                print('The membership matrix had only ', K_memb, ' clusters. Added zero columns.')
                zero_cols = np.zeros((n, diff))
                memb_matrix = np.hstack((memb_matrix, zero_cols))
            prob_matrix = memb_matrix @ B @ memb_matrix.transpose()
            probability_matrices.append(prob_matrix)

        SBMs["prob_matrix"] = probability_matrices
        self.SBMs = SBMs

    def get_adjacency_matrices(self):
        n = self.n_nodes

        SBMs = self.SBMs
        adj_matrices = []
        for _, graph in SBMs.iterrows():
            prob_matrix = graph['prob_matrix']
            if len(prob_matrix) != n:
                print('The probability matrix has the wrong size: ', len(prob_matrix), ' while n: ', n)
                break
            # construct adjacency matrix
            adj_matrix_full = np.random.binomial(1, prob_matrix, size=(n, n))

            # make it symmetric
            adj_matrix_lower_triangle = np.tril(adj_matrix_full, k=-1)
            adj_matrix = adj_matrix_lower_triangle + adj_matrix_lower_triangle.T
            adj_matrices.append(adj_matrix)

        SBMs["adj_matrix"] = adj_matrices
        self.SBMs = SBMs

    def simulate(self):
        self.get_T_states()
        self.get_probability_matrices()
        self.get_adjacency_matrices()
        return self.SBMs
