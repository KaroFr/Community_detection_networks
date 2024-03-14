import os

import numpy as np
import pandas as pd

from ErrorMeasures import LeiRinaldoMetric_1_fromMatrices
from GALE_Online import GALE_Online
from Helpers import getMembershipMatrix
from SBM_Online import SBM_Online
from SubgraphSelector_Online import SubgraphSelector_Online

# make a directory for the upcoming results, if not already existing
if not os.path.isdir('results'):
    os.mkdir('results')

arr_1 = np.arange(100, 2000, step=200)
arr_2 = np.arange(2000, 5000, step=500)
arr_3 = np.arange(5000, 38000, step=1000)
n_nodes_array = np.concatenate([arr_1, arr_2, arr_3])

n_clusters = 5
initial_distribution = []
rho = 0.6
alpha = 0.1
epsilon = 0.3
forgetting_factor = 0.8

T = 10

n_subgraphs = 100
size_subgraphs_divisor = 10

for n_nodes in n_nodes_array:

    # load the results csv (if already existing) to save the variables
    try:
        results_df = pd.read_csv('results/results_csv_large_n.csv', sep=';', index_col=False)
        ID = max(results_df['ID']) + 1
    except FileNotFoundError:
        ID = 1

    size_subgraphs = int(np.floor(n_nodes / size_subgraphs_divisor))

    ########################################################
    ########## Initiate subgraphs, SBM and GALE
    SBM_object = SBM_Online(n_clusters=n_clusters, n_nodes=n_nodes, rho=rho, alpha=alpha, epsilon=epsilon)
    Selector_object = SubgraphSelector_Online(n_nodes=n_nodes, n_subgraphs=n_subgraphs, size_subgraphs=size_subgraphs, n_clusters=n_clusters,
                                              parent_alg='evSC', forgetting_factor=forgetting_factor)
    indices = Selector_object.getIndices()
    GALE_object = GALE_Online(indices=indices, n_nodes=n_nodes, n_clusters=n_clusters, theta=0.3)

    ########################################################
    ########## cluster subgraphs and align for every time step
    for t in np.arange(T):
        # SBM
        labels_true, adj = SBM_object.simulate_next()

        # cluster Subgraphs
        labels_subgraphs = Selector_object.predict_subgraph_labels(adj)

        # align subgraphs of first SBM
        membership_estimate = GALE_object.performGALE(labels_subgraphs)

    SBM_setting = SBM_object.get_values()
    subgraphs_results = SBM_setting.join(Selector_object.get_values())
    GALE_results = subgraphs_results.join(GALE_object.get_values())
    GALE_results['LeiRinaldoMetric'] = LeiRinaldoMetric_1_fromMatrices(membership_estimate, getMembershipMatrix(labels_true))

    try:
        results_df = pd.concat([results_df, GALE_results], ignore_index=True)
    except NameError:
        results_df = pd.concat([GALE_results], ignore_index=True)

    results_df.to_csv('results/results_csv_large_n.csv', sep=';', index=False)