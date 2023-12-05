import os
from tqdm import tqdm

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

arr_1 = np.arange(300, 2000, step=200)
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

n_subgraphs = 10
size_subgraphs_divisor = 3

for n_nodes in n_nodes_array:

    # load the results csv (if already existing) to save the variables
    try:
        results_df = pd.read_csv('results/results_dynamic.csv', sep=';', index_col=False)
        ID = max(results_df['ID']) + 1
    except FileNotFoundError:
        ID = 1

    size_subgraphs = int(np.floor(n_nodes / size_subgraphs_divisor))

    GALE_LeiRinaldo_metric_evSC = []
    GALE_LeiRinaldo_metric_SC = []
    GALE_runtimes_evSC = []
    GALE_runtimes_SC = []
    runtime_subgraph_selection = []
    runtime_subgraph_clustering = []

    for _ in tqdm(np.arange(4)):
        ########################################################
        ########## Initiate SBM
        SBM_object = SBM_Online(n_clusters=n_clusters, n_nodes=n_nodes, rho=rho, alpha=alpha, epsilon=epsilon)

        ########################################################
        ########## Initiate subgraphs and GALE for evSC
        Selector_object_evSC = SubgraphSelector_Online(n_nodes=n_nodes, n_subgraphs=n_subgraphs,
                                                       size_subgraphs=size_subgraphs,
                                                       n_clusters=n_clusters,
                                                       parent_alg='evSC', forgetting_factor=forgetting_factor)
        indices = Selector_object_evSC.getIndices()
        GALE_object_evSC = GALE_Online(indices=indices, n_nodes=n_nodes, n_clusters=n_clusters, theta=0.3)

        ########################################################
        ########## Initiate subgraphs and GALE for SC
        Selector_object_SC = SubgraphSelector_Online(n_nodes=n_nodes, n_subgraphs=n_subgraphs,
                                                     size_subgraphs=size_subgraphs,
                                                     n_clusters=n_clusters, indices=indices,
                                                     parent_alg='SC', forgetting_factor=forgetting_factor)
        GALE_object_SC = GALE_Online(indices=indices, n_nodes=n_nodes, n_clusters=n_clusters, theta=0.3)

        ########################################################
        ########## cluster subgraphs and align for every time step
        for t in np.arange(T):
            # SBM
            labels_true, adj = SBM_object.simulate_next()

            # cluster Subgraphs
            labels_subgraphs_evSC = Selector_object_evSC.predict_subgraph_labels(adj)
            labels_subgraphs_SC = Selector_object_SC.predict_subgraph_labels(adj)

            # align subgraphs of first SBM
            membership_estimate_evSC = GALE_object_evSC.performGALE(labels_subgraphs_evSC)
            membership_estimate_SC = GALE_object_SC.performGALE(labels_subgraphs_SC)

        # get values
        SBM_setting = SBM_object.get_values()
        subgraphs_results_evSC = SBM_setting.join(Selector_object_evSC.get_values())
        subgraphs_results_SC = SBM_setting.join(Selector_object_SC.get_values())
        GALE_results_evSC = subgraphs_results_evSC.join(GALE_object_evSC.get_values())
        GALE_results_SC = subgraphs_results_SC.join(GALE_object_SC.get_values())

        # get metric
        GALE_LeiRinaldo_metric_evSC.append(
            LeiRinaldoMetric_1_fromMatrices(membership_estimate_evSC, getMembershipMatrix(labels_true)))
        GALE_LeiRinaldo_metric_SC.append(
            LeiRinaldoMetric_1_fromMatrices(membership_estimate_SC, getMembershipMatrix(labels_true)))

        # get runtime
        GALE_runtimes_evSC.append(GALE_results_evSC['runtime_GALE'])
        GALE_runtimes_SC.append(GALE_results_SC['runtime_GALE'])
        runtime_subgraph_selection.append(GALE_results_evSC['runtime_subgraph_selection'])
        runtime_subgraph_clustering.append(GALE_results_evSC['runtime_subgraph_clustering'])

    # save values for evSC
    GALE_results_evSC['LeiRinaldoMetric'] = np.mean(GALE_LeiRinaldo_metric_evSC)
    GALE_results_evSC['runtime_GALE'] = np.mean(GALE_runtimes_evSC)
    GALE_results_evSC['runtime_subgraph_selection'] = np.mean(runtime_subgraph_selection)
    GALE_results_evSC['runtime_subgraph_clustering'] = np.mean(runtime_subgraph_clustering)

    # save values for SC
    GALE_results_SC['LeiRinaldoMetric'] = np.mean(GALE_LeiRinaldo_metric_SC)
    GALE_results_SC['runtime_GALE'] = np.mean(GALE_runtimes_SC)
    GALE_results_SC['runtime_subgraph_selection'] = np.mean(runtime_subgraph_selection)
    GALE_results_SC['runtime_subgraph_clustering'] = np.mean(runtime_subgraph_clustering)

    try:
        results_df = pd.concat([results_df, GALE_results_evSC, GALE_results_SC], ignore_index=True)
    except NameError:
        results_df = pd.concat([GALE_results_evSC, GALE_results_SC], ignore_index=True)

    results_df.to_csv('results/results_dynamic.csv', sep=';', index=False)
