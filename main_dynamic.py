import os
from tqdm import tqdm

import numpy as np
import pandas as pd

from ErrorMeasures import LeiRinaldoMetric_1_fromMatrices, LeiRinaldoMetric_1_fromLabels
from GALE_Online import GALE_Online
from Helpers import getMembershipMatrix
from SBM_Online import SBM_Online
from SpectralClustering import SpectralClustering
from SubgraphSelector_Online import SubgraphSelector_Online

# make a directory for the upcoming results, if not already existing
if not os.path.isdir('results'):
    os.mkdir('results')

arr_1 = np.arange(300, 2000, step=200)
arr_2 = np.arange(2000, 5000, step=500)
arr_3 = np.arange(5000, 70000, step=1000)
n_nodes_array = np.concatenate([arr_1, arr_2, arr_3])

n_clusters = 5
initial_distribution = []
rho = 0.6
alpha = 0.1
kappa = 0.1
forgetting_factor = 0.8

T = 10

n_subgraphs = 6
subgraph_sel_alg = 'partition_overlap'
# size_subgraphs_divisor = 3

n_repeat = 5

for n_nodes in n_nodes_array:
    print('n_nodes = ', n_nodes)

    # load the results csv (if already existing) to save the variables
    try:
        results_df = pd.read_csv('results/results_dynamic.csv', sep=';', index_col=False)
        ID = max(results_df['ID']) + 1
    except FileNotFoundError:
        ID = 1

    # size_subgraphs = int(np.floor(n_nodes / size_subgraphs_divisor))
    size_subgraphs = 2 * np.ceil(n_nodes / n_subgraphs)

    GALE_evSC_LeiRinaldo_metric = []
    GALE_evSC_runtimes = []
    evSC_LeiRinaldo_metric = []
    evSC_runtimes = []
    runtime_subgraph_selection = []
    runtime_subgraph_clustering = []

    for _ in tqdm(np.arange(n_repeat)):
        ########################################################
        ########## Initiate SBM
        SBM_object = SBM_Online(n_clusters=n_clusters, n_nodes=n_nodes, rho=rho, alpha=alpha, kappa=kappa)

        ########################################################
        ########## Initiate subgraphs and GALE for evSC
        Selector_object_evSC = SubgraphSelector_Online(n_nodes=n_nodes, n_subgraphs=n_subgraphs, n_clusters=n_clusters,
                                                       size_subgraphs=size_subgraphs, subgraph_sel_alg=subgraph_sel_alg,
                                                       parent_alg='evSC', forgetting_factor=forgetting_factor)
        # indices need to be calculated only once
        indices = Selector_object_evSC.getIndices()
        # initiate GALE object
        GALE_object_evSC = GALE_Online(indices=indices, n_nodes=n_nodes, n_clusters=n_clusters, theta=0.3)

        ########################################################
        ########## for every time step calculate adjacency estimate matrix and perform ev SC and GALE
        labels_true, adj_estimate = SBM_object.simulate_next()
        for t in np.arange(T):
            # SBM
            labels_true, adj = SBM_object.simulate_next()
            adj_estimate = forgetting_factor * adj + (1 - forgetting_factor) * adj_estimate

            # cluster Subgraphs
            labels_subgraphs_evSC = Selector_object_evSC.predict_subgraph_labels(adj)

            del adj

        SBM_setting = SBM_object.get_values()
        del SBM_object

        # perform GALE: align subgraphs
        membership_estimate_evSC = GALE_object_evSC.performGALE(labels_subgraphs_evSC)

        # perform global ev SC: cluster the graph
        SC_object = SpectralClustering(ID=ID, adjacency=adj_estimate, n_clusters=n_clusters,
                                       P_estimate='adjacency')
        SC_estimate = SC_object.performSC()

        ########################################################
        ########## get values

        # get values of ev. SC
        evSC_results = SBM_setting.join(SC_object.get_values())
        del SC_object
        # get metric of ev SC
        evSC_LeiRinaldo_metric.append(LeiRinaldoMetric_1_fromLabels(SC_estimate, labels_true))
        del SC_estimate
        # get runtime of SC
        evSC_runtimes.append(evSC_results['runtime'])

        # get values of GALE
        subgraphs_results_evSC = SBM_setting.join(Selector_object_evSC.get_values())
        GALE_evSC_results = subgraphs_results_evSC.join(GALE_object_evSC.get_values())
        del Selector_object_evSC, GALE_object_evSC
        # get metric of GALE
        GALE_evSC_LeiRinaldo_metric.append(
            LeiRinaldoMetric_1_fromMatrices(membership_estimate_evSC, getMembershipMatrix(labels_true)))
        del membership_estimate_evSC
        # get runtimes of GALE
        GALE_evSC_runtimes.append(GALE_evSC_results['runtime_GALE'])
        runtime_subgraph_selection.append(GALE_evSC_results['runtime_subgraph_selection'])
        runtime_subgraph_clustering.append(GALE_evSC_results['runtime_subgraph_clustering'])

    # save values
    GALE_evSC_results['LeiRinaldoMetric_mean'] = np.mean(GALE_evSC_LeiRinaldo_metric)
    GALE_evSC_results['runtime_GALE'] = np.mean(GALE_evSC_runtimes)
    GALE_evSC_results['runtime_subgraph_selection'] = np.mean(runtime_subgraph_selection)
    GALE_evSC_results['runtime_subgraph_clustering'] = np.mean(runtime_subgraph_clustering)
    GALE_evSC_results['runtime_mean'] = (GALE_evSC_results['runtime_GALE']
                                         + GALE_evSC_results['runtime_subgraph_selection']
                                         + GALE_evSC_results['runtime_subgraph_clustering']
                                         + GALE_evSC_results['runtime_traversal_GALE'])
    GALE_evSC_results['n_repeat'] = n_repeat
    evSC_results['LeiRinaldoMetric_mean'] = np.mean(evSC_LeiRinaldo_metric)
    evSC_results['runtime_mean'] = np.mean(evSC_runtimes)
    del evSC_results['runtime']
    evSC_results['n_repeat'] = n_repeat
    evSC_results['algorithm'] = 'evSC'
    evSC_results['forgetting_factor'] = forgetting_factor

    try:
        results_df = pd.concat([results_df, evSC_results, GALE_evSC_results], ignore_index=True)
    except NameError:
        results_df = pd.concat([evSC_results, GALE_evSC_results], ignore_index=True)

    results_df.to_csv('results/results_dynamic.csv', sep=';', index=False)
