import os
import numpy as np
import pandas as pd

from GALE_Offline import GALE_Offline
from Helpers import getMembershipMatrix
from HierarchicalClustering import HierarchicalClustering
from PACE_Offline import PACE_Offline
from PACE_Online import PACE_Online
from SBM_Offline import SBM_Offline
from SBM_Online import SBM_Online
from SpectralClustering import SpectralClustering
from ErrorMeasures import LeiRinaldoMetric_1_fromLabels, LeiRinaldoMetric_1_fromMatrices
from tqdm import tqdm
from SubgraphSelector_Offline import SubgraphSelector_Offline

if __name__ == '__main__':
    # make a directory for the upcoming results, if not already existing
    from SubgraphSelector_Online import SubgraphSelector_Online

    if not os.path.isdir('results'):
        os.mkdir('results')

    arr_1 = np.arange(200, 2000, step=200)
    arr_2 = np.arange(2000, 5000, step=500)
    arr_3 = np.arange(5000, 38000, step=1000)
    n_nodes_array = np.concatenate([arr_1, arr_2, arr_3])

    n_clusters = 5
    initial_distribution = []
    rho = 0.6
    alpha = 0.1

    n_subgraphs = 10
    size_subgraphs_divisor = 3
    # size_subgraphs = 1500

    for n_nodes in n_nodes_array:
        # expected degree < n_nodes * alpha
        # -> this doubles the degree
        regularization_tau = n_nodes * alpha

        # load the results csv (if already existing) to save the variables
        try:
            results_df = pd.read_csv('results/results_csv_large_n_partition.csv', sep=';', index_col=False)
            ID = max(results_df['ID']) + 1
        except FileNotFoundError:
            ID = 1

        # size_subgraphs = int(np.floor(n_nodes / size_subgraphs_divisor))

        SC_LeiRinaldo_metric = []
        # rSC_LeiRinaldo_metric = []
        # HC_LeiRinaldo_metric = []
        PACE_LeiRinaldo_metric = []
        GALE_LeiRinaldo_metric = []
        SC_runtimes = []
        # rSC_runtimes = []
        # HC_runtimes = []
        PACE_runtimes = []
        GALE_runtimes = []

        for _ in tqdm(np.arange(1)):
            print('--------------------------')
            print('------ simulate SBM ------')
            print('--------------------------')
            SBM_object = SBM_Offline(n_clusters=n_clusters, n_nodes=n_nodes, rho=rho, alpha=alpha, n_time_steps=1)
            SBMs = SBM_object.simulate()
            labels_true = SBMs['labels'][0]
            adj_matrix = SBMs['adj_matrix'][0]
            SBM_setting = SBM_object.get_values()

            del SBM_object
            ########################################################
            ########## regularized Spectral Clustering
            # print('----------------------------')
            # print('--- perform reg. adj. SC ---')
            # print('----------------------------')
            #
            # rSC_adj_object = SpectralClustering(ID=ID, adjacency=adj_matrix, n_clusters=n_clusters,
            #                                     P_estimate='regularized', regularization_tau=regularization_tau)
            # rSC_adj_estimate = rSC_adj_object.performSC()
            # rSC_adj_results = SBM_setting.join(rSC_adj_object.get_values())
            #
            # rSC_LeiRinaldo_metric.append(LeiRinaldoMetric_1_fromLabels(rSC_adj_estimate, labels_true))
            # rSC_runtimes.append(rSC_adj_results['runtime'])
            # del rSC_adj_object, rSC_adj_estimate

            ########################################################
            ########## Spectral Clustering
            print('----------------------------')
            print('--- perform adjacency SC ---')
            print('----------------------------')

            SC_adj_object = SpectralClustering(ID=ID, adjacency=adj_matrix, n_clusters=n_clusters,
                                               P_estimate='adjacency')
            SC_adj_estimate = SC_adj_object.performSC()
            SC_adj_results = SBM_setting.join(SC_adj_object.get_values())

            SC_LeiRinaldo_metric.append(LeiRinaldoMetric_1_fromLabels(SC_adj_estimate, labels_true))
            SC_runtimes.append(SC_adj_results['runtime'])
            del SC_adj_object, SC_adj_estimate

            ########################################################
            ########## Hierarchical Clustering
            # print('----------------------------')
            # print('--- perform adjacency HC ---')
            # print('----------------------------')
            #
            # HC_adj_object = HierarchicalClustering(ID=ID, adjacency=adj_matrix, n_clusters=n_clusters)
            # HC_adj_estimate = HC_adj_object.performHC()
            # HC_adj_results = SBM_setting.join(HC_adj_object.get_values())
            #
            # HC_LeiRinaldo_metric.append(LeiRinaldoMetric_1_fromLabels(HC_adj_estimate, labels_true))
            # HC_runtimes.append(HC_adj_results['runtime'])
            # del HC_adj_object, HC_adj_estimate, adj_matrix

            ########################################################
            ########## Divide and cluster subgraphs
            Selector_object = SubgraphSelector_Offline(ID=ID, SBMs=SBMs, n_subgraphs=n_subgraphs,
                                                       n_clusters=n_clusters, parent_alg='SC',
                                                       subgraph_sel_alg='partition_overlap')
            subgraphs_df = Selector_object.getSubgraphs()
            subgraphs_results = SBM_setting.join(Selector_object.get_values())

            del Selector_object, SBMs

            ########################################################
            ########## PACE with SC
            print('----------------------------')
            print('--- perform PACE -----------')
            print('----------------------------')
            PACE_object = PACE_Offline(subgraphs_df=subgraphs_df, n_nodes=n_nodes, n_clusters=n_clusters)
            PACE_labels_estimate = PACE_object.performPACE()[0]
            PACE_results = subgraphs_results.join(PACE_object.get_values())

            PACE_LeiRinaldo_metric.append(LeiRinaldoMetric_1_fromLabels(PACE_labels_estimate, labels_true))
            PACE_runtimes.append(PACE_results['runtime'])

            del PACE_object, PACE_labels_estimate
            ########################################################
            ########## GALE with SC
            print('----------------------------')
            print('--- perform GALE -----------')
            print('----------------------------')
            GALE_object = GALE_Offline(subgraphs_df=subgraphs_df, n_nodes=n_nodes, n_clusters=n_clusters, theta=0.0)
            GALE_memb_estimate = GALE_object.performGALE()[0]
            GALE_results = subgraphs_results.join(GALE_object.get_values())

            GALE_LeiRinaldo_metric.append(
                LeiRinaldoMetric_1_fromMatrices(GALE_memb_estimate, getMembershipMatrix(labels_true)))
            GALE_runtimes.append(GALE_results['runtime'])

            del GALE_object, GALE_memb_estimate, subgraphs_df, subgraphs_results

        SC_adj_results['LeiRinaldoMetric_mean'] = np.mean(SC_LeiRinaldo_metric)
        SC_adj_results['runtime_mean'] = np.mean(SC_runtimes)
        # rSC_adj_results['LeiRinaldoMetric_mean'] = np.mean(rSC_LeiRinaldo_metric)
        # rSC_adj_results['runtime_mean'] = np.mean(rSC_runtimes)
        # HC_adj_results['LeiRinaldoMetric_mean'] = np.mean(HC_LeiRinaldo_metric)
        # HC_adj_results['runtime_mean'] = np.mean(HC_runtimes)
        PACE_results['LeiRinaldoMetric_mean'] = np.mean(PACE_LeiRinaldo_metric)
        PACE_results['runtime_mean'] = np.mean(PACE_runtimes)
        GALE_results['LeiRinaldoMetric_mean'] = np.mean(GALE_LeiRinaldo_metric)
        GALE_results['runtime_mean'] = np.mean(GALE_runtimes)

        try:
            results_df = pd.concat([results_df, SC_adj_results, PACE_results, GALE_results], ignore_index=True)
        except NameError:
            results_df = pd.concat([SC_adj_results, PACE_results, GALE_results], ignore_index=True)

        results_df.to_csv('results/results_csv_large_n_partition.csv', sep=';', index=False)
