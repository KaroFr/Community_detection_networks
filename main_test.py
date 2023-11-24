import os
import numpy as np
import pandas as pd
from SBM import SBM
from SpectralClustering import SpectralClustering
from ErrorMeasures import SarkarMetric_fromLabels, LeiRinaldoMetric_1_fromLabels
from Helpers import getMembershipMatrix, getClusteringMatrix
from tqdm import tqdm

# make a directory for the upcoming results, if not already existing
if not os.path.isdir('results'):
    os.mkdir('results')

# load the results csv (if already existing) to save the variables
try:
    results_df = pd.read_csv('results/results_csv.csv', sep=';', index_col=False)
    ID = max(results_df['ID']) + 1
except FileNotFoundError:
    ID = 1

n_nodes = 3000
n_clusters = 5
rho = 0.7
alphas = np.arange(start=0.02, stop=1.02, step=0.02)

for alpha in alphas:

    adj_Sarkar_metric = []
    Lap_Sarkar_metric = []
    adj_LeiRinaldo_metric = []
    Lap_LeiRinaldo_metric = []

    for _ in tqdm(np.arange(50)):
        print('--------------------------')
        print('------ simulate SBM ------')
        print('--------------------------')
        SBM_object = SBM(n_clusters=n_clusters, n_nodes=n_nodes, rho=rho, alpha=alpha, n_time_steps=1)
        SBMs = SBM_object.simulate()
        labels_true = SBMs['labels'][0]
        prob_matrix_true = SBMs['probability_matrix'][0]
        adj_matrix = SBMs['adj_matrix'][0]
        SBM_setting = SBM_object.get_values()

        membership_mat_true = getMembershipMatrix(labels_true)
        clustering_mat_true = getClusteringMatrix(membership_mat_true)

        ########################################################
        ########## Spectral Clustering
        print('----------------------------')
        print('--- perform adjacency SC ---')
        print('----------------------------')
        ID += 1
        SC_adj_object = SpectralClustering(ID=ID, adjacency=adj_matrix, n_clusters=n_clusters, P_estimate='adjacency')
        SC_adj_estimate = SC_adj_object.performSC()
        SC_adj_results = SBM_setting.join(SC_adj_object.get_values())

        adj_Sarkar_metric.append(SarkarMetric_fromLabels(SC_adj_estimate, labels_true))
        adj_LeiRinaldo_metric.append(LeiRinaldoMetric_1_fromLabels(SC_adj_estimate, labels_true))

        del SC_adj_object, SC_adj_estimate

        print('----------------------------')
        print('--- perform Laplacian SC ---')
        print('----------------------------')
        ID += 1
        SC_lap_object = SpectralClustering(ID=ID, adjacency=adj_matrix, n_clusters=n_clusters, P_estimate='Laplacian')
        SC_lap_estimate = SC_lap_object.performSC()
        SC_lap_results = SBM_setting.join(SC_lap_object.get_values())

        Lap_Sarkar_metric.append(SarkarMetric_fromLabels(SC_lap_estimate, labels_true))
        Lap_LeiRinaldo_metric.append(LeiRinaldoMetric_1_fromLabels(SC_lap_estimate, labels_true))

        del SC_lap_object, SC_lap_estimate

    SC_adj_results['SarkarMetric_mean'] = np.mean(adj_Sarkar_metric)
    SC_adj_results['LeiRinaldoMetric_mean'] = np.mean(adj_LeiRinaldo_metric)
    SC_lap_results['SarkarMetric_mean'] = np.mean(Lap_Sarkar_metric)
    SC_lap_results['LeiRinaldoMetric_mean'] = np.mean(Lap_LeiRinaldo_metric)

    try:
        results_df = pd.concat([results_df, SC_adj_results, SC_lap_results], ignore_index=True)
    except NameError:
        results_df = pd.concat([SC_adj_results, SC_lap_results], ignore_index=True)

results_df.to_csv('results/results_csv.csv', sep=';', index=False)
