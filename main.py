import os
import pandas as pd
import numpy as np
from SBM import simulate_SBM_2
from SpectralClustering import SpectralClustering
from PACE import PACE
from GALE import GALE
from ErrorMeasures import SarkarMetric_fromLabels, SarkarMetric_fromMatrices, LeiRinaldoMetric_1_fromLabels, LeiRinaldoMetric_1_fromMatrices
from Helpers import getMembershipMatrix, getClusteringMatrix
import time

# make a directory for the upcoming results, if not already existing
if not os.path.isdir('results'):
    os.mkdir('results')

# load the results csv (if already existing) to save the variables
try:
    results_df = pd.read_csv('results/results_csv.csv', sep=';', index_col=False)
    ID = max(results_df['ID']) + 1
except FileNotFoundError:
    ID = 1

print('--------------------------')
print('------ simulate SBM ------')
print('--------------------------')
K = 8
s = 10000
r = 0.3
p = 0.6

adj_true, clustering_labels_true = simulate_SBM_2(K, s, r, p)
membership_mat_true = getMembershipMatrix(clustering_labels_true)
clustering_mat_true = getClusteringMatrix(membership_mat_true)
#print('true labels')
#print(clustering_labels_true)

#######################################################
############## set parameters
n_subgraphs = 10
size_subgraphs = int(K*s/4)

PACE_tau = 6.0
PACE_threshold = 0.5

GALE_tau = 1

#######################################################
############## PACE
print('--------------------------')
print('------ perform PACE ------')
print('--------------------------')

PACE_object = PACE(ID=ID, adjacency=adj_true, n_subgraphs=n_subgraphs, size_subgraphs=size_subgraphs, tau=PACE_tau, n_clusters=K, apply_threshold=False)
PACE_estimate_labels = PACE_object.performPACE()
PACE_estimate_clustering_matrix = PACE_object.clustering_matrix_estimate

PACE_results = PACE_object.get_values()

#print(' Result from PACE:')
#print(PACE_estimate_labels)
start_time_metric = time.time()
PACE_results['SarkarMetric'] = SarkarMetric_fromMatrices(PACE_estimate_clustering_matrix, clustering_mat_true)
second_time_metric = time.time()
PACE_results['LeiRinaldoMetric_1'] = LeiRinaldoMetric_1_fromLabels(PACE_estimate_labels, clustering_labels_true)
end_time_metric = time.time()
print('Calculating the metrics took ', np.round(second_time_metric - start_time_metric, 4), ' and ', np.round(end_time_metric - second_time_metric, 4), ' seconds')

del PACE_object, PACE_estimate_labels, PACE_estimate_clustering_matrix
########################################################
########### PACE with threshold
ID += 1
print('-----------------------------------------')
print('------ perform PACE with threshold ------')
print('-----------------------------------------')

PACE_object_threshold = PACE(ID=ID, adjacency=adj_true, n_subgraphs=n_subgraphs, size_subgraphs=size_subgraphs, tau=PACE_tau, n_clusters=K, apply_threshold=True, threshold=PACE_threshold)
PACE_estimate_labels_threshold = PACE_object_threshold.performPACE()
PACE_estimate_clustering_matrix_threshold = PACE_object_threshold.clustering_matrix_estimate_threshold

PACE_results_threshold = PACE_object_threshold.get_values()

#print(' Result from PACE:')
#print(PACE_estimate_labels_threshold)
start_time_metric = time.time()
PACE_results_threshold['SarkarMetric'] = SarkarMetric_fromMatrices(PACE_estimate_clustering_matrix_threshold, clustering_mat_true)
second_time_metric = time.time()
PACE_results_threshold['LeiRinaldoMetric_1'] = LeiRinaldoMetric_1_fromLabels(PACE_estimate_labels_threshold, clustering_labels_true)
end_time_metric = time.time()
print('Calculating the metrics took ', np.round(second_time_metric - start_time_metric, 4), ' and ', np.round(end_time_metric - second_time_metric, 4), ' seconds')


del PACE_object_threshold, PACE_estimate_labels_threshold, PACE_estimate_clustering_matrix_threshold
#######################################################
############## GALE
ID += 1
print('--------------------------')
print('------ perform GALE ------')
print('--------------------------')

GALE_object = GALE(ID=ID, adjacency=adj_true, n_subgraphs=n_subgraphs, size_subgraphs=size_subgraphs, tau=GALE_tau, n_clusters=K, weightedTraversal=False)
GALE_estimate_membership_matrix = GALE_object.performGALE()
GALE_estimate_clustering_matrix = getClusteringMatrix(GALE_estimate_membership_matrix)

GALE_results_1 = GALE_object.get_values()

#print(' Result from GALE:')
#print(GALE_estimate_membership_matrix)

start_time_metric = time.time()
GALE_results_1['SarkarMetric'] = SarkarMetric_fromMatrices(GALE_estimate_clustering_matrix, clustering_mat_true)
second_time_metric = time.time()
GALE_results_1['LeiRinaldoMetric_1'] = LeiRinaldoMetric_1_fromMatrices(GALE_estimate_membership_matrix, membership_mat_true)
end_time_metric = time.time()
print('Calculating the metrics took ', np.round(second_time_metric - start_time_metric, 4), ' and ', np.round(end_time_metric - second_time_metric, 4), ' seconds')


del GALE_object, GALE_estimate_membership_matrix, GALE_estimate_clustering_matrix
#######################################################
############## GALE with weighted traversal
ID += 1
print('--------------------------')
print('------ perform GALE ------')
print('--------------------------')

GALE_object = GALE(ID=ID, adjacency=adj_true, n_subgraphs=n_subgraphs, size_subgraphs=size_subgraphs, tau=GALE_tau, n_clusters=K, weightedTraversal=True)
GALE_estimate_membership_matrix = GALE_object.performGALE()
GALE_estimate_clustering_matrix = getClusteringMatrix(GALE_estimate_membership_matrix)

GALE_results_2 = GALE_object.get_values()

#print(' Result from GALE:')
#print(GALE_estimate_membership_matrix)

start_time_metric = time.time()
GALE_results_2['SarkarMetric'] = SarkarMetric_fromMatrices(GALE_estimate_clustering_matrix, clustering_mat_true)
second_time_metric = time.time()
GALE_results_2['LeiRinaldoMetric_1'] = LeiRinaldoMetric_1_fromMatrices(GALE_estimate_membership_matrix, membership_mat_true)
end_time_metric = time.time()
print('Calculating the metrics took ', np.round(second_time_metric - start_time_metric, 4), ' and ', np.round(end_time_metric - second_time_metric, 4), ' seconds')


del GALE_object, GALE_estimate_membership_matrix, GALE_estimate_clustering_matrix
########################################################
########### Spectral Clustering
ID += 1
print('--------------------------')
print('------ perform SC ------')
print('--------------------------')
SC_object = SpectralClustering(ID=ID, P_estimate=adj_true, K=K)
SC_estimate = SC_object.performSC()

SC_results = SC_object.get_values()

#print(' Result from SC:')
#print(SC_estimate)
start_time_metric = time.time()
SC_results['SarkarMetric'] = SarkarMetric_fromLabels(SC_estimate, clustering_labels_true)
second_time_metric = time.time()
SC_results['LeiRinaldoMetric_1'] = LeiRinaldoMetric_1_fromLabels(SC_estimate, clustering_labels_true)
end_time_metric = time.time()
print('Calculating the metrics took ', np.round(second_time_metric - start_time_metric, 4), ' and ', np.round(end_time_metric - second_time_metric, 4), ' seconds')


del SC_object, SC_estimate

try:
    df = pd.concat([results_df, PACE_results, PACE_results_threshold, GALE_results_1, GALE_results_2, SC_results], ignore_index=True)
except NameError:
    df = pd.concat([PACE_results, PACE_results_threshold, GALE_results_1, GALE_results_2, SC_results], ignore_index=True)

df.to_csv('results/results_csv.csv', sep=';', index=False)
