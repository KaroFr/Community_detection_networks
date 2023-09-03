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
K = 3
s = 4
r = 0.3
p = 0.6

adj_true, clustering_labels_true = simulate_SBM_2(K, s, r, p)
membership_mat_true = getMembershipMatrix(clustering_labels_true)
clustering_mat_true = getClusteringMatrix(membership_mat_true)
print('true labels')
print(clustering_labels_true)

#######################################################
############## PACE
print('--------------------------')
print('------ perform PACE ------')
print('--------------------------')
time_start_PACE = time.time()

PACE_object = PACE(ID=ID, adjacency=adj_true, n_subgraphs=5, size_subgraphs=10, tau=6.0, n_clusters=K, apply_threshold=False)
PACE_estimate_labels = PACE_object.performPACE()
PACE_estimate_clustering_matrix = PACE_object.clustering_matrix_estimate

time_end_PACE = time.time()

PACE_results = PACE_object.get_values()
PACE_results['time'] = time_end_PACE - time_start_PACE

print(' Result from PACE:')
print(PACE_estimate_labels)
PACE_results['SarkarMetric'] = SarkarMetric_fromMatrices(PACE_estimate_clustering_matrix, clustering_mat_true)
PACE_results['LeiRinaldoMetric_1'] = LeiRinaldoMetric_1_fromLabels(PACE_estimate_labels, clustering_labels_true)

########################################################
########### PACE with threshold
ID += 1
print('-----------------------------------------')
print('------ perform PACE with threshold ------')
print('-----------------------------------------')
time_start_PACE = time.time()

PACE_object = PACE(ID=ID, adjacency=adj_true, n_subgraphs=5, size_subgraphs=10, tau=6.0, n_clusters=K, apply_threshold=True, threshold=0.5)
PACE_estimate_labels = PACE_object.performPACE()
PACE_estimate_clustering_matrix = PACE_object.clustering_matrix_estimate_threshold

time_end_PACE = time.time()

PACE_results_threshold = PACE_object.get_values()
PACE_results_threshold['time'] = time_end_PACE - time_start_PACE

print(' Result from PACE:')
print(PACE_estimate_labels)
PACE_results_threshold['SarkarMetric'] = SarkarMetric_fromMatrices(PACE_estimate_clustering_matrix, clustering_mat_true)
PACE_results_threshold['LeiRinaldoMetric_1'] = LeiRinaldoMetric_1_fromLabels(PACE_estimate_labels, clustering_labels_true)

#######################################################
############## GALE
ID += 1
print('--------------------------')
print('------ perform GALE ------')
print('--------------------------')
time_start_GALE = time.time()

GALE_object = GALE(ID=ID, adjacency=adj_true, n_subgraphs=5, size_subgraphs=10, tau=1, n_clusters=K)
GALE_estimate = GALE_object.performGALE()

time_end_GALE = time.time()

GALE_results = GALE_object.get_values()
GALE_results['time'] = time_end_GALE - time_start_GALE

print(' Result from GALE:')
print(np.round(GALE_estimate, 2))
GALE_results['LeiRinaldoMetric_1'] = LeiRinaldoMetric_1_fromMatrices(GALE_estimate, membership_mat_true)

########################################################
########### Spectral Clustering
ID += 1
print('--------------------------')
print('------ perform SC ------')
print('--------------------------')
time_start_SC = time.time()
SC_object = SpectralClustering(ID=ID, P_estimate=adj_true, K=K)
SC_estimate = SC_object.performSC()

time_end_SC = time.time()

SC_results = SC_object.get_values()
SC_results['time'] = time_end_SC - time_start_SC

print(' Result from SC:')
print(SC_estimate)
SC_results['LeiRinaldoMetric_1'] = LeiRinaldoMetric_1_fromLabels(SC_estimate, clustering_labels_true)
SC_results['SarkarMetric'] = SarkarMetric_fromLabels(SC_estimate, clustering_labels_true)

try:
    df = pd.concat([results_df, PACE_results, PACE_results_threshold, GALE_results, SC_results], ignore_index=True)
except NameError:
    df = pd.concat([PACE_results, PACE_results_threshold, GALE_results, SC_results], ignore_index=True)

df.to_csv('results/results_csv.csv', sep=';', index=False)
