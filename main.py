import os
import pandas as pd
from SBM import simulate_SBM_2
from SpectralClustering import SpectralClustering
from PACE import PACE
from GALE import GALE
from ErrorMeasures import *
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

adj, y_true = simulate_SBM_2(K, s, r, p)
membership_matrix = getMembershipMatrix(y_true)
clustering_matrix = membership_matrix @ membership_matrix.transpose()
print(membership_matrix)

#######################################################
############## PACE
print('--------------------------')
print('------ perform PACE ------')
print('--------------------------')
time_start_PACE = time.time()

PACE_object = PACE(ID=ID, adjacency=adj, n_subgraphs=5, size_subgraphs=10, tau=6.0, n_clusters=K)
PACE_estimate_clustering_matrix = PACE_object.performPACE()
PACE_object.applyFinalClustering()
PACE_estimate_labels = PACE_object.clustering_labels_estimate

time_end_PACE = time.time()
print(' Time needed for PACE: %s seconds ' % (time_end_PACE - time_start_PACE))
PACE_results = PACE_object.get_values()
PACE_results['time'] = time_end_PACE - time_start_PACE

print(' Result from PACE:')
print(PACE_estimate_labels)
PACE_results['SarkarMetric'] = SarkarMetric_fromMatrices(PACE_estimate_clustering_matrix, clustering_matrix)
PACE_results['LeiRinaldoMetric_1'] = LeiRinaldoMetric_1_fromLabels(PACE_estimate_labels, y_true)

########################################################
########### PACE with threshold
PACE_object.applyThresholdToEstimate(threshold=0.5)
PACE_estimate_threshold = PACE_object.clustering_matrix_estimate_threshold
PACE_results['SarkarMetric_threshold'] = SarkarMetric_fromMatrices(PACE_estimate_threshold, clustering_matrix)

#######################################################
############## GALE
ID += 1
print('--------------------------')
print('------ perform GALE ------')
print('--------------------------')
time_start_GALE = time.time()

GALE_object = GALE(ID=ID, adjacency=adj, n_subgraphs=5, size_subgraphs=10, tau=1, n_clusters=K)
GALE_estimate = GALE_object.performGALE()

time_end_GALE = time.time()
print(' Time needed for GALE: %s seconds ' % (time_end_GALE - time_start_GALE))
GALE_results = GALE_object.get_values()
GALE_results['time'] = time_end_GALE - time_start_GALE

print(' Result from GALE:')
print(np.round(GALE_estimate, 2))
print(' Metric: ', LeiRinaldoMetric_1_fromMatrices(GALE_estimate, membership_matrix))
GALE_results['LeiRinaldoMetric_1'] = LeiRinaldoMetric_1_fromMatrices(GALE_estimate, membership_matrix)

########################################################
########### Spectral Clustering
ID += 1
print('--------------------------')
print('------ perform SC ------')
print('--------------------------')
time_start_SC = time.time()
SC_object = SpectralClustering(ID=ID, P_estimate=adj, K=K)
SC_estimate = SC_object.performSC()

time_end_SC = time.time()
print(' Time needed for SC: %s seconds ' % (time_end_SC - time_start_SC))
SC_results = SC_object.get_values()
SC_results['time'] = time_end_SC - time_start_SC

print(' Result from SC:')
print(SC_estimate)
print(' Metric: ', LeiRinaldoMetric_1_fromLabels(SC_estimate, y_true))
print(' Metric: ', SarkarMetric_fromLabels(SC_estimate, y_true))
SC_results['LeiRinaldoMetric_1'] = LeiRinaldoMetric_1_fromLabels(SC_estimate, y_true)
SC_results['SarkarMetric'] = SarkarMetric_fromLabels(SC_estimate, y_true)

try:
    df = pd.concat([results_df, PACE_results, GALE_results, SC_results], ignore_index=True)
except NameError:
    df = pd.concat([PACE_results, GALE_results, SC_results], ignore_index=True)

print(df)
df.to_csv('results/results_csv.csv', sep=';', index=False)
