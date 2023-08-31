from SBM import simulate_SBM_2
from SpectralClustering import SpectralClustering
from PACE import PACE
from GALE import GALE
from ErrorMeasures import *
import time

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
print(clustering_matrix)

#######################################################
############## PACE
print('--------------------------')
print('------ perform PACE ------')
print('--------------------------')
time_start_PACE = time.time()

PACE_object = PACE(ID=1, adjacency=adj, n_subgraphs=5, size_subgraphs=10, tau=6.0, n_clusters=K)
PACE_estimate = PACE_object.performPACE()

time_end_PACE = time.time()
print(' Time needed for PACE: %s seconds ' % (time_end_PACE - time_start_PACE))

print(' Result from PACE:')
print(np.round(PACE_estimate, 2))
#print(' Metric: ', SarkarMetric_matrix(PACE_estimate, clustering_matrix))

########################################################
########### PACE with threshold
PACE_object.applyThresholdToEstimate(threshold=0.5)
clustering_matrix_threshold = PACE_object.result_estimate_threshold
print(' Result from PACE with threshold:')
print(clustering_matrix_threshold)
#print(' Metric: ', SarkarMetric_matrix(clustering_matrix_threshold, clustering_matrix))

#######################################################
############## GALE
print('--------------------------')
print('------ perform GALE ------')
print('--------------------------')
time_start_GALE = time.time()

GALE_object = GALE(ID=1, adjacency=adj, n_subgraphs=5, size_subgraphs=10, tau=1, n_clusters=K)
GALE_estimate = GALE_object.performGALE()

time_end_GALE = time.time()
print(' Time needed for GALE: %s seconds ' % (time_end_GALE - time_start_GALE))

print(' Result from GALE:')
print(np.round(GALE_estimate, 2))
#print(' Metric: ', LeiRinaldoMetric_1(GALE_estimate, clustering_matrix))

########################################################
########### Spectral Clustering
print('--------------------------')
print('------ perform SC ------')
print('--------------------------')
time_start_SC = time.time()
SC_object = SpectralClustering(ID=1, P_estimate=adj, K=K)
result_LeiRinaldo = SC_object.performSC()

time_end_SC = time.time()
print(' Time needed for SC: %s seconds ' % (time_end_SC - time_start_SC))

print(' Result from SC:')
print(result_LeiRinaldo)
#print(' Metric: ', SarkarMetric_labels(result_LeiRinaldo, y_true))

# result_LeiRinaldo = SpectralClustering(adj, k)
# L = getLaplacian(adj)
# result_RoheEtAl = SpectralClustering(L, k)
#
# print('The true labels are: ', y_true)
# print('Lei Rinaldo yields:  ', result_LeiRinaldo)
# print('Rohe et al yields:   ', result_RoheEtAl)
#
# print('Sarkar Metric of Lei Rinaldo: ', SarkarMetric(result_LeiRinaldo, y_true))
# print('Sarkar Metric of Rohe et al:  ', SarkarMetric(result_RoheEtAl, y_true))
#
# print('L Metric of Lei Rinaldo: ', LeiRinaldoMetric_1(result_LeiRinaldo, y_true))
# print('L Metric of Rohe et al:  ', LeiRinaldoMetric_1(result_RoheEtAl, y_true))
#
# print('\Tilde{L} Metric of Lei Rinaldo: ', LeiRinaldoMetric_2(result_LeiRinaldo, y_true))
# print('\Tilde{L} Metric of Rohe et al:  ', LeiRinaldoMetric_2(result_RoheEtAl, y_true))
