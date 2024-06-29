import numpy as np
import pandas as pd
from tqdm import tqdm

from SBM_Offline import SBM_Offline
from SpectralClustering import SpectralClustering
from ErrorMeasures import LeiRinaldoMetric_1_fromLabels

rho = 0.6
n_nodes = 2000
n_clusters = 5
T = 100

n_for = 10

forgetting_factors = np.arange(0.0, 1.02, step=0.02)
alphas = [0.1]
epsilons = np.arange(0.0, 1.01, step=0.01)
# epsilons = [0.01, 0.025, 0.05, 0.075, 0.1]

LeiRinaldoMetric_SC = 0
LeiRinaldoMetric_evSC_arr = np.zeros(len(forgetting_factors))

for alpha in alphas:

    for epsilon in epsilons:

        # load the results csv (if already existing) to save the variables
        try:
            results_df_1 = pd.read_csv('results/results_ev_SC.csv', sep=';', index_col=False)
        except FileNotFoundError:
            ID = 1
        try:
            results_df_2 = pd.read_csv('results/results_ev_SC_optimal.csv', sep=';', index_col=False)
        except:
            ID = 1

        for _ in tqdm(np.arange(n_for)):

            print('Simulate SBM')
            SBM_object = SBM_Offline(n_clusters=n_clusters, n_nodes=n_nodes, rho=rho, alpha=alpha, n_time_steps=T, epsilon=epsilon)
            SBMs = SBM_object.simulate()
            labels_true = SBMs['labels'][T - 1]

            # static Spectral Clustering
            SC_object = SpectralClustering(adjacency=SBMs['adj_matrix'][T - 1], n_clusters=n_clusters, P_estimate='adjacency')
            labels_SC = SC_object.performSC()
            LeiRinaldoMetric_SC += LeiRinaldoMetric_1_fromLabels(labels_SC, labels_true)

            for index, forgetting_factor in enumerate(forgetting_factors):

                # calculate estimate for evolutionary SC
                adj_matrix_estimate = SBMs['adj_matrix'][0]
                for t in np.arange(T):
                    adj_matrix = SBMs['adj_matrix'][t]
                    adj_matrix_estimate = forgetting_factor * adj_matrix + (1 - forgetting_factor) * adj_matrix_estimate

                # evolutionary Spectral Clustering
                SC_object_ev = SpectralClustering(adjacency=adj_matrix_estimate, n_clusters=n_clusters, P_estimate='adjacency')
                labels_SC_ev = SC_object_ev.performSC()

                # get misclustering error
                LeiRinaldoMetric_evSC_arr[index] += LeiRinaldoMetric_1_fromLabels(labels_SC_ev, labels_true)

        LeiRinaldoMetric_SC = LeiRinaldoMetric_SC / n_for
        LeiRinaldoMetric_evSC_arr = LeiRinaldoMetric_evSC_arr / n_for

        # data frame of the whole run
        SBM_setting = SBM_object.get_values()
        SBM_setting['Metric_SC'] = LeiRinaldoMetric_SC
        results_ev_df = pd.DataFrame({'forgetting_factor ': forgetting_factors,
                                      'Metric_evSC': LeiRinaldoMetric_evSC_arr})
        results_ev_df = pd.concat([SBM_setting, results_ev_df], axis=1)

        try:
            results_df_1 = pd.concat([results_df_1, results_ev_df], ignore_index=True)
        except NameError:
            results_df_1 = pd.concat([results_ev_df], ignore_index=True)

        results_df_1.to_csv('results/results_ev_SC.csv', sep=';', index=False)

        # data frame of the optimal values only
        opt_index = np.argmin(LeiRinaldoMetric_evSC_arr)
        SBM_setting['opt_forgetting_factor'] = forgetting_factors[opt_index]
        SBM_setting['opt_Metric_evSC'] = LeiRinaldoMetric_evSC_arr[opt_index]

        try:
            results_df_2 = pd.concat([results_df_2, SBM_setting], ignore_index=True)
        except NameError:
            results_df_2 = pd.concat([SBM_setting], ignore_index=True)

        results_df_2.to_csv('results/results_ev_SC_optimal.csv', sep=';', index=False)
