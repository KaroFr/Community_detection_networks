import numpy as np
import pandas as pd
from tqdm import tqdm

from SBM_Offline import SBM_Offline
from SBM_Online import SBM_Online
from SpectralClustering import SpectralClustering
from ErrorMeasures import LeiRinaldoMetric_1_fromLabels

rho = 0.6
n_nodes = 2000
n_clusters = 5
T = 10

n_for = 10

forgetting_factors = np.arange(0.0, 1.02, step=0.02)
alphas = [0.1]
kappas = np.arange(0.0, 1.01, step=0.02)
# kappas = [0.01, 0.025, 0.05, 0.075, 0.1]

LeiRinaldoMetric_SC = 0
LeiRinaldoMetric_evSC_arr = np.zeros(len(forgetting_factors))

for alpha in alphas:

    for kappa in kappas:

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

            for index, forgetting_factor in enumerate(forgetting_factors):
                print('Simulate SBM')
                SBM_object = SBM_Online(n_clusters=n_clusters, n_nodes=n_nodes, rho=rho, alpha=alpha, kappa=kappa)

                # calculate estimate for evolutionary SC
                labels_true, adj_matrix_estimate = SBM_object.simulate_next()
                for t in np.arange(T):
                    labels_true, adj_matrix = SBM_object.simulate_next()
                    adj_matrix_estimate = forgetting_factor * adj_matrix + (1 - forgetting_factor) * adj_matrix_estimate

                # evolutionary Spectral Clustering for last time step
                SC_object_ev = SpectralClustering(adjacency=adj_matrix_estimate, n_clusters=n_clusters, P_estimate='adjacency')
                labels_SC_ev = SC_object_ev.performSC()
                LeiRinaldoMetric_evSC_arr[index] += LeiRinaldoMetric_1_fromLabels(labels_SC_ev, labels_true)

                # static Spectral Clustering for last time step
                SC_object = SpectralClustering(adjacency=adj_matrix, n_clusters=n_clusters, P_estimate='adjacency')
                labels_SC = SC_object.performSC()
                LeiRinaldoMetric_SC += LeiRinaldoMetric_1_fromLabels(labels_SC, labels_true)
                print('index = ', index)

        LeiRinaldoMetric_SC = LeiRinaldoMetric_SC / (n_for*len(forgetting_factors))
        print(LeiRinaldoMetric_SC)
        LeiRinaldoMetric_evSC_arr = LeiRinaldoMetric_evSC_arr / n_for

        # data frame of the whole run
        SBM_setting = SBM_object.get_values()
        SBM_setting['Metric_SC'] = LeiRinaldoMetric_SC
        results_ev_df = pd.DataFrame({'forgetting_factor': forgetting_factors,
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
