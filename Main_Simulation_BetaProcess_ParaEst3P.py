# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 19:52:19 2024

@author: yanbi

Estimate the model parameter for discrete-time Beta process:
    (1) Need to estimate three parameters, alpha, beta, b
"""

# import the public function

import pandas as pd
import numpy as np
import time

# import self-defined function

from BetaProcess import BetaProcess
from BetaProcessEstimateDiscrete3P import BetaProcessEstimateDiscrete3P

#%%

# build the dataframes that would save the simulation result
# df_est_point: saves the parameters estimation results, for point estimation


S = 1000
tau = 1
m = 70

# df_est_point: saves the parameters estimation results, only for point estimation
column_names_point = ['tau', 'n', 'w', 'alpha', 'beta', 'b'] 
name_list_point = ['alpha_hat', 'beta_hat', 'b_hat',
                   'alpha_mle', 'beta_mle', 'b_mle',
                   'alpha_mm', 'beta_mm', 'b_mm']
column_names_point = column_names_point + [f"{name}_s{i}" for name in name_list_point for i in range(1, S+1)]
df_est_point = pd.DataFrame(columns=column_names_point)

#%%
# Record the start time
start_time = time.time()

# index of the row in the dataframe
idx = -1
for n in [20]:
    for w in [7.5]:
        for (alpha, be, b) in [(4.5, 5.5, 0.6), (4.5, 6.0, 0.6), (4.5, 6.5, 0.6), (4.5, 6.5, 1.0)]:
            idx = idx+1
            for s in range(S):
                # provide the initial values of degradation time and degradation levels
                ini_values = np.zeros((n,2))
                for i in range(n):
                    # the first place saves the initial value of degradation time
                    ini_values[i][0] = 0
                    ini_values[i][1] = 0
  
                # generate the degradation path using Beta process model
                x, t, L, dx = BetaProcess(tau, alpha, be, b, m, w, n, ini_values, discrete=1, linear=1, params=[0,0])

                # estimate the model parameters in Beta process using the simulated data above
                num = 50
                method = 'new-point'                    
                alpha_hat, beta_hat, b_hat, SIGMA_hat, time_hat, num_ite_hat = BetaProcessEstimateDiscrete3P(x, t, method, num, tau)
                
                method = 'MLE-point'
                alpha_mle, beta_mle, b_mle, SIGMA_mle, time_mle, num_ite_mle = BetaProcessEstimateDiscrete3P(x, t, method, num, tau)
                    
                method = 'MM-point'
                alpha_mm, beta_mm, b_mm, SIGMA_mm, time_mm, num_ite_mm = BetaProcessEstimateDiscrete3P(x, t, method, num, tau)
                    
                    
                # df_est_point: saves the parameters estimation results, only for point estimation
                df_est_point.at[idx, 'tau'] = tau
                df_est_point.at[idx, 'n'] = n
                df_est_point.at[idx, 'w'] = w
                df_est_point.at[idx, 'alpha'] = alpha
                df_est_point.at[idx, 'beta'] = be
                df_est_point.at[idx, 'b'] = b
                df_est_point.at[idx, f'alpha_hat_s{s+1}'] = alpha_hat
                df_est_point.at[idx, f'beta_hat_s{s+1}'] = beta_hat
                df_est_point.at[idx, f'b_hat_s{s+1}'] = b_hat
                df_est_point.at[idx, f'alpha_mle_s{s+1}'] = alpha_mle
                df_est_point.at[idx, f'beta_mle_s{s+1}'] = beta_mle
                df_est_point.at[idx, f'b_mle_s{s+1}'] = b_mle
                df_est_point.at[idx, f'alpha_mm_s{s+1}'] = alpha_mm
                df_est_point.at[idx, f'beta_mm_s{s+1}'] = beta_mm
                df_est_point.at[idx, f'b_mm_s{s+1}'] = b_mm


# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")
