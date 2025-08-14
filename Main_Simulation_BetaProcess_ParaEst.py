# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:15:03 2023

@author: yanbi

This code is used for simulation for parameter estimation of the beta process

"""


# import the public function

import numpy as np
import pandas as pd
import time

# import the self defined functions

from BetaProcess import BetaProcess
from BetaProcessEstimateDiscrete import BetaProcessEstimateDiscrete

#%%

# build the dataframes that saves the simulation result
# df_est_point: saves the parameters estimation results, only for point estimation
# df_est_ci: saves the parameters estimation results, only for the interval estimation

S = 1000
tau = 1
m = 40
scale = 1

# df_est_point: saves the parameters estimation results, only for point estimation
column_names_point = ['tau', 'n', 'w', 'alpha', 'beta'] 
name_list_point = ['alpha_hat', 'beta_hat', 'alpha_mle', 'beta_mle', 'alpha_mm', 'beta_mm']
column_names_point = column_names_point + [f"{name}_s{i}" for name in name_list_point for i in range(1, S+1)]
df_est_point = pd.DataFrame(columns=column_names_point)

# df_est_ci: saves the parameters estimation results, only for the interval estimation
column_names_ci = ['tau', 'n', 'w', 'alpha', 'beta'] 

name_list_ci = ['sigma11_hat', 'sigma12_hat', 'sigma22_hat',
                'sigma11_mle', 'sigma12_mle', 'sigma22_mle',
                'sigma11_mm',  'sigma12_mm',  'sigma22_mm']
column_names_ci = column_names_ci + [f"{name}_s{i}" for name in name_list_ci for i in range(1, S+1)]
df_est_ci = pd.DataFrame(columns=column_names_ci)

#%%
# index of the row in the dataframe

# Record the start time
start_time = time.time()

idx = -1
for n in [50]:
    for w in [7.5]:
        for (alpha,be) in [(4.5,5.5)]:
                idx = idx+1
                for s in range(S):
                    # provide the initial values of degradation time and degradation levels
                    ini_values = np.zeros((n,2))
                    for i in range(n):
                        # the first place saves the initial value of degradation time
                        ini_values[i][0] = 0
                        ini_values[i][1] = 0
  
                    # generate the degradation path using Beta process model
                    x, t_x, L_x, dx = BetaProcess(tau, alpha, be, scale, m, w, n, ini_values, discrete=1, linear=1, params=[0,0])

                    # estimate the model parameters in Beta process using the simulated data above
                    alpha_hat, beta_hat, SIGMA_hat, time_hat = BetaProcessEstimateDiscrete(x, t_x, method = 'new')
                    alpha_mm,  beta_mm,  SIGMA_mm,  time_mm  = BetaProcessEstimateDiscrete(x, t_x, method = 'MM')
                    alpha_mle, beta_mle, SIGMA_mle, time_mle = BetaProcessEstimateDiscrete(x, t_x, method = 'MLE')
                    
                    # df_est_point: saves the parameters estimation results, only for point estimation
                    df_est_point.at[idx, 'tau'] = tau
                    df_est_point.at[idx, 'n'] = n
                    df_est_point.at[idx, 'w'] = w
                    df_est_point.at[idx, 'alpha'] = alpha
                    df_est_point.at[idx, 'beta'] = be
                    df_est_point.at[idx, f'alpha_hat_s{s+1}'] = alpha_hat
                    df_est_point.at[idx, f'beta_hat_s{s+1}'] = beta_hat
                    df_est_point.at[idx, f'alpha_mle_s{s+1}'] = alpha_mle
                    df_est_point.at[idx, f'beta_mle_s{s+1}'] = beta_mle
                    df_est_point.at[idx, f'alpha_mm_s{s+1}'] = alpha_mm
                    df_est_point.at[idx, f'beta_mm_s{s+1}'] = beta_mm
                    
                    
                    # df_est_ci: saves the parameters estimation results, only for the interval estimation
                    df_est_ci.at[idx, 'tau'] = tau
                    df_est_ci.at[idx, 'n'] = n
                    df_est_ci.at[idx, 'w'] = w
                    df_est_ci.at[idx, 'alpha'] = alpha
                    df_est_ci.at[idx, 'beta'] = be
                    df_est_ci.at[idx, f'sigma11_hat_s{s+1}'] = SIGMA_hat[0][0]
                    df_est_ci.at[idx, f'sigma12_hat_s{s+1}'] = SIGMA_hat[0][1]
                    df_est_ci.at[idx, f'sigma22_hat_s{s+1}'] = SIGMA_hat[1][1]
                    df_est_ci.at[idx, f'sigma11_mle_s{s+1}'] = SIGMA_mle[0][0]
                    df_est_ci.at[idx, f'sigma12_mle_s{s+1}'] = SIGMA_mle[0][1]
                    df_est_ci.at[idx, f'sigma22_mle_s{s+1}'] = SIGMA_mle[1][1]
                    df_est_ci.at[idx, f'sigma11_mm_s{s+1}']  = SIGMA_mm[0][0]
                    df_est_ci.at[idx, f'sigma12_mm_s{s+1}']  = SIGMA_mm[0][1]
                    df_est_ci.at[idx, f'sigma22_mm_s{s+1}']  = SIGMA_mm[1][1]


# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time


print(f"Elapsed time: {elapsed_time} seconds")
