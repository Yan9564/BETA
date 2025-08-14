# import the public function

import numpy as np
import pandas as pd

# import the self defined functions

from BetaProcess import BetaProcess
from BetaProcessEstimateDiscrete import BetaProcessEstimateDiscrete



#%%

# build the dataframes that would save the simulation result
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

#%%
# index of the row in the dataframe
# you can load the estimates from Table 3 so that this simlation can be skipped
idx = -1
for n in [1000]:
    for w in [7.5]:
        for (alpha,be) in [(4.5,5.5), (4.5,6), (4.5,6.5)]:
                idx = idx+1
                for s in range(S):
                    # provide the initial values of degradation time and degradation levels
                    ini_values = np.zeros((n,2))
                    for i in range(n):
                        # the first place saves the initial value of degradation time
                        ini_values[i][0] = 0
                        ini_values[i][1] = 0
  
                    # generate the degradation path using Beta process model
                    x, t_x, L_x, Delta_x = BetaProcess(tau, alpha, be, scale, m, w, n, ini_values, discrete=1, linear=1, params=[0,0])

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
                    df_est_point.at[idx, f'time_hat_s{s+1}'] = time_hat
                    df_est_point.at[idx, f'time_mle_s{s+1}'] = time_mle
                    df_est_point.at[idx, f'time_mm_s{s+1}'] = time_mm
                    
                
#%%

column_names = ['tau', 'n', 'w', 'alpha', 'beta', 
                'time_hat_mean', 'time_hat_std', 
                'time_mm_mean', 'time_mm_std',
                'time_mle_mean', 'time_mle_std'] 
df_est_time = pd.DataFrame(columns=column_names)

for i in range(len(df_est_point)):
    time_hats = df_est_point.loc[i, [f"time_hat_s{i}" for i in range(1, S+1)]]
    time_mms = df_est_point.loc[i, [f"time_mm_s{i}" for i in range(1, S+1)]]
    time_mles = df_est_point.loc[i, [f"time_mle_s{i}" for i in range(1, S+1)]]
    
    # information
    df_est_time.at[i, 'tau'] = df_est_point.loc[i, 'tau']
    df_est_time.at[i, 'n'] = df_est_point.loc[i, 'n']
    df_est_time.at[i, 'w'] = df_est_point.loc[i, 'w']
    df_est_time.at[i, 'alpha'] = df_est_point.loc[i, 'alpha']
    df_est_time.at[i, 'beta'] = df_est_point.loc[i, 'beta']
    
    # running time
    df_est_time.at[i, 'time_hat_mean'] = np.mean(time_hats)
    df_est_time.at[i, 'time_mm_mean'] = np.mean(time_mms)
    df_est_time.at[i, 'time_mle_mean'] = np.mean(time_mles)
    df_est_time.at[i, 'time_hat_std'] = np.std(time_hats)
    df_est_time.at[i, 'time_mm_std'] = np.std(time_mms)
    df_est_time.at[i, 'time_mle_std'] = np.std(time_mles)
   



