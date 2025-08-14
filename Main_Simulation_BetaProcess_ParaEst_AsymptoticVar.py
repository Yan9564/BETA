# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:15:03 2023

@author: yanbi

This code is used for simulation for parameter estimation of the beta process

"""


# import the public function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# df_est_ci: saves the parameters estimation results, only for the interval estimation
column_names_ci = ['tau', 'n', 'w', 'alpha', 'beta'] 

name_list_ci = ['sigma11_hat', 'sigma12_hat', 'sigma22_hat',
                'sigma11_mle', 'sigma12_mle', 'sigma22_mle',
                'sigma11_mm',  'sigma12_mm',  'sigma22_mm']
column_names_ci = column_names_ci + [f"{name}_s{i}" for name in name_list_ci for i in range(1, S+1)]
df_est_ci = pd.DataFrame(columns=column_names_ci)

#%%
# index of the row in the dataframe
idx = -1
for n in [50]:
    for w in [7.5]:
        for (alpha,be) in [(4.5,3.5), (4.5,4), (4.5,4.5), (4.5,5), (4.5,5.5), (4.5,6), (4.5,6.5), 
                           (3.5,4.5), (4, 4.5), (5,4.5), (5.5,4.5), (6,4.5), (6.5,4.5)]:
                idx = idx+1
                for s in range(S):
                    print('(alpha,beta)=('+str(alpha)+','+str(be)+')-----s='+str(s))
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
                    
                    # df_est_ci: saves the parameters estimation results, only for the interval estimation
                    df_est_ci.at[idx, 'tau'] = tau
                    df_est_ci.at[idx, 'n'] = n
                    df_est_ci.at[idx, 'w'] = w
                    df_est_ci.at[idx, 'alpha'] = alpha
                    df_est_ci.at[idx, 'beta'] = be
                    df_est_ci.at[idx, f'alpha_hat_s{s+1}'] = alpha_hat
                    df_est_ci.at[idx, f'beta_hat_s{s+1}'] = beta_hat
                    df_est_ci.at[idx, f'alpha_mle_s{s+1}'] = alpha_mle
                    df_est_ci.at[idx, f'beta_mle_s{s+1}'] = beta_mle
                    df_est_ci.at[idx, f'alpha_mm_s{s+1}'] = alpha_mm
                    df_est_ci.at[idx, f'beta_mm_s{s+1}'] = beta_mm
                    df_est_ci.at[idx, f'sigma11_hat_s{s+1}'] = SIGMA_hat[0][0]
                    df_est_ci.at[idx, f'sigma12_hat_s{s+1}'] = SIGMA_hat[0][1]
                    df_est_ci.at[idx, f'sigma22_hat_s{s+1}'] = SIGMA_hat[1][1]
                    df_est_ci.at[idx, f'sigma11_mle_s{s+1}'] = SIGMA_mle[0][0]
                    df_est_ci.at[idx, f'sigma12_mle_s{s+1}'] = SIGMA_mle[0][1]
                    df_est_ci.at[idx, f'sigma22_mle_s{s+1}'] = SIGMA_mle[1][1]
                    df_est_ci.at[idx, f'sigma11_mm_s{s+1}']  = SIGMA_mm[0][0]
                    df_est_ci.at[idx, f'sigma12_mm_s{s+1}']  = SIGMA_mm[0][1]
                    df_est_ci.at[idx, f'sigma22_mm_s{s+1}']  = SIGMA_mm[1][1]


#%%

# Interval Estimation: calculation the mean value of the covariance matrix over all the simulation

# save the simulation result in the dataframe
column_names = ['tau', 'n', 'w', 'alpha', 'beta'] 
column_names = column_names + ['sigma11_hat_mean', 'sigma12_hat_mean', 'sigma22_hat_mean']
column_names = column_names + ['sigma11_mle_mean', 'sigma12_mle_mean', 'sigma22_mle_mean']
df_est_ci_mean = pd.DataFrame(columns=column_names)

for i in range(len(df_est_ci)):
    print('==================================')
    n = df_est_ci.at[i, 'n']
    w = df_est_ci.at[i, 'w']
    alpha = df_est_ci.at[i, 'alpha']
    beta = df_est_ci.at[i, 'beta']
    tau = df_est_ci.at[i, 'tau']
    print('n:', n)
    print('w:', w)
    print('alpha:', alpha)
    print('beta:', beta)
    
    sigma11_hats = df_est_ci.loc[i,'sigma11_hat_s1':'sigma11_hat_s'+str(S)]
    sigma12_hats = df_est_ci.loc[i,'sigma12_hat_s1':'sigma12_hat_s'+str(S)]
    sigma22_hats = df_est_ci.loc[i,'sigma22_hat_s1':'sigma22_hat_s'+str(S)]
    sigma11_mles = df_est_ci.loc[i,'sigma11_mle_s1':'sigma11_mle_s'+str(S)]
    sigma12_mles = df_est_ci.loc[i,'sigma12_mle_s1':'sigma12_mle_s'+str(S)]
    sigma22_mles = df_est_ci.loc[i,'sigma22_mle_s1':'sigma22_mle_s'+str(S)]
    sigma11_mms = df_est_ci.loc[i,'sigma11_mm_s1':'sigma11_mm_s'+str(S)]
    sigma12_mms = df_est_ci.loc[i,'sigma12_mm_s1':'sigma12_mm_s'+str(S)]
    sigma22_mms = df_est_ci.loc[i,'sigma22_mm_s1':'sigma22_mm_s'+str(S)]

    
    df_est_ci_mean.at[i, 'n'] = n
    df_est_ci_mean.at[i, 'w'] = w
    df_est_ci_mean.at[i, 'tau'] = tau
    df_est_ci_mean.at[i, 'alpha'] = alpha
    df_est_ci_mean.at[i, 'beta'] = beta
    # bias
    df_est_ci_mean.at[i, 'sigma11_hat_mean'] = sigma11_hats.mean()
    df_est_ci_mean.at[i, 'sigma12_hat_mean'] = sigma12_hats.mean()
    df_est_ci_mean.at[i, 'sigma22_hat_mean'] = sigma22_hats.mean()
    df_est_ci_mean.at[i, 'sigma11_mle_mean'] = sigma11_mles.mean()
    df_est_ci_mean.at[i, 'sigma12_mle_mean'] = sigma12_mles.mean()
    df_est_ci_mean.at[i, 'sigma22_mle_mean'] = sigma22_mles.mean()
    df_est_ci_mean.at[i, 'sigma11_mm_mean'] = sigma11_mms.mean()
    df_est_ci_mean.at[i, 'sigma12_mm_mean'] = sigma12_mms.mean()
    df_est_ci_mean.at[i, 'sigma22_mm_mean'] = sigma22_mms.mean()


#%%
# Interval Estimation: plot the mean value of the covariance matrix

# var beta VS beta

plt.rcParams["figure.figsize"] = (3.3,3)

df = df_est_ci_mean[(df_est_ci_mean['n']==50) & (df_est_ci_mean['alpha']==4.5)]
plt.plot(df.beta, df.sigma22_hat_mean, 'r-o', markerfacecolor='none', label = 'our method')
plt.plot(df.beta, df.sigma22_mle_mean, 'k--v', markerfacecolor='none', label = 'ML')
plt.plot(df.beta, df.sigma22_mm_mean, 'b--s', markerfacecolor='none', label = 'MM')
plt.ylim(bottom=0)
plt.xticks(df.beta.tolist())
plt.legend()
plt.xlabel(r'$\beta$')
plt.ylabel(r'asymptotic var of $\beta$')
plt.tight_layout()


#%%
# Interval Estimation: plot the mean value of the covariance matrix

# var alpha VS beta
plt.rcParams["figure.figsize"] = (3.3,3)

df = df_est_ci_mean[(df_est_ci_mean['n']==50) & (df_est_ci_mean['beta']==4.5)]
df = df.sort_values(by='alpha', ascending=True)
plt.plot(df.alpha, df.sigma11_hat_mean, 'r-o', markerfacecolor='none', label = 'our method')
plt.plot(df.alpha, df.sigma11_mle_mean, 'k--v', markerfacecolor='none', label = 'ML')
plt.plot(df.alpha, df.sigma11_mm_mean, 'b--s', markerfacecolor='none', label = 'MM')
plt.ylim(bottom=-0.5)
plt.xticks(df.alpha.tolist())
plt.legend()
plt.xlabel(r'$\alpha$')
plt.ylabel(r'asymptotic var of $\alpha$')
plt.tight_layout()
    

#%% 
# Interval Estimation: compare the coverage probability
from scipy.stats import norm
tail_probability = 0.05
critical_value = norm.ppf(1 - tail_probability/2)

#%%

column_names = ['tau', 'n', 'w', 'alpha', 'beta'] 
column_names = column_names + ['alpha_ci_hat', 'beta_ci_hat']
column_names = column_names + ['alpha_ci_mle', 'beta_ci_mle']
column_names = column_names + ['alpha_ci_mm',  'beta_ci_mm']

df_est_ci_cp = pd.DataFrame(columns=column_names)
df_est_point = df_est_ci

for i in range(len(df_est_ci)):
    print('==================================')
    n = df_est_ci.at[i, 'n']
    w = df_est_ci.at[i, 'w']
    alpha = df_est_ci.at[i, 'alpha']
    beta = df_est_ci.at[i, 'beta']
    tau = df_est_ci.at[i, 'tau']
    print('n:', n)
    print('w:', w)
    print('alpha:', alpha)
    print('beta:', beta)
    df_est_ci_cp.at[i,'n'] = n
    df_est_ci_cp.at[i,'w'] = w
    df_est_ci_cp.at[i,'alpha'] = alpha
    df_est_ci_cp.at[i,'beta'] = beta
    df_est_ci_cp.at[i,'tau'] = tau
    
    for estimator in ['hat','mle','mm']:
        cp_alpha = 0
        cp_beta = 0
        for s in range(S):
            s = s+1
            # calculate the confidence interval by different method
            # alpha
            alpha_length = critical_value*np.sqrt(df_est_ci.loc[i,'sigma11_'+ estimator +'_s'+str(s)])
            alpha_lower = df_est_point.loc[i, 'alpha_'+ estimator +'_s'+str(s)] - alpha_length
            alpha_upper = df_est_point.loc[i, 'alpha_'+ estimator +'_s'+str(s)] + alpha_length
            # beta
            beta_length = critical_value*np.sqrt(df_est_ci.loc[i,'sigma22_'+ estimator +'_s'+str(s)])
            beta_lower = df_est_point.loc[i, 'beta_'+ estimator +'_s'+str(s)] - beta_length
            beta_upper = df_est_point.loc[i, 'beta_'+ estimator +'_s'+str(s)] + beta_length
            if alpha_lower< alpha < alpha_upper:
                cp_alpha = cp_alpha +1   
            if beta_lower< beta < beta_upper:
                cp_beta = cp_beta +1   
        df_est_ci_cp.at[i,'alpha_ci_'+estimator] = cp_alpha/S
        df_est_ci_cp.at[i,'beta_ci_'+estimator] = cp_beta/S


#%%

# plot the cp result from interval estimation for alpha

plt.rcParams["figure.figsize"] = (3.3,3)

df = df_est_ci_cp[(df_est_ci_cp['n']==50) & (df_est_ci_cp['alpha']==4.5)]
plt.plot(df.beta, df.beta_ci_hat, 'r-o', markerfacecolor='none', label = 'our method')
plt.plot(df.beta, df.beta_ci_mle, 'k--v', markerfacecolor='none', label = 'ML')
plt.plot(df.beta, df.beta_ci_mm,  'b--s', markerfacecolor='none', label = 'MM')
plt.plot([min(df.beta), max(df.beta)], [1-tail_probability, 1-tail_probability], 'g:', label = str(1-tail_probability))
plt.ylim(bottom=0.88)
plt.xticks(df.beta.tolist())
plt.legend()
plt.xlabel(r'$\beta$')
plt.ylabel('coverage probability')
plt.tight_layout()
    
#%%

# plot the cp result from interval estimation for beta

plt.rcParams["figure.figsize"] = (3.3,3)

df = df_est_ci_cp[(df_est_ci_cp['n']==50) & (df_est_ci_cp['beta']==4.5)]
df = df.sort_values(by='alpha', ascending=True)
plt.plot(df.alpha, df.alpha_ci_hat, 'r-o', markerfacecolor='none', label = 'our method')
plt.plot(df.alpha, df.alpha_ci_mle, 'k--v', markerfacecolor='none', label = 'ML')
plt.plot(df.alpha, df.alpha_ci_mm,  'b--s', markerfacecolor='none', label = 'MM')
plt.plot([min(df.alpha), max(df.alpha)], [1-tail_probability, 1-tail_probability], 'g:', label = str(1-tail_probability))
plt.ylim(bottom=0.88)
plt.xticks(df.alpha.tolist())
plt.legend()
plt.xlabel(r'$\alpha$')
plt.ylabel('coverage probability')
plt.tight_layout()
