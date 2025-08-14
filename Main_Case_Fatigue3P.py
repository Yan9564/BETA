# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 20:39:08 2024

@author: yanbi
"""

import matplotlib.pyplot as plt
import numpy as np
from Wiener_IG import Wiener_IG
from GammaDegra import GammaDegra
from IGDegra import IGDegra


import pandas as pd
from scipy.special import betaln
# import self defifned functions

from BetaProcessEstimateDiscrete3P import BetaProcessEstimateDiscrete3P
from BetaProcessEstimate3PNonlinear import BetaProcessEstimate3PNonlinear
from BetaProcessFPT import BetaProcessFPT

            
import numdifftools as nd
from scipy.special import beta as BETA_FUC

#%%

# load data
path_t = r'\data_fatigue_rescale_t'
path_x = r'\data_fatigue_rescale_x'
t = pd.read_excel(path_t+'.xlsx').to_numpy()[:,1:]
x = pd.read_excel(path_x+'.xlsx').to_numpy()[:,1:]


n = x.shape[0]
m = x.shape[1]
w = 2.8267105247915163 # this is from the Main_Case_Fatigue3P_preprocess

#%%

# plot the rescaled data

plt.rcParams["figure.figsize"] = (3,2.5)

for i in range(n):
    plt.plot(t[i][:], x[i][:], 'v--', color = 'tab:gray', markersize=4, linewidth=1)
    
plt.plot([0,14], [w, w], '-', color = 'red', label = 'failure threshold', linewidth=1)
   
plt.xlabel(r'cycle (x1,0000 times)')
plt.ylabel(r'fatigue length (inch)')
plt.legend()
plt.ylim([0,4])
plt.xlim([0,12])
plt.tight_layout()


#%%

# estimate model parameters in BETA process
tau = 1
num = 50
method = 'new-point'
alpha_hat, beta_hat, b_hat, SIGMA_hat, time_hat, idx_hat = BetaProcessEstimateDiscrete3P(x, t, method, num, tau)


# provide some simulated degradation path using the estimation results
alpha = alpha_hat
be = beta_hat
m = np.shape(x)[1]
N = n
# provide the initial values of degradation time and degradation levels
ini_values = np.zeros((n,2))
#%%
# estimate model parameters in BETA process
# interval estimation

(n,m) = x.shape
dx = np.zeros((n,m))
dt = np.zeros((n,m))
# calculate the degradation increment
for i in range(n):
    for j in range(1,m):
        dx[i][j] = x[i][j] - x[i][j-1]
        if dx[i][j]<0:
            dx[i][j] = 0
        else:
            dx[i][j] = dx[i][j]
            
        dt[i][j] = t[i][j] - t[i][j-1]
        if dt[i][j]<0:
            dt[i][j] = 0
        else:
            dt[i][j] = dt[i][j]
            

def beta_func(alpha_symbol, beta_symbol):
    return BETA_FUC(alpha_symbol, beta_symbol)


x_value = alpha_hat
y_value = beta_hat

first_dalpha = nd.Derivative(lambda alpha_symbol: beta_func(alpha_symbol, y_value), n=1)(x_value)
second_dalpha = nd.Derivative(lambda alpha_symbol: beta_func(alpha_symbol, y_value), n=2)(x_value)
zero_dalpha = beta_func(x_value, y_value)


first_dbeta = nd.Derivative(lambda beta_symbol: beta_func(x_value, beta_symbol), n=1)(y_value)
second_dbeta = nd.Derivative(lambda beta_symbol: beta_func(x_value, beta_symbol), n=2)(y_value)
zero_dbeta = beta_func(x_value, y_value)

NM = np.count_nonzero(~np.isnan(dx))
var_alpha = second_dalpha/zero_dalpha-first_dalpha**2/zero_dalpha**2
var_alpha = var_alpha*m*n
var_alpha = 1/var_alpha
var_beta = second_dbeta/zero_dbeta-first_dbeta**2/zero_dbeta**2
var_beta = var_beta*m*n
var_beta = 1/var_beta
var_b = (1-beta_hat)*np.nansum(1/(b_hat-dx)**2) + NM*(alpha_hat+beta_hat-1)/b_hat**2
var_b = -1/var_b

#%%
# reliablity curves for the proposed model (linear case)
# with interval estimation
S = 1000 # number of samples from the interval estimates of model parameters
m = 50
R_mcmc = np.zeros((S,m))
for s in range(S):
    alpha_hat_s =  np.random.normal(alpha_hat, np.sqrt(var_alpha))
    beta_hat_s =  np.random.normal(beta_hat, np.sqrt(var_beta))
    b_hat_s =  np.random.normal(b_hat, np.sqrt(var_b))
    CDF_s, T_s = BetaProcessFPT(w, alpha_hat_s, beta_hat_s, b_hat_s, tau, m, discrete=1, linear=1, params=[1,1])
    R_s = 1-np.array(CDF_s)
    R_mcmc[s,:] = R_s
    
# calculate the reliability
lower_quantiles = np.percentile(R_mcmc, 2.5, axis=0) 
upper_quantiles = np.percentile(R_mcmc, 97.5, axis=0) 
#%%
# the proposed method (Nonlinear Case)
# the point estimation for the reliability

# step 1: estimate the model parameters in Beta degradation process
initial_params = [alpha_hat, beta_hat, b_hat, 1]
alpha_hat_nl, beta_hat_nl, b_hat_nl, r_hat_nl, lnL_beta_nl = BetaProcessEstimate3PNonlinear(x, t, initial_params)
print('alpha_hat_nl, beta_hat_nl, b_hat_nl, r_hat_nl :')
print(f"{alpha_hat_nl:.4f}, {beta_hat_nl:.4f}, {b_hat_nl:.4f}, {r_hat_nl:.4f}")
# step 2: build the FPT distribution using the proposed method
CDF_pro, T = BetaProcessFPT(w, alpha_hat, beta_hat, b_hat, tau, m, discrete=1, linear=1, params=[1,1])
# step 3: build the FPT distirbution using the empirical distribution (ED)
# and then evalue the proposed method with the ED method
# calculate the reliability
R_pro = 1-np.array(CDF_pro)

AIC_beta_nl = -2*lnL_beta_nl+2*4

print('lnL_beta_nl :', lnL_beta_nl)
print('AIC_beta_nl :', AIC_beta_nl)


#%%

# the proposed method 
# the point estimation for the reliability


# step 1: estimate the model parameters in Beta degradation process
num = 20
method = 'new-point'                    
alpha_hat, beta_hat, b_hat, SIGMA_hat, time_hat, idx_hat = BetaProcessEstimateDiscrete3P(x, t, method, num, tau)
# step 2: build the FPT distribution using the proposed method
m = 50
CDF_pro, T = BetaProcessFPT(w, alpha_hat, beta_hat, b_hat, tau, m, discrete=1, linear=1, params=[1,1])
# step 3: build the FPT distirbution using the empirical distribution (ED)
# and then evalue the proposed method with the ED method
# calculate the reliability
R_pro = 1-np.array(CDF_pro)

def neg_log_likelihood(params, x):
    alpha, be, b = params
    n = x.shape[0]
    m = x.shape[1]
    dx = np.zeros((n,m))
    for i in range(n):
        for j in range(1,m):
            dx[i][j] = x[i][j] - x[i][j-1]
    dx = dx[dx>0]
    x2, x1 = np.sort(dx.flatten())[-2:]
    dx = dx[dx>0]
    N = len(dx)
    term2 = (alpha - 1) * np.sum(np.log(dx))
    term3 = (be - 1) * np.sum(np.log(b - dx))
    term4 = -N * (alpha + be - 1) * np.log(b)
    term5 = -N * betaln(alpha, be)
    return (term2 + term3 + term4 + term5)

lnL_beta = neg_log_likelihood([alpha_hat, beta_hat, b_hat], x)
AIC_beta = -2*lnL_beta+2*3
print('lnL_beta :', lnL_beta)
print('AIC_beta :', AIC_beta)


#%%

# benchmark : Wiener process
results_WienerIG = Wiener_IG(x, t, w, T)
CDF_WienerIG = results_WienerIG[2]
R_WienerIG = 1-np.array(CDF_WienerIG)
lnL_Wiener = results_WienerIG[3]
AIC_Wiener = results_WienerIG[4]

results_GammaDegra = GammaDegra(x, t, w, T)
CDF_GammaDegra = results_GammaDegra[2]
R_GammaDegra = 1-np.array(CDF_GammaDegra)
lnL_Gamma = results_GammaDegra[3]
AIC_Gamma = results_GammaDegra[4]

results_IGDegra = IGDegra(x, t, w, T)
CDF_IGDegra = results_IGDegra[2]
R_IGDegra = 1-np.array(CDF_IGDegra)
lnL_IG = results_IGDegra[3]
AIC_IG = results_IGDegra[4]


#%%

# compare the PFT distributions of proposed Beta process, Wiener, Gamma and IG


plt.rcParams["figure.figsize"] = (4,2.5)
fig = plt.figure(dpi=300)


# plot the FPT
plt.plot(T, R_pro, linestyle='-', linewidth=0.6, marker='o', markerfacecolor='none', label=r'$\mathcal{M}_{D}^{Beta}$ Point Est.', color="black")
plt.plot(T, lower_quantiles, linestyle='--', linewidth=0.6, label=r'$\mathcal{M}_{D}^{Beta}$ 95% CI', color="black")
plt.plot(T, upper_quantiles, linestyle='--', linewidth=0.6, color="black")
plt.plot(T, R_WienerIG, linestyle='-', linewidth=0.6, marker='*', markerfacecolor='none', label=r'$\mathcal{M}_{D}^{Wiener}$', color="green")
plt.plot(T, R_GammaDegra, linestyle='-', linewidth=0.6, marker='v', markerfacecolor='none', label=r'$\mathcal{M}_{D}^{Gamma}$', color="blue")
plt.plot(T, R_IGDegra, linestyle='-', linewidth=0.6, marker='p', markerfacecolor='none', label=r'$\mathcal{M}_{D}^{IG}$', color="red")

plt.xlabel(r'cycle (x1,0000 times)')
plt.ylabel(r'reliability')
plt.ylim([-0.05,1.05])
plt.xlim([0,18])
plt.legend()
plt.tight_layout()


