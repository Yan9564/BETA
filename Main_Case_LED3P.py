# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:15:51 2024

@author: yanbi
"""


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

#%%

# load data
led_index = 2

if led_index == 1:
    path_t = r'\data_led1_t'
    path_x = r'\data_led1_x'
elif led_index == 2:
    path_t = r'\data_led2_t'
    path_x = r'\data_led2_x'
    
t = pd.read_excel(path_t+'.xlsx').to_numpy()[:,1:]
x = pd.read_excel(path_x+'.xlsx').to_numpy()[:,1:]


zeros_column = np.zeros((t.shape[0], 1))

# Concatenate the zeros column before the data in t
t = np.hstack((zeros_column, t))
x = np.hstack((zeros_column, x))

n = x.shape[0]
m = x.shape[1]
w = 0.25 

#%%

# rescale the time interval
row = np.linspace(0, t.shape[1]-1, t.shape[1])
# Create an array with 8 identical rows
array = np.tile(row, (t.shape[0], 1))
t = array
x = x*10

#%%
# the proposed method (Linear Case)
# estimate model parameters in BETA process
tau = t[0,1]
num = 20
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
# the proposed method (Nonlinear Case)
# the point estimation for the reliability

# step 1: estimate the model parameters in Beta degradation process
initial_params = [alpha_hat, beta_hat, b_hat, tau]
alpha_hat_nl, beta_hat_nl, b_hat_nl, r_hat_nl, lnL_beta_nl = BetaProcessEstimate3PNonlinear(x, t, initial_params)
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
