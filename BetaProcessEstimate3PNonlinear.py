# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 18:32:00 2024

This function is used for estimating the model parmaters in 3-para Beta distribution

@author: yanbi
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln

def finite_sum(arr):
    finite_values = arr[np.isfinite(arr)]
    return np.sum(finite_values)

def top_two_largest(arr):
    arr_copy = arr.copy()
    first_largest = np.nanmax(arr_copy)
    arr_copy[arr_copy == first_largest] = np.nan
    second_largest = np.nanmax(arr_copy)    
    return first_largest, second_largest


def log_likelihood(params, x, t):
    alpha, be, b, r = params
    n,m = x.shape
    dx = np.zeros((n,m))
    dlamb = np.zeros((n,m))
    dxt = np.zeros((n,m))
    for i in range(n):
        for j in range(1,m):
            dx[i][j] = x[i][j] - x[i][j-1]
            dlamb[i][j] = t[i][j]**r - t[i][j-1]**r
            dxt[i][j] = dx[i][j]/dlamb[i][j]
    dlamb = dlamb[dx>0]
    dxt = dxt[dx>0]
    dx = dx[dx>0]
    x2, x1 = top_two_largest(dxt.flatten())
    term1 = np.log(b-x1) - np.log(b-x2)
    term2 = (alpha*dlamb + alpha/(alpha+be)*(dlamb-1)-1) * np.log(dx)
    term2 = finite_sum(term2)
    term3 = (be*dlamb + be/(alpha+be)*(dlamb-1) -1) * np.log(b*dlamb-dx)
    term3 = finite_sum(term3)
    term4 = -((alpha+be+1)*dlamb-2) * np.log(b*dlamb)
    term4 = finite_sum(term4)
    term5 = - betaln(alpha*dlamb+alpha/(alpha+be)*(dlamb-1), be*dlamb + be/(alpha+be)*(dlamb-1))
    term5 = finite_sum(term5)
    likelihood = term1+ term2 + term3 + term4 + term5
    return likelihood

# Define the log-likelihood function 
def penalized_log_likelihood(params, x, t):
    alpha, be, b, r = params
    n,m = x.shape
    dx = np.zeros((n,m))
    dlamb = np.zeros((n,m))
    dxt = np.zeros((n,m))
    for i in range(n):
        for j in range(1,m):
            dx[i][j] = x[i][j] - x[i][j-1]
            dlamb[i][j] = t[i][j]**r - t[i][j-1]**r
            dxt[i][j] = dx[i][j]/dlamb[i][j]
    dlamb = dlamb[dx>0]
    dxt = dxt[dx>0]
    dx = dx[dx>0]
    x2, x1 = top_two_largest(dxt.flatten())
    term1 = np.log(b-x1) - np.log(b-x2)
    term1 = finite_sum(term1)
    term2 = (alpha*dlamb + alpha/(alpha+be)*(dlamb-1)-1) * np.log(dx)
    term2 = finite_sum(term2)
    term3 = (be*dlamb + be/(alpha+be)*(dlamb-1) -1) * np.log(b*dlamb-dx)
    term3 = finite_sum(term3)
    term4 = -((alpha+be+1)*dlamb-2) * np.log(b*dlamb)
    term4 = finite_sum(term4)
    term5 = - betaln(alpha*dlamb + alpha/(alpha+be)*(dlamb-1), be*dlamb + be/(alpha+be)*(dlamb-1))
    term5 = finite_sum(term5)
    likelihood = term1 + term2 + term3 + term4 + term5
    return -likelihood

def BetaProcessEstimate3PNonlinear(x,t, initial_params):
    # L: number of iterations
    # x: degradation data
    # Optimize to find the maximum likelihood estimates
    options={'disp': False, 'maxfun': 10}
    bounds=[(0,None),(0,None),(0,None),(0,None)]
    result = minimize(penalized_log_likelihood, 
                      initial_params, 
                      args=(x,t),
                      method = 'TNC', # TNC is ok
                      options=options,
                      bounds=bounds)
    alpha_est, beta_est, b_est, r_est = result.x
    # print("Optimization message:", result.message)
    lnL_beta_nl = log_likelihood(result.x, x, t)
    return alpha_est, beta_est, b_est, r_est, lnL_beta_nl




