# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 19:47:19 2024

@author: yanbi
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import self defifned functions
from BetaProcessEstimateDiscrete3PIteration import BetaProcessEstimateDiscrete3PIteration

#%%

# load data
led_index = 1

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

# change the time interval
# Generate one row with values from 0 to 38
row = np.linspace(0, t.shape[1]-1, t.shape[1])

# Create an array with 8 identical rows
array = np.tile(row, (t.shape[0], 1))

t = array
x = x*10

#%%

# estimate model parameters in BETA process
tau = 1
num = 30
method = 'new-point'
alpha_est, beta_est, b_est, alpha_hat, beta_hat, b_hat, likeli1 = BetaProcessEstimateDiscrete3PIteration(x, t, num, tau)
indices = np.arange(num)
index = 27
#%%

plt.rcParams["figure.figsize"] = (2.8,1.8)

plt.plot(indices[1:], alpha_hat[1:], marker='o', markersize=3, c='blue', markerfacecolor='none', linewidth=0.8) 
plt.plot(indices[index], alpha_hat[index], marker='*', markersize=6, linewidth=0.8, c='red')
plt.xlabel('number of iterations')
plt.ylabel('estimates')
plt.tight_layout()

#%%

plt.plot(indices[1:], beta_hat[1:], marker='o', markersize=3, c='blue', markerfacecolor='none', linewidth=0.8) 
plt.plot(indices[index], beta_hat[index], marker='*', markersize=6, linewidth=0.8, c='red') 
plt.xlabel('number of iterations')
plt.ylabel('estimates')
plt.tight_layout()

#%%

plt.plot(indices[1:], b_hat[1:], marker='o', markersize=3, c='blue', markerfacecolor='none', linewidth=0.8)
plt.plot(indices[index], b_hat[index], marker='*', markersize=6, linewidth=0.8, c='red') 
plt.xlabel('number of iterations')
plt.ylabel('estimates')
plt.tight_layout()


#%%


plt.plot(indices[1:], likeli1[1:], marker='o', markersize=3, c='blue', markerfacecolor='none', linewidth=0.8)
plt.plot(indices[index], likeli1[index], marker='*', markersize=6, c='red', linewidth=0.8, label = 'max likelihood')
plt.xlabel('number of iterations')
plt.ylabel('likelihoods')
plt.legend()
plt.tight_layout()


