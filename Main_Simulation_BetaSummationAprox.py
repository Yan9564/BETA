# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:24:31 2024

@author: yanbi
"""


import numpy as np

# Define the shape parameters
for m in [2,5,10,15,50]:
    print('----------------- m = ', (m))
    for (a, b, scale) in [(4.5, 5.5, 0.6), (4.5, 6.0, 0.6), 
                          (4.5, 6.5, 0.6), (4.5, 6.5, 1.0)]:
        S = 10000
        Y_naive = []
        Y_propo = []
        # Generate 10,000 samples
        for s in range(S):
            Ws = np.random.beta(a, b, m)*scale
            Y_naive.append(np.sum(Ws))
            
        alpha = a*(m+(m-1)/(a+b))
        beta = b*(m+(m-1)/(a+b))
        Y_propo.append(m*scale*np.random.beta(alpha, beta, S))


        # Calculate percentiles
        p_naive = np.percentile(Y_naive, 95) 
        p_propo = np.percentile(Y_propo, 95) 

        
        print(f" & ${p_naive:.4f}$ & ${p_propo:.4f}$")

