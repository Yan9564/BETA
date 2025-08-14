# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 21:32:41 2023

@author: yanbi
"""

import numpy as np
import matplotlib.pyplot as plt

def ProbabilityPlot(df, n_censored, save_flag, save_path, save_name, x_min, x_max):
    plt.figure()
    plt.rcParams["figure.figsize"] = (2.6,2.6)
    L = df.sort_values(by='L', ascending=True)
    n = df.size
    F = []
    Y = []
    for i in range(n):
        F = (i+1)/(n+n_censored+1)
        Y.append(np.log(1/(1-F)))
    # Fit the line
    slope, intercept = np.polyfit(L.values.reshape(-1), Y, 1)
    # Generate fitted Y values
    fitted_L = np.array([i for i in range(0, int(np.max(L)+10))])
    fitted_Y = slope * fitted_L + intercept
    
    # plot the probability plot    
    plt.plot(L, Y, 'k v', label = 'failure time')
    plt.plot(fitted_L, fitted_Y, 'r--', label = 'fitted line')
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlim([x_min, x_max])
    plt.xlabel(r'failure time $t$ (hours)')
    plt.ylabel(r'$-\ln \left[ 1-\hat{F}(t_{(i)}) \right]$')
    plt.tight_layout()
    if save_flag == 1:
        plt.savefig(save_path+save_name+'.eps', dpi=700)
        print('==================================')
        print('Figure saved')
    elif save_flag == 0:
        print('==================================')    
        print('Figure is not saved')
    return L, Y