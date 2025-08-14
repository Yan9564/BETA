# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 21:33:11 2023
@author: yanbi
"""

import numpy as np

def FitExp(df):
    L = df.L
    n = len(L)
    theta = np.min(L)
    lamb = n/(np.sum(L)-n*theta)
    print('E[L]:', np.mean(L))
    print('lambda:',lamb)
    print('theta:', theta)
    print('failure_free/EL:',theta/(theta+1/lamb)*100)
    return lamb, theta
