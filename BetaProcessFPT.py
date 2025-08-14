# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:34:09 2023

@author: yanbi

This is for returning the approximated FPT distribution of Beta process 
given the model parameters in the process.
"""

from LambdaTimeScale import LambdaTimeScale as Lambda
import scipy.special as sc

def BetaProcessFPT(w, alpha, be, scale, tau, m, discrete=1, linear=1, params=[1,1]):
    CDF = [0]
    T = [0]
    # ----------------------------------------------------------------------------
    # this is the discrete-linear Beta process
    if discrete == 1 and linear == 1:
        for j in range(1,m):
            J = j
            a = alpha*J + alpha*(J-1)/(alpha+be)
            b = be*J + be*(J-1)/(alpha+be)
            T.append(J*tau)
            if J*scale < w:
                CDF.append(0)
            else:
                CDF.append(1-sc.betainc(a, b, w/(J*scale)))
    # ----------------------------------------------------------------------------
    # this is the discrete-nonlinear Beta process
    if discrete == 1 and linear == 0:
        for j in range(1,m):
            J = Lambda(j,params)
            a = alpha*J + alpha*(J-1)/(alpha+be)
            b = be*J + be*(J-1)/(alpha+be)
            T.append(j*tau)
            if J < w:
                CDF.append(0)
            else:
                CDF.append(1-sc.betainc(a, b, w/(J*scale)))
    
    return CDF,T


