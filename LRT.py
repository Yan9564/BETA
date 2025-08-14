# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 21:32:06 2023
@author: yanbi
"""

import numpy as np
from scipy.stats import chi2
from scipy.stats import expon
from scipy.stats import weibull_min

def LRT(alpha, df, method, dis):
    
    print('------------------------------')
    
    # the first method of calculate the statistic
    """
    x_min = df.L.min()
    n = df.size
    Test = 2*n*np.log(1+n*x_min/(np.sum(df.L)-n*x_min))
    """
    
    # the second method of calculate the statistic
    data = df.L
    if dis == 'expo2':
        log_likelihood1 = np.sum(expon.logpdf(data, expon.fit(data)[0], expon.fit(data)[1]))
        log_likelihood0 = np.sum(expon.logpdf(data, 0, expon.fit(data, floc=0)[1]))
    elif dis == 'weil3':
        params_weibull0 = weibull_min.fit(data, floc=0, method = 'MLE')
        print('params_weibull0 :', params_weibull0)
        params_weibull1 = weibull_min.fit(data, floc=np.min(data)+1, method = 'MLE')
        print('params_weibull1 :', params_weibull1)
        log_likelihood0 = np.sum(weibull_min.logpdf(data, 
                                                    c = params_weibull0[0], 
                                                    loc = 0, 
                                                    scale = params_weibull0[2]))
        log_likelihood_weibull = weibull_min.logpdf(data, 
                                                    c = params_weibull1[0], 
                                                    loc = params_weibull1[1], 
                                                    scale = params_weibull1[2])
        print(log_likelihood_weibull)
        log_likelihood1 = np.nansum(log_likelihood_weibull[np.isfinite(log_likelihood_weibull)])
    Test = 2*log_likelihood1-2*log_likelihood0
    
    # Degrees of freedom, in your case it's 1
    degrees_of_freedom = 1
    chi_square = chi2.ppf(1-alpha, degrees_of_freedom)
    # Calculate the critical chi-square value
    if method == 'proposed':
        chi_square = chi_square/2
        p_value = chi2.sf(Test, degrees_of_freedom)
    elif method == 'naive':
        p_value = chi2.sf(Test, degrees_of_freedom)
    
    print(f"Test statistic is : {Test:.4f}")
    print(f"Chi-square value for {(1-alpha)*100:.4f}% confidence level: {chi_square:.4f}")
    
    print(f"p value is : {p_value:.10f}")
    
    if Test<=chi_square:
        TestResult = 'accept'
    else:
        TestResult = 'reject'
    
    """
    if alpha<p_value:
        TestResult = 'accept'
    else:
        TestResult = 'reject'
    """
        
    print(f"Test result: {TestResult}")
    return TestResult