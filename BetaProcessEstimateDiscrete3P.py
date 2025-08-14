import numpy as np
from scipy.optimize import minimize
import time
from scipy.special import betaln
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def BetaProcessEstimateDiscrete3P(x,t,method,num,tau):
    # find the sample size n and the number of measurements m
    idx = 0
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
    # build bootstrap samples
    # Number of bootstrap samples
    num_bootstrap_samples = 100
    alpha_bstrap = np.zeros((num_bootstrap_samples))
    beta_bstrap = np.zeros((num_bootstrap_samples))
    b_bstrap = np.zeros((num_bootstrap_samples))
    # Creating bootstrap samples
    # Each bootstrap sample will be a new 10x20 array, resampled with replacement from the original rows
    bootstrap_samples = np.array([np.random.choice(n, n, replace=True) for _ in range(num_bootstrap_samples)])
    bootstrap_data = dx[bootstrap_samples, :]
    bootstrap_time = dt[bootstrap_samples, :]
    SIGMA = np.zeros((3,3))
    
    def alpha_beta_scale(b, dx):
        # for the new estimator
        dx = dx[dx>0]
        N = len(dx)
        B1 = np.sum(1/(b-dx))
        B2 = np.sum(1/dx)
        beta_hat  = (N*B1-B1*B2*b)/(N*B1+N*B2-B1*B2*b)
        alpha_hat = (N*B2-B1*B2*b)/(N*B1+N*B2-B1*B2*b)
        return alpha_hat, beta_hat

    def b_check(dx, dt, b):
        # ensure nan is not happen
        # ensure that b is not exactly the largest value of dx
        if (np.nanmax(dx/dt) >= b):
            b = np.ma.masked_invalid(dx/dt).max()+0.07
        else:
            b = b 
        return b
    
    # Define the log-likelihood function for estimation in MLE
    def penalized_log_likelihood(params, dx):
        alpha, be, b = params
        if b <= np.max(dx):
            return np.inf
        x2, x1 = np.sort(dx.flatten())[-2:]
        dx = dx[dx>0]
        N = len(dx)
        term1 = np.log(b-x1) - np.log(b-x2)
        term2 = (alpha - 1) * np.sum(np.log(dx))
        term3 = (be - 1) * np.sum(np.log(b - dx))
        term4 = -N * (alpha + be - 1) * np.log(b)
        term5 = -N * betaln(alpha, be)
        return (term1 + term2 + term3 + term4 + term5)
    
    # Define the log-likelihood function for estimation in MLE
    def neg_log_likelihood(params, b, dx):
        alpha, be = params
        dx = dx[dx>0]
        bdx = b - dx
        bdx = bdx[bdx>0]
        N = len(bdx)
        term2 = (alpha - 1) * np.sum(np.log(dx))
        term3 = (be - 1) * np.sum(np.log(bdx))
        term4 = -N * (alpha + be - 1) * np.log(b)
        term5 = -N * betaln(alpha, be)
        return -(term2 + term3 + term4 + term5)
    
    def neg_penalized_log_likelihood(b, alpha_ast, beta_ast, dx):
        if b <= np.max(dx):
            return np.inf  
        x2, x1 = np.sort(dx.flatten())[-2:]/tau
        dx = dx[dx>0]
        N = len(dx)
        term1 = np.log(b - x1)
        term2 = -np.log(b - x2)
        term3 = (alpha_ast - 1) * np.sum(np.log(dx))
        term4 = (beta_ast - 1) * np.sum(np.log(b - dx))
        term5 = -N * (alpha_ast + beta_ast - 1) * np.log(b)
        term6 = -N * betaln(alpha_ast, beta_ast)
        return -(term1 + term2 + term3 + term4 + term5 + term6)
    
    def MM_estimator(dx, dt):
        dx = dx[dt>0]
        dt = dt[dt>0]
        # calculate the sample mean, and ensure that the samples are not zero, not negative
        dx_mean = np.mean(dx[dx > 0])
        dx_var = np.var(dx[dx > 0])
        # estimate alpha and beta using Method of Moments
        b_est = np.ma.masked_invalid(dx/dt).max()
        alpha_est = (b_est * dx_mean**2 - dx_mean**3)/(b_est * dx_var) - dx_mean
        beta_est = (b_est/dx_mean - 1) * alpha_est
        return alpha_est, beta_est, b_est
    
    def new_estimator(dx, dt):
        b_hat = np.zeros((num))
        alpha_hat = np.zeros((num))
        beta_hat = np.zeros((num))
        likeli1 = np.zeros((num))
        alpha_hat[0], beta_hat[0], b_hat[0] = MM_estimator(dx, dt)
        x2, x1 = np.sort(dx.flatten())[-2:]/tau
        for l in range(1,num):
            result = minimize(neg_penalized_log_likelihood,
                          x0=b_hat[l-1], 
                          args=(alpha_hat[l-1], beta_hat[l-1], dx),
                          bounds=[(x1, None)])
            b_hat[l] = result.x
            b_hat[l] = b_check(dx, dt, b_hat[l])
            alpha_hat[l], beta_hat[l] = alpha_beta_scale(b_hat[l], dx)
            likeli1[l] = penalized_log_likelihood([alpha_hat[l], beta_hat[l], b_hat[l]], dx)
        idx = np.argmax(likeli1[1:])+1
        alpha_est = alpha_hat[idx]
        beta_est = beta_hat[idx]
        b_est = b_hat[idx]
        return alpha_est, beta_est, b_est, idx
    
    def MLE_estimator(dx, dt):
        # estimate alpha and beta using Method of Moments
        alpha_mm, beta_mm, b_mm  = MM_estimator(dx, dt)
        initial_params = alpha_mm, beta_mm
        # Minimize the negative log-likelihood
        def callback(xk):
            # print(f'Objective function value at this iteration: {neg_log_likelihood(xk, b_mm, dx)}')
            xk = xk
        result = minimize(neg_log_likelihood, 
                          initial_params, 
                          args= (b_mm, dx), 
                          method='L-BFGS-B', 
                          bounds=[(0, None), (0, None)],
                          callback=callback)
        # MLEs for alpha and beta
        alpha_est, beta_est = result.x
        b_est = b_mm
        return alpha_est, beta_est, b_est
    
    if method == 'MM-point':
        start_time = time.perf_counter()
        # ------------------ Point estimation ------------------
        alpha_est, beta_est, b_est = MM_estimator(dx, dt)
        end_time = time.perf_counter()
        run_time = end_time - start_time
    elif method == 'MM-interval':
        start_time = time.perf_counter()
        # ------------------ Point estimation ------------------
        alpha_est, beta_est, b_est = MM_estimator(dx, dt)
        # ------------------Interval estimation ----------------  
        # Using bootstrap method
        for j in range(num_bootstrap_samples):
            dx_bstrap = bootstrap_data[j]
            dt_bstrap = bootstrap_time[j]
            alpha_bstrap[j], beta_bstrap[j], b_bstrap[j] = MM_estimator(dx_bstrap, dt_bstrap)  
        SIGMA[0,0] = np.var(alpha_bstrap)
        SIGMA[1,1] = np.var(beta_bstrap)
        SIGMA[2,2] = np.var(b_bstrap)
        end_time = time.perf_counter()
        run_time = end_time - start_time
    elif method == 'new-point':
        start_time = time.perf_counter()
        alpha_est, beta_est, b_est, idx = new_estimator(dx, dt)
        end_time = time.perf_counter()
        run_time = end_time - start_time
    elif method == 'new-interval':
        start_time = time.perf_counter()
        alpha_est, beta_est, b_est = new_estimator(dx, dt)
        # Using bootstrap method
        for j in range(num_bootstrap_samples):
            dx_bstrap = bootstrap_data[j]
            dt_bstrap = bootstrap_time[j]
            alpha_bstrap[j], beta_bstrap[j], b_bstrap[j] = new_estimator(dx_bstrap, dt_bstrap)           
        SIGMA[0,0] = np.var(alpha_bstrap)
        SIGMA[1,1] = np.var(beta_bstrap)
        SIGMA[2,2] = np.var(b_bstrap)
        end_time = time.perf_counter()
        run_time = end_time - start_time
    elif method == 'MLE-point':
        start_time = time.perf_counter()
        # ------------------ Point estimation ------------------
        alpha_est, beta_est, b_est = MLE_estimator(dx, dt)
        end_time = time.perf_counter()
        run_time = end_time - start_time
    elif method == 'MLE-interval':
        start_time = time.perf_counter()
        # ------------------ Point estimation ------------------
        alpha_est, beta_est, b_est = MLE_estimator(dx, dt)
        # ------------------Interval estimation ----------------  
        
        # Using bootstrap method
        for j in range(num_bootstrap_samples):
            dx_bstrap = bootstrap_data[j]
            dt_bstrap = bootstrap_time[j]
            alpha_bstrap[j], beta_bstrap[j], b_bstrap[j] = MLE_estimator(dx_bstrap, dt_bstrap)
        SIGMA[0,0] = np.var(alpha_bstrap)
        SIGMA[1,1] = np.var(beta_bstrap)
        SIGMA[2,2] = np.var(b_bstrap)
        end_time = time.perf_counter()
        run_time = end_time - start_time


    return alpha_est, beta_est, b_est, SIGMA, run_time, idx
