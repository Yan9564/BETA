import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from joblib import Parallel, delayed
import os

# import some self defined function
from ComparisonSimulation import ComparisonSimulation

#%%
# Set the default font to Times New Roman
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"})

# %%

'''
simulation section 
'''

# path of saving the results
save_path = r'path'

# simulate bete process
tau = 1
alpha = 4.5
be = 10
b = 0.6
m = 100
n = 100
w = 7.5
ini_values = np.zeros((n,2))
for i in range(n):
    # the first place saves the initial value of degradation time
    ini_values[i][0] = 0
    ini_values[i][1] = 0
    
#%%
# Parallel Simulation
# lifetime simulation

def run_one_rep(s, tau, alpha, be, b, m, n, w, ini_values, save_dir, max_retry=100):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    data_fit = 'LifeData'
    tries = 0
    while True:
        resultsL, flag_iteration_L = ComparisonSimulation(tau, alpha, be, b, m, n, w, data_fit, ini_values)
        if flag_iteration_L != 2:
            out_path = os.path.join(save_dir, f"resultsL_rep{s}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(resultsL, f)
            return s, out_path
        tries += 1
        if tries >= max_retry:
            return s, None

def main():
    S = 1000
    save_dir = r'F:\12-Frechect-TANG\results-v4\simulation'
    os.makedirs(save_dir, exist_ok=True)

    start = time.time()
    results = Parallel(n_jobs=5, prefer="processes", verbose=10)(
        delayed(run_one_rep)(s, tau, alpha, be, b, m, n, w, ini_values, save_dir)
        for s in range(1, S+1)
    )
    ok = sum(1 for _, p in results if p is not None)
    print(f"Saved {ok}/{S} results; elapsed {time.time()-start:.2f}s")

if __name__ == "__main__":
    main()
#%%
# Parallel Simulation
# degradation

def run_one_rep(s, tau, alpha, be, b, m, n, w, ini_values, save_dir, max_retry=100):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    data_fit = 'DegradationData'
    tries = 0
    while True:
        resultsD, flag_iteration_D = ComparisonSimulation(tau, alpha, be, b, m, n, w, data_fit, ini_values)
        if flag_iteration_D != 0:
            out_path = os.path.join(save_dir, f"resultsD_rep{s}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(resultsL, f)
            return s, out_path
        tries += 1
        if tries >= max_retry:
            return s, None

def main():
    S = 1000
    save_dir = r'path'
    os.makedirs(save_dir, exist_ok=True)

    start = time.time()
    results = Parallel(n_jobs=5, prefer="processes", verbose=10)(
        delayed(run_one_rep)(s, tau, alpha, be, b, m, n, w, ini_values, save_dir)
        for s in range(1, S+1)
    )
    ok = sum(1 for _, p in results if p is not None)
    print(f"Saved {ok}/{S} results; elapsed {time.time()-start:.2f}s")

if __name__ == "__main__":
    main()

# %% 

# compare the approximated PFT with the empirical PFT for one repliation of simulation
data_fit = 'LifeData'
resultsL, flag_iteration_L = ComparisonSimulation(tau, alpha, be, b, m, n, w, data_fit, ini_values)
data_fit = 'DegradationData'
resultsD, flag_iteration_D = ComparisonSimulation(tau, alpha, be, b, m, n, w, data_fit, ini_values)

# plot

plt.rcParams["figure.figsize"] = (4.8, 2.8)
fig = plt.figure(dpi=300)

T_dis = resultsL['T_dis']

# plot the FPT
plt.plot(T_dis, 1-np.array(resultsD['CDF_pro_dis']), linestyle='-', linewidth=0.6, marker='o', markerfacecolor='none', label=r'$\mathcal{M}^{Beta}_{D}$', color="black")
plt.plot(T_dis, 1-np.array(resultsD['CDF_true_dis']), linestyle='--', linewidth=0.6, marker='x', markerfacecolor='none', label='True', color="limegreen")
plt.plot(T_dis, 1-np.array(resultsL['CDF_emp_values']), linestyle=':', linewidth=0.6, marker='X', markerfacecolor='none', label=r'$\mathcal{M}^{ED}_{L}$', color="tab:pink")
plt.plot(T_dis, 1-np.array(resultsL['CDF_wbl2']), linestyle='--', linewidth=0.6, marker='*', markerfacecolor='none', label=r'$\mathcal{M}^{Wbl2}_{L}$', color="tab:green")
plt.plot(T_dis, 1-np.array(resultsL['CDF_wbl3']), linestyle='-.', linewidth=0.6, marker='v', markerfacecolor='none', label=r'$\mathcal{M}^{Wbl3}_{L}$', color="tab:blue")
plt.plot(T_dis, 1-np.array(resultsL['CDF_lognorm']), linestyle=':', linewidth=0.6, marker='p', markerfacecolor='none', label=r'$\mathcal{M}^{Ln}_{L}$', color="tab:red")
plt.plot(T_dis, 1-np.array(resultsL['CDF_expon']), linestyle='--', linewidth=0.6, marker='s', markerfacecolor='none', label=r'$\mathcal{M}^{Exp}_{L}$', color="tab:purple")
plt.plot(T_dis, 1-np.array(resultsL['CDF_expon2']), linestyle='--', linewidth=0.6, marker='.', markerfacecolor='none', label=r'$\mathcal{M}^{Exp2}_{L}$', color="tab:orange")


plt.xlabel(r'$t$')
plt.ylabel(r'$R(t)$')
plt.ylim([-0.05,1.05])
plt.legend()
plt.tight_layout()

# %%

# compare the approximated PFT with the empirical PFT for one repliation of simulation

plt.rcParams["figure.figsize"] = (4.8, 2.8)
fig = plt.figure(dpi=300)


# plot the FPT
plt.plot(T_dis, 1-np.array(resultsD['CDF_pro_dis']), linestyle='-', linewidth=0.6, marker='o', markerfacecolor='none', label=r'$\mathcal{M}^{Beta}_{D}$', color="black")
plt.plot(T_dis, 1-np.array(resultsD['CDF_true_dis']), linestyle='--', linewidth=0.6, marker='x', markerfacecolor='none', label='True', color="limegreen")
plt.plot(resultsD['T_dis'], 1-np.array(resultsD['CDF_Wiener']), linestyle='--', linewidth=0.6, marker='*', markerfacecolor='none', label=r'$\mathcal{M}^{Wiener}_{D}$', color="tab:green")
plt.plot(resultsD['T_dis'], 1-np.array(resultsD['CDF_GammaDegra']), linestyle='-.', linewidth=0.6, marker='v', markerfacecolor='none', label=r'$\mathcal{M}^{Gamma}_{D}$', color="tab:blue")
plt.plot(resultsD['T_dis'], 1-np.array(resultsD['CDF_IGDegra']), linestyle=':', linewidth=0.6, marker='p', markerfacecolor='none', label=r'$\mathcal{M}^{IG}_{D}$', color="tab:red")


plt.xlabel(r'$t$')
plt.ylabel(r'$R(t)$')
plt.ylim([-0.05,1.05])
plt.legend()
plt.tight_layout()
