# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 22:57:09 2024

@author: yanbi
"""

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (2,2.5)

# results from Main_Simulation_BetaSummationAprox
m_values = [2, 5, 10, 15, 50]
X1, Z1 = [0.7516, 1.6858, 3.1672, 4.6305, 14.5351], [0.7572, 1.6819, 3.1716, 4.6256, 14.5379]
X2, Z2 = [0.7258, 1.6104, 3.0375, 4.4215, 13.8621], [0.7211, 1.6114, 3.0263, 4.4300, 13.8763]
X3, Z3 = [0.6979, 1.5419, 2.9028, 4.2313, 13.2558], [0.6997, 1.5437, 2.8944, 4.2331, 13.2536]
X4, Z4 = [1.1543, 2.5840, 4.8332, 7.0568, 22.1057], [1.1677, 2.5736, 4.8399, 7.0477, 22.0962]

#%%

plt.plot(m_values, X1, '-o', c='blue', markerfacecolor='none', label="X(m)", linewidth=2)
plt.plot(m_values, Z1, ':x', c='red', markerfacecolor='none', label="Z(m)")
plt.xlabel(r"$m$")
plt.ylabel("$95\%$ percentiles")
plt.ylim([0,20])
plt.legend()
plt.tight_layout()
 
#%%


plt.plot(m_values, X2, '-o', c='blue', markerfacecolor='none', label="X(m)", linewidth=2)
plt.plot(m_values, Z2, ':x', c='red', markerfacecolor='none', label="Z(m)")
plt.xlabel(r"$m$")
plt.ylabel("$95\%$ percentiles")
plt.ylim([0,20])
plt.legend() 
plt.tight_layout()

#%%

plt.plot(m_values, X3, '-o', c='blue', markerfacecolor='none', label="X(m)", linewidth=2)
plt.plot(m_values, Z3, ':x', c='red', markerfacecolor='none', label="Z(m)")
plt.xlabel(r"$m$")
plt.ylabel("$95\%$ percentiles")
plt.ylim([0,20])
plt.legend()  
plt.tight_layout()

    
#%%


plt.plot(m_values, X4, '-o', c='blue', markerfacecolor='none', label="X(m)", linewidth=2)
plt.plot(m_values, Z4, ':x', c='red', markerfacecolor='none', label="Z(m)")
plt.xlabel(r"$m$")
plt.ylabel("$95\%$ percentiles")
plt.ylim([0,25])
plt.legend() 
plt.tight_layout()
