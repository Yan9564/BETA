import pandas as pd

from LRT import LRT
from ProbabilityPlot import ProbabilityPlot
from FitExp import FitExp

method = 'proposed'
# method = 'naive'
           
# test on the real data
import matplotlib


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


save_path = r'path'
alpha = 0.05

#%%

save_flag = 0
n_censored = 0
path = r'\data_lifetime_HFB-CE03F_373'
df = pd.read_excel(path+'.xlsx')
dis = 'expo2'
TestResult = LRT(alpha, df, method, dis)
lamb, theta = FitExp(df)
save_name = '\data_lifetime_HFB-CE03F_373'
x_min = 7600
x_max = 7900
L, Y  = ProbabilityPlot(df, n_censored, save_flag, save_path, save_name, x_min, x_max)



#%%
save_flag = 0
n_censored = 90
path = r'\data_lifetime_deviceA - 40'
df = pd.read_excel(path+'.xlsx')
dis = 'expo2'
TestResult = LRT(alpha, df, method, dis)
lamb, theta = FitExp(df)
save_name = '\data_lifetime_deviceA'
x_min = 0
x_max = 5000
L, Y  = ProbabilityPlot(df, n_censored, save_flag, save_path, save_name, x_min, x_max)


