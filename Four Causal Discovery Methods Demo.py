import numpy as np
import pandas as pd
import statsmodels.api as sm
import pyinform
import seaborn as sns
import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.parcorr_wls import ParCorrWLS
from tigramite import plotting as tp
from ccm import ccm
np.random.seed(123)
n = 8000
x1 = np.zeros(n + 1)
x2 = np.zeros(n + 1)
x3 = np.zeros(n + 1)
x4 = np.zeros(n + 1)
x5 = np.zeros(n + 1)


for i in range(1, n + 1):
    x1[i] = 0.95 * np.sqrt(2) * x1[i - 1] - 0.9025 * x1[i - 1] + np.random.normal(0, 1)
    x2[i] = 0.5 * x1[i - 1] + np.random.normal(0, 1)
    x3[i] = -0.4 * x1[i - 1] + np.random.normal(0, 1)
    x4[i] = -0.5 * x1[i - 6] + 0.25 * np.sqrt(2) * x4[i - 1] + 0.25 * np.sqrt(2) * x5[i - 1] + np.random.normal(0, 1)
    x5[i] = -0.25 * np.sqrt(2) * x4[i - 1] + 0.25 * np.sqrt(2) * x5[i - 1] + np.random.normal(0, 1)

x1 = x1[1:]
x2 = x2[1:]
x3 = x3[1:]
x4 = x4[1:]
x5 = x5[1:]

df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5})
start_date = '2023-01-01'

date_range = pd.date_range(start=start_date, periods=len(df), freq='H')

# 将日期范围设置为DataFrame的索引
df.index = date_range


dfdata = df

min_val = dfdata.min().min()
if min_val < 0:
    df += abs(min_val)

max_val = dfdata.max().max()
df_normalized = dfdata / max_val
linear_system = df_normalized
#%%
def linear_gc(X, Y, max_delay):
    max_gc = -np.inf
    best_delay = 0
    
    for delay in range(1, max_delay + 1):
        X_now = X[:-delay]
        Y_now = Y[:-delay]
        Y_fut = Y[delay:]

        regression_uni = sm.OLS(Y_fut, sm.add_constant(Y_now)).fit()
        regression_mult = sm.OLS(Y_fut, sm.add_constant(np.column_stack([Y_now, X_now]))).fit()
        var_eps_uni = regression_uni.resid.var()
        var_eps_mult = regression_mult.resid.var()
        gc = np.log(var_eps_uni / var_eps_mult)
        
        if gc > max_gc:
            max_gc = gc
            best_delay = delay
    
    return max_gc, best_delay

def calculate_entropy(prob):
    return -np.nansum(prob * np.log2(prob + np.finfo(float).eps))
def calc_te(X, Y, k):
    return pyinform.transferentropy.transfer_entropy(X.astype(int), Y.astype(int), k)
def calculate_transfer_entropy(X, Y, max_delay):
    Nbins = 10
    L = len(X)
    max_te = -np.inf
    best_delay = 0
    
    combined_data = np.hstack((X, Y))
    max_val = np.max(combined_data)
    binsize = (max_val + 1) / Nbins
    bins_w = np.linspace(0, max_val + 1, Nbins + 1)
    
    for delay in range(1, max_delay + 1):
        Py = np.zeros(Nbins)
        Pypr_y = np.zeros((Nbins, Nbins))
        Py_x = np.zeros((Nbins, Nbins))
        Pypr_y_x = np.zeros((Nbins, Nbins, Nbins))
        
        rn_ypr = np.digitize(Y[delay:], bins_w) - 1
        rn_y = np.digitize(Y[:-delay], bins_w) - 1
        rn_x = np.digitize(X[:-delay], bins_w) - 1
        
        for k in range(L - delay):
            Py[rn_y[k]] += 1
            Pypr_y[rn_ypr[k], rn_y[k]] += 1
            Py_x[rn_y[k], rn_x[k]] += 1
            Pypr_y_x[rn_ypr[k], rn_y[k], rn_x[k]] += 1
        
        Py /= (L - delay)
        Pypr_y /= (L - delay)
        Py_x /= (L - delay)
        Pypr_y_x /= (L - delay)
        
        Hy = calculate_entropy(Py)
        Hypr_y = calculate_entropy(Pypr_y.ravel())
        Hy_x = calculate_entropy(Py_x.ravel())
        Hypr_y_x = calculate_entropy(Pypr_y_x.ravel())
        
        te = Hypr_y + Hy_x - Hy - Hypr_y_x
        
        if te > max_te:
            max_te = te
            best_delay = delay
    
    return max_te, best_delay

def fapply_pairwise(data, func, max_delay):
    n_cols = data.shape[1]
    result_values = np.zeros((n_cols, n_cols))
    result_delays = np.zeros((n_cols, n_cols))
    result_annotations = np.empty((n_cols, n_cols), dtype=object)
    
    for i in range(n_cols):
        for j in range(n_cols):
            max_val, best_delay = func(data[:, i], data[:, j], max_delay)
            result_values[i, j] = max_val
            result_delays[i, j] = best_delay
            result_annotations[i, j] = f"{max_val:.2f}/{best_delay}"

    return result_values, result_delays, result_annotations

max_delay = 8

gc_values, gc_delays, gc_annotations = fapply_pairwise(linear_system.values, linear_gc, max_delay)
te_values, te_delays, te_annotations = fapply_pairwise(linear_system.values, calculate_transfer_entropy, max_delay)

var_names = ["x1", "x2", "x3", "x4", "x5"]
gc_df_values = pd.DataFrame(gc_values, index=var_names, columns=var_names)
te_df_values = pd.DataFrame(te_values, index=var_names, columns=var_names)
#%%
import matplotlib.patches as patches

font_size = 20
annot_kws = {"size": 20}
shrink = 0.005
highlight_cells = [(0, 1), (0, 2), (0, 3), (3, 4), (4, 3)]

# Granger Causality Heat Map
plt.figure(figsize=(10, 8))
ax = sns.heatmap(gc_df_values, annot=gc_annotations, fmt="", cmap='Blues', cbar=True, annot_kws=annot_kws)
plt.xlabel('Effect', fontsize=font_size)
plt.ylabel('Cause', fontsize=font_size)
plt.title('Granger Causality Results ($Strength/Delay$)', fontsize=font_size)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Actual Causal Relationship
for (row, col) in highlight_cells:
    rect = patches.Rectangle(
        (col + shrink, row + shrink),      
        1 - 2 * shrink,                    
        1 - 2 * shrink,                   
        linewidth=3,
        edgecolor='darkred',
        facecolor='none'
    )
    ax.add_patch(rect)

plt.show()

# Transfer Entropy Heat Map
plt.figure(figsize=(10, 8))
ax = sns.heatmap(te_df_values, annot=te_annotations, fmt="", cmap='Blues', cbar=True, annot_kws=annot_kws)
plt.xlabel('Effect', fontsize=font_size)
plt.ylabel('Cause', fontsize=font_size)
plt.title('Transfer Entropy Results ($Strength/Delay$)', fontsize=font_size)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

for (row, col) in highlight_cells:
    rect = patches.Rectangle((col, row), 1, 1, linewidth=3, edgecolor='darkred', facecolor='none')
    ax.add_patch(rect)

plt.show()
#%%
#PCMCI  
data_values = linear_system.values
time_indices = np.arange(len(linear_system))
variable_names = linear_system.columns.tolist() 
tigramite_data = pp.DataFrame(data_values, datatime=time_indices, var_names=variable_names)

parcorr = RobustParCorr(significance='analytic')
pcmci = PCMCI(
    dataframe=tigramite_data, 
    cond_ind_test=parcorr,
    verbosity=1)

tau_max = 8
pc_alpha = 0.01
pcmci.verbosity = 2

results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)

link_matrix = results['p_matrix'].copy()
link_matrix[link_matrix < 0.001] = 1  
link_matrix[link_matrix != 1] = 0

print(link_matrix)
tp.plot_graph(
    val_matrix=results['val_matrix'],
    graph=results['graph'],
    var_names=variable_names,
    link_colorbar_label='cross-MCI',
    node_colorbar_label='auto-MCI',
    show_autodependency_lags=False
)

plt.title('Causal graph based on PCMCI+ RobustParCorr')
plt.show()
#%%
df_CCM = df
tau = 1      
E = 3        
L = len(df_CCM)  
maxdelay = 5
ccm_results = np.zeros((len(df_CCM.columns), len(df_CCM.columns)))
delay_matrix = np.zeros((len(df_CCM.columns), len(df_CCM.columns)))


for i, col_i in enumerate(df_CCM.columns):
    for j, col_j in enumerate(df_CCM.columns):
        if i != j:
            best_causality = -np.inf  
            best_delay = np.nan
            for d in range(1, maxdelay + 1):
                current_length = L - d  
                if current_length <= 0:
                    continue
                
                ccm_ij = ccm(df_CCM[col_i][:-d].reset_index(drop=True),
                             df_CCM[col_j][d:].reset_index(drop=True),
                             tau, E, current_length)
                causality_val = ccm_ij.causality()[0]                 
                if causality_val > best_causality:
                    best_causality = causality_val
                    best_delay = d
            ccm_results[i, j] = best_causality
            delay_matrix[i, j] = best_delay
        else:
            ccm_results[i, j] = np.nan
            delay_matrix[i, j] = np.nan

annotations = np.empty_like(ccm_results, dtype=object)
for i in range(ccm_results.shape[0]):
    for j in range(ccm_results.shape[1]):
        if i == j or np.isnan(ccm_results[i, j]):
            annotations[i, j] = ""
        else:
            annotations[i, j] = f"{ccm_results[i, j]:.2f}/{int(delay_matrix[i, j])}"

font_size = 20  
annot_kws = {"size": 20}  
plt.figure(figsize=(10, 8))
sns.heatmap(ccm_results, annot=annotations, fmt="", cmap='Blues', cbar=True, annot_kws=annot_kws)
plt.xticks(np.arange(len(df_CCM.columns)) + 0.5, labels=df_CCM.columns, fontsize=14)
plt.yticks(np.arange(len(df_CCM.columns)) + 0.5, labels=df_CCM.columns, fontsize=14)
plt.xlabel('Effect', fontsize=font_size)
plt.ylabel('Cause', fontsize=font_size)
plt.title('CCM Causality Results ($Strength/Delay$)', fontsize=font_size)
plt.show()