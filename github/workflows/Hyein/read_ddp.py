import json
file_path = "D:\pykan\.ignore\kan_ddp\kan_ddp_202510\scan_result_KAN.json"

with open(file_path, 'rb') as f:
    d = json.load(f)

figure_path = r"D:\pykan\kan\experiments\figures"
#%
import pandas as pd

df = pd.DataFrame(d)
params_df = df['params'].apply(pd.Series)
params_df.columns = [f'param_{col}' for col in params_df.columns]
df_expanded = pd.concat([df.drop(['params'], axis=1), params_df], axis=1)

df_expanded['data_name'] = df_expanded['data_name'].astype('category')
data_categories = df_expanded['data_name'].cat.categories
#%%
import matplotlib.pyplot as plt
import numpy as np

data_name = 'AgNP'
df_specific_data = df_expanded[df_expanded['data_name'] == data_name]
params = list(params_df.columns)

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
axs = axs.flatten()
for idx_param, param in enumerate(params):
    ax = axs[idx_param]
    ax.scatter(np.log10(df_specific_data[param]), df_specific_data['mse_train'], color='blue', alpha=0.7, edgecolors='k',
               label='train')
    ax.scatter(np.log10(df_specific_data[param]), df_specific_data['mse_test'], color='red', alpha=0.7, edgecolors='k',
               s=8, label='test')
    ax.set_title("log " + param)
    ax.set_ylim(-0.2, 1.2)
for idx_param, param in enumerate(['seed', 'effective_input_dim', 'complexity']):
    ax = axs[idx_param+6]
    ax.scatter(df_specific_data[param], df_specific_data['mse_train'], color='blue', alpha=0.7, edgecolors='k',
               label='train')
    ax.scatter(df_specific_data[param], df_specific_data['mse_test'], color='red', alpha=0.7, edgecolors='k',
               s=8, label='test')
    ax.set_title(param)
    ax.set_ylim(-0.2, 1.2)
plt.savefig(f"{figure_path}\\param_scan_{data_name}.png")
plt.show()
