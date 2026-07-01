from kan.experiments.multkan_hparam_sweep_materials import sweep_multkan, evaluate_params
import numpy as np
import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from kan.custom_processing import remove_outliers_iqr
import json
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"This script is running on {device}.")
data_name = "P3HT"

dir_current = os.getcwd()
save_heading = os.path.join(dir_current, "github", "workflows", "Hyein", "multkan_sweep_autosave",
                            data_name + "_" + datetime.datetime.now().strftime('%Y%m%d_%H%M'))
filepath = os.path.join(dir_current, "github", "workflows", "Hyein", "data", f"{data_name}.csv")

filedata = pd.read_csv(filepath)
name_X = filedata.columns[:-1].tolist()
name_y = filedata.columns[-1]
df_in = filedata[name_X]
df_out = filedata[[name_y]]
print(f"TARGET: {name_y}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df_in_final, df_out_final = remove_outliers_iqr(df_in, df_out, rr=20)

removed_count = len(df_in) - len(df_in_final)
print(f"# of data after removing outliers: {len(df_in_final)} 개 ({removed_count} 개 제거됨)")

X = df_in_final[name_X].values
y = df_out_final[name_y].values.reshape(-1, 1)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2,
                                                  random_state=42)
print(f"Train set: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")
print(f"Validation set: {len(X_val)} ({len(X_val) / len(X) * 100:.1f}%)")
print(f"Test set: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_norm = scaler_X.fit_transform(X_train)
y_train_norm = scaler_y.fit_transform(y_train)

X_val_norm = scaler_X.transform(X_val)
X_test_norm = scaler_X.transform(X_test)

y_val_norm = scaler_y.transform(y_val)
y_test_norm = scaler_y.transform(y_test)

y = df_out_final[name_y].values.reshape(-1, 1)
num_input = X_train.shape[1]
out = sweep_multkan(
      X_train_norm, y_train_norm, X_val_norm, y_val_norm, X_test_norm, y_test_norm,
      param_grid={
          'width': [[num_input, num_input, 1]],
          'grid': [10],
          'k': [3],
          'mult_arity': [0],
          'steps': [50],
          'opt': ['LBFGS'],
          'lr': [0.1],
          'update_grid': [True],
          'lamb': [1e-3],
          'lamb_coef': [1e-1],
          'lamb_coefdiff': [1e-2],
          'lamb_entropy': [1e-1],
          'prune': [True],
          'pruning_th': [0.1, 0.2, 0.5],
          'symbolic': [True],
      },
      # seeds=[0, 1],      # run each config with multiple seeds
      seeds=[i for i in range(10)],      # run each config with multiple seeds
      use_cuda=False,     # set False to force CPU
      save_heading=save_heading,
  )
#%%
import matplotlib.pyplot as plt
df = pd.read_excel(save_heading + ".xlsx", sheet_name='results')

param_cols = [col for col in df.columns if 'param' in col and df[col].nunique() > 1]
num_params = len(param_cols)
cols = 3
rows = int(np.ceil(num_params / cols))

fig, axs = plt.subplots(rows, cols, figsize=(10, 4*rows), constrained_layout=True)
axs = np.atleast_1d(axs).flatten()

for i, param_name in enumerate(param_cols):
    ax = axs[i]
    ax.scatter(df[param_name], df['r2_val'], alpha=0.7, c='blue', edgecolors='k')
    ax.set_xlabel(param_name, fontsize=14)
    ax.set_ylabel('r2_val')
    ax.grid(True, linestyle='--', alpha=0.6)

# Hide any unused subplots if the grid is larger than the number of plots
for i in range(num_params, len(axs)):
    axs[i].axis('off')

plt.suptitle("R2 Values for Different Parameters", fontsize=14)
plt.show()

best = out['best']

#%%
res, _, _, _ = evaluate_params(
    X_train_norm, y_train_norm, X_val_norm, y_val_norm, best['params'], X_test_norm, y_test_norm, 0, scaler_y, device.type,
    save_heading=save_heading
)

print("Evaluation: ")
print(res)
