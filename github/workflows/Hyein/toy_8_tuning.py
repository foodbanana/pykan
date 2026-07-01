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

x1_grid = np.linspace(-1, 1, 40)
x2_grid = np.linspace(-1, 1, 40)

x1, x2= np.meshgrid(x1_grid, x2_grid)
X = np.stack((x1.flatten(), x2.flatten()), axis=1)
y = x1**2 / (x2 + 1.02) / 10 + np.exp(2 * x2)
save_heading = os.path.join(os.getcwd(), "github", "workflows", "Hyein", "multkan_sweep_autosave",
                            f"toy_8_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}")
y = y.flatten().reshape(-1, 1)

feature_range = (0.1, 0.9)
#%%
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)  # 0.2 × 0.8 = 0.16 (전체의 16%)

print(f"전체 데이터셋 크기: {len(X)}")
print(f"훈련셋 크기: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"검증셋 크기: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"테스트셋 크기: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

scaler_X = MinMaxScaler(feature_range=feature_range)
scaler_y = MinMaxScaler(feature_range=feature_range)

X_train_norm = scaler_X.fit_transform(X_train)
y_train_norm = scaler_y.fit_transform(y_train)

X_val_norm = scaler_X.transform(X_val)
X_test_norm = scaler_X.transform(X_test)

y_val_norm = scaler_y.transform(y_val)
y_test_norm = scaler_y.transform(y_test)
num_input = X_train.shape[1]

# First: Learning rate and lambda and steps

out = sweep_multkan(
      X_train_norm, y_train_norm, X_val_norm, y_val_norm, X_test_norm, y_test_norm,
      param_grid={
          'width': [
              # [num_input, 1],
              [num_input, num_input, 1],
              # [num_input, num_input, num_input, 1],
          ],
          'grid_range': [feature_range],
          'lr': [0.1],
          'lamb': [0.01],    # 0.01 (ddp)
          'stop_grid_update_step': [20],
          'lamb_entropy': [0.001, 0.01, 0.1, 1],
          'lamb_coef': [0.001, 0.01, 0.1, 1],
          'lamb_coefdiff': [0.005, 0.05, 0.5, 1],
          'prune': [True],
          'pruning_th': [0.05],
          'symbolic': [True],
          'sym_weight_simple': [0.],
          'sym_a_range': [(-50, 50)],
          'sym_b_range': [(-50, 50)],
      },
      seeds=[i for i in range(10)],      # run each config with multiple seeds
      n_jobs=1,          # number of parallel worker processes
      use_cuda=False,     # set False to force CPU
      save_heading=save_heading,
  )
# Second: Lambdas
# out = sweep_multkan(
#       X_train_norm, y_train_norm, X_val_norm, y_val_norm, X_test_norm, y_test_norm,
#       param_grid={
#           'width': [
#               [num_input, num_input, 1],
#           ],
#           'lr': [0.01],
#           'lamb': [0.001],
#           'stop_grid_update_step': [20],
#           'lamb_entropy': [0.001, 0.01, 0.1, 1],
#           'lamb_coef': [0.1], #[0.001, 0.01, 0.1, 1],
#           'lamb_coefdiff': [0.5], #[0.005, 0.05, 0.5, 1],
#           'prune': [True],
#           'pruning_th': [0.01, 0.02, 0.05, 0.1],
#           'symbolic': [True],
#       },
#       seeds=[i for i in range(5)],      # run each config with multiple seeds
#       n_jobs=1,          # number of parallel worker processes
#       use_cuda=False,     # set False to force CPU
#       save_heading=save_heading,
#   )

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
plt.savefig(f"{save_heading}_r2_values.png")
plt.show()

best = out['best']

res, model, _, _ = evaluate_params(
    X_train_norm, y_train_norm, X_val_norm, y_val_norm, best['params'], X_test_norm, y_test_norm, 0, scaler_y, device.type,
    save_heading=save_heading
)
model.saveckpt(path=f"{save_heading}_model")

print("Evaluation: ")
print(res)