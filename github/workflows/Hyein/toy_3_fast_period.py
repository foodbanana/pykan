#%%
import pandas as pd
from kan.custom_processing import (plot_data_per_interval,
                                   plot_activation_and_spline_coefficients, get_masks)
import matplotlib.pyplot as plt
import os
import datetime
import json
import numpy as np

root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein')
save_dir = os.path.join(root_dir, "custom_figures")
time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')

fn = "20251020_173312_auto_sin(2x0)+5x1"
ground_truth = lambda x0, x1: np.sin(2*x0) + 5*x1
save_tag = 'fastperiodic_' + fn
save_heading = os.path.join(save_dir, save_tag)

df = pd.read_excel(os.path.join(root_dir, 'multkan_sweep_autosave', fn + ".xlsx"), sheet_name='best_avg_by_params')
d_opt = df

#%%
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from kan.experiments.multkan_hparam_sweep_materials import evaluate_params, _to_tensor, _build_dataset
import numpy as np
from kan.utils import ex_round

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"This script is running on {device}.")

x1_grid = np.linspace(-np.pi, np.pi, 30)
x2_grid = np.linspace(-1, 1, 30)

x1, x2= np.meshgrid(x1_grid, x2_grid)
X = np.stack((x1.flatten(), x2.flatten()), axis=1)

d_opt_flat = d_opt.iloc[0]
d_opt_flat = d_opt_flat.to_dict()
params = {k: v for k, v in d_opt_flat.items() if "param_" in k}
params = {key.replace('param_', ''): value for key, value in params.items()}

y_mesh = ground_truth(x1, x2)
y = y_mesh.flatten().reshape(-1, 1)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)  # 0.2 × 0.8 = 0.16 (전체의 16%)

scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))
X_train_norm = scaler_X.fit_transform(X_train)
y_train_norm = scaler_y.fit_transform(y_train)
X_val_norm = scaler_X.transform(X_val)
X_test_norm = scaler_X.transform(X_test)
y_val_norm = scaler_y.transform(y_val)
y_test_norm = scaler_y.transform(y_test)

params['symbolic'] = False

res, model, fit_kwargs, dataset = evaluate_params(
    X_train_norm, y_train_norm, X_val_norm, y_val_norm, params, X_test_norm, y_test_norm,
    0, scaler_y, device.type,
    save_heading=save_heading
)
model.plot()
plt.show()
print(res)
with open(os.path.join(save_dir, f"{save_tag}_result.json"), 'w') as f:
    json.dump(vars(res), f, indent=4)

model.suggest_symbolic(0,0,0, weight_simple=0.5, a_range=(-50, 50))
#%%
dataset_sym = _build_dataset(X_train_norm, y_train_norm, X_val_norm, y_val_norm, X_test_norm, y_test_norm, device)
# dataset_sym2 = {k: v.detach().requires_grad_(True) for k, v in dataset.items()}

lib = ['sin', 'cos', 'x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', '1/x', '1/x^2']
# sym_weight_simple = 0.5
sym_weight_simple = params.get('sym_weight_simple', 0.8)
sym_r2_threshold = params.get('sym_r2_threshold', 0.)

model.auto_symbolic(lib=lib,
                    weight_simple=sym_weight_simple, r2_threshold=sym_r2_threshold,
                    a_range=(-20, 20))
model.fit(dataset_sym, **fit_kwargs)
model.plot()
plt.show()

sym_fun = ex_round(model.symbolic_formula()[0][0], 4)
#%%
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(x1, x2, y_mesh, cmap='viridis', edgecolor='none')
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('y')
fig.colorbar(surface, shrink=0.5, aspect=5)
plt.savefig(os.path.join(save_dir, f"{save_tag}_ground_truth.png"))
plt.show()

plot_activation_and_spline_coefficients(model, save_heading=save_heading, x=dataset, layers=None)

# Compute attribution score
scores_tot = model.node_scores[0].detach().cpu().numpy()
fig, ax = plt.subplots()
ax.bar(list(range(scores_tot.shape[0])), scores_tot.tolist())
plt.savefig(os.path.join(save_dir, f"{save_tag}_scores_L0.png"))
plt.show()

#%% Raw data analysis
X_norm = scaler_X.transform(X)
y_norm = scaler_y.transform(y)
name_X = [f'x{idx}' for idx in range(X_norm.shape[1])]
name_y = ['y']
y_pred_norm = model(_to_tensor(X_norm, device)).detach().cpu().numpy()
y_pred = scaler_y.inverse_transform(y_pred_norm)

# mask_idx = 1
# mask_interval = [-1 + 0.5 * i for i in range(5)]
mask_idx = 1
# mask_interval = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
mask_scaled_interval = [0, 0.4, 0.6, 1]
mask_interval = [scaler_X.inverse_transform(np.array([[x0, x0]]))[0,mask_idx] for x0 in mask_scaled_interval]

fig_x1, ax_x1 = plot_data_per_interval(X, y, name_X, name_y, mask_idx, mask_interval)
plt.savefig(os.path.join(save_dir, f"{save_tag}_data_colored.png"))
for idx_x in range(X.shape[1]):
    ax_x1[idx_x].scatter(X[:, idx_x], y_pred, color='k', s=.9, label='Prediction')
plt.savefig(os.path.join(save_dir, f"{save_tag}_data_and_prediction.png"))
plt.show()

# Validation data analysis
# y_pred_norm_val = model(_to_tensor(X_val_norm, device))
# y_pred_val = scaler_y.inverse_transform(y_pred_norm_val.detach().cpu().numpy())
# fig_val, ax_val = plt.subplots(nrows=1, ncols=X.shape[1], figsize=(15, 3), constrained_layout=True, sharey=True)
# for idx_x in range(X.shape[1]):
#     ax = ax_val[idx_x]
#     ax.scatter(X_val[:, idx_x], y_val, color='tab:gray')
#     ax.scatter(X_val[:, idx_x], y_pred_val, color='k', s=.9, label='Prediction')
#     ax.set_title(name_X[idx_x], fontsize=8)
# plt.show()
#%%
torch.save(model.state_dict(), f"{save_heading}_model.pt")

#%% Plot attribution scores for each interval
masks = get_masks(X, mask_idx, mask_interval)
scores_interval = []
acts_interval = []
for mask in masks:
    if np.any(mask):
        x_masked = X[mask, :]   # 이게 아니라 fabricated, 임의의 input을 주게 되면 어떨까?
        x_norm_masked = scaler_X.transform(x_masked)
        x_tensor_masked = torch.tensor(x_norm_masked, dtype=torch.float32, device=device)
        model.forward(x_tensor_masked)
        acts_interval.append(model.acts)
        scores_interval.append(model.feature_score.detach().cpu().numpy().copy())
    else:
        scores_interval.append(np.zeros(scores_tot.shape))

width = 0.25
fig, ax = plt.subplots()
xticks = np.arange(len(masks))
xticklabels = [f'{lb:.2f} < x{mask_idx} <= {ub:.2f}' for lb, ub in zip(mask_interval[:-1], mask_interval[1:])]
max_score = max([max(s) for s in scores_interval])
for idx in range(scores_tot.shape[0]):
    bars = ax.bar(xticks + idx * width, [s[idx] for s in scores_interval], width, label=f"x{idx}")
    ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=3)
ax.margins(x=0.1)
ax.set_ylim(0, max_score * 1.1)

ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, rotation=10, ha='center', fontsize=8)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{save_tag}_scores_L0_interval.png"))
plt.show()
print(scores_interval)

with open(os.path.join(save_dir, f"{save_tag}_sym_res.txt"), 'w') as f:
    f.write(f"{sym_fun}\n")
