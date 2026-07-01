import pandas as pd
import numpy as np
import torch
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from kan.custom import MultKAN
from sklearn.metrics import mean_squared_error, r2_score
from kan.custom_processing import remove_outliers_iqr, evaluate_model_performance, plot_activation_functions
import datetime
from kan.experiments.multkan_hparam_sweep_materials import _seed_everything
seed = 0
_seed_everything(seed=seed)

save_tag = 'toy' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = "D:\pykan\github\workflows\Hyein\custom_figures"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"This script is running on {device}.")

x1_grid = np.linspace(-np.pi, np.pi, 30)
x2_grid = np.linspace(-1, 1, 30)
x1, x2= np.meshgrid(x1_grid, x2_grid)
X = np.stack((x1.flatten(), x2.flatten()), axis=1)
y = 5 * np.sin(2*x1) + x2

y = y.flatten().reshape(-1, 1)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
_seed_everything(seed=seed)

scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))
X_train_norm = scaler_X.fit_transform(X_train) # 훈련 데이터로 스케일러 학습 및 변환 (fit_transform)
y_train_norm = scaler_y.fit_transform(y_train)

X_val_norm = scaler_X.transform(X_val)
y_val_norm = scaler_y.transform(y_val)
X_test_norm = scaler_X.transform(X_test)
y_test_norm = scaler_y.transform(y_test)

X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32, device=device)
X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train_norm, dtype=torch.float32, device=device)
y_val_tensor = torch.tensor(y_val_norm, dtype=torch.float32, device=device)
y_test_tensor = torch.tensor(y_test_norm, dtype=torch.float32, device=device)

dataset = {'train_input': X_train_tensor,'train_label': y_train_tensor,
            'val_input': X_val_tensor, 'val_label': y_val_tensor,
            'test_input': X_test_tensor,'test_label': y_test_tensor }

for key, value in dataset.items():
    print(f"{key}: {value.shape}")

params = {
    'width': [X_train.shape[1], 6, 1],
    'grid_range': [0.1, 0.9],
    'grid': 10,
    'k': 3,
    'mult_arity': 0,
    'steps': 50,  # 200
    'opt': 'LBFGS',
    'lr': 1.0,
    'update_grid': True,
    'lamb': 0.001,
    'lamb_l1': 1.0,
    'lamb_coef': 5,
    'lamb_coefdiff': 0.0,
    'lamb_entropy': 5.,
    'batch': -1,
    'log': 1,
    'prune': True,
    'pruning_node_th': 0.01,
    'pruning_edge_th': 3e-2,
    'symbolic': True,
    'sym_weight_simple': 0.5,
    'sym_r2_threshold': 0.,
}

#
import matplotlib.pyplot as plt

X_norm = scaler_X.transform(X)
y_norm = scaler_y.transform(y)

nx = X_norm.shape[1]
fig, axs = plt.subplots(nrows=1, ncols=nx, figsize=(15, 3))
for idx_x in range(nx):
    ax = axs[idx_x]
    ax.scatter(X_norm[:, idx_x], y_norm, color='black')
# plt.savefig(os.path.join(save_dir, f"{save_tag}_data.png"))
plt.show()
#%%
model_kwargs = {k: params[k] for k in ['width', 'grid', 'grid_range', 'k', 'mult_arity', 'seed', 'device'] if k in params}

model = MultKAN(**model_kwargs)

fit_kyewords = [
    'opt', 'steps', 'lamb', 'lamb_l1', 'lamb_entropy', 'lamb_coef', 'lamb_coefdiff', 'update_grid', 'lr', 'batch', 'log',
]
fit_kwargs = {key: params[key] for key in fit_kyewords}

# KAN 학습
model.fit(dataset, **fit_kwargs)
model.plot()
plt.show()
# val_pred, val_actual, val_metrics = evaluate_model_performance(model, dataset, scaler_y, display=True)

if params['prune']:
    # Unified pruning threshold handling: if 'pruning_th' is provided, use it for both node_th and edge_th
    node_th = params['pruning_node_th']
    edge_th = params['pruning_edge_th']
    model = model.prune(node_th=node_th, edge_th=edge_th)
    model.plot()
    plt.show()

# if params['symbolic']:
#     lib = ['sin', 'cos', 'x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', '1/x', '1/x^2']
#     sym_weight_simple = params['sym_weight_simple']
#     sym_r2_threshold = params['sym_r2_threshold']
#     model.auto_symbolic(lib=lib, weight_simple=sym_weight_simple, r2_threshold=sym_r2_threshold)
#     model.fit(dataset, **fit_kwargs)
#     model.plot()
#     plt.show()


#%% Test if calling forward function varies the node scores: True!
# it doesn't work for symbolified functions because it generates linear function nodes
test1 = torch.tensor([[0.9, 0.9], [0.8, 0.8], [0.8, 0.9]], dtype=torch.float32, device=device)
out_test1 = model.forward(test1)
score_test1 = model.node_scores[0].detach().cpu().numpy()
print("score_test1:========================")
print(score_test1)

test2 = torch.tensor([[0.1, 0.1], [0.1, 0.2], [0.2, 0.2]], dtype=torch.float32, device=device)
out_test2 = model.forward(test2)
score_test2 = model.node_scores[0].detach().cpu().numpy()
print("score_test2:========================")
print(score_test2)
#%%
val_pred, val_actual, val_metrics = evaluate_model_performance(model, dataset, scaler_y, display=True)
# test_pred, test_actual, test_metrics = evaluate_model_performance(model, dataset, scaler_y, "test")

plt.figure(figsize=(4, 4))  # 도화지 그리기~

plt.scatter(val_actual, val_pred, facecolors='none', edgecolors='r', s=15, label='Model Predictions')

# 제일 작은 값, 제일 큰 값 설정
min_val = min(val_actual.min(), val_pred.min())
max_val = max(val_actual.max(), val_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Fit (y=x)')  # y = x 선긋기

# 그래프 제목과 축 레이블 설정
plt.xlabel("Actual" , fontsize=9)   # Actual 다음에 우리가 보고자 하는 output predicting 변수가 뜸
plt.ylabel("Predicted", fontsize=9) #
plt.title(f'Test Set: Actual vs. Predicted (R² = {val_metrics["r2"]:.4f})', fontsize=11)
plt.legend()
plt.grid(True)  # 격자 on
plt.axis('equal') # x, y축 스케일을 동일하게 설정
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"{save_tag}_validation.png"))
plt.show()
#%%
from kan.custom_processing import plot_data_per_interval, plot_spline_coefficients
save_heading = os.path.join(save_dir, save_tag)
X_norm = scaler_X.transform(X)
y_norm = scaler_y.transform(y)
name_X = [f'x{idx}' for idx in range(X_norm.shape[1])]
name_y = ['y']

fig_x1, ax_x1 = plot_data_per_interval(X_norm, y_norm, name_X, name_y, 1, [0, 0.4])
plt.savefig(f"{save_heading}_data_colored.png")
plt.show()

# Plot learned activation functions (splines) per edge after training/pruning
plot_activation_functions(model, save_heading=save_heading, x=dataset, layers=None)
plot_spline_coefficients(model, save_heading=save_heading)

model.forward(torch.tensor(X_norm, dtype=torch.float32, device=device))
scores = model.node_scores[0].detach().cpu().numpy()
print(scores)
fig, ax = plt.subplots()
ax.bar(list(range(scores.shape[0])), scores.tolist())
plt.savefig(os.path.join(save_dir, f"{save_tag}_scores_L0.png"))
plt.show()