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

x1_grid = np.linspace(-np.pi, np.pi, 60)
x2_grid = np.linspace(-1, 1, 30)

x1, x2= np.meshgrid(x1_grid, x2_grid)
X = np.stack((x1.flatten(), x2.flatten()), axis=1)
# y = 10 * np.abs(x1) + 5*x2**2
y = np.sin(2*x1) + x2
eqn = "sin(2x0)+5x1"
save_heading = os.path.join(os.getcwd(), "github","workflows", "Hyein", "multkan_sweep_autosave",
                            f"fastperiodic_{eqn}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}")
y = y.flatten().reshape(-1, 1)

#%%
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)  # 0.2 × 0.8 = 0.16 (전체의 16%)

print(f"전체 데이터셋 크기: {len(X)}")
print(f"훈련셋 크기: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"검증셋 크기: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"테스트셋 크기: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# 1. MinMaxScaler 객체 생성 --- 범위를 0.1~0.9로 재설정
scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))

X_train_norm = scaler_X.fit_transform(X_train) # 훈련 데이터로 스케일러 학습 및 변환 (fit_transform)
y_train_norm = scaler_y.fit_transform(y_train) # X_train의 각 변수(컬럼)별로 최소값은 0, 최대값은 1이 되도록 변환됩니다.

X_val_norm = scaler_X.transform(X_val)
X_test_norm = scaler_X.transform(X_test)

y_val_norm = scaler_y.transform(y_val)
y_test_norm = scaler_y.transform(y_test)

# 딥러닝을 진행하기 전 모든 데이터셋을 tensor로 변환  # 원래는 numpy 배열이었음 --- 아까 scikitlearn의 train test split 이나 .fit transform 스케일러를 사용하였기에
# X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
# X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32, device=device)
# X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
# y_train_tensor = torch.tensor(y_train_norm, dtype=torch.float32, device=device)
# y_val_tensor = torch.tensor(y_val_norm, dtype=torch.float32, device=device)
# y_test_tensor = torch.tensor(y_test_norm, dtype=torch.float32, device=device)

# y = df_out_final[name_y].values.reshape(-1, 1)
out = sweep_multkan(
      X_train_norm, y_train_norm, X_val_norm, y_val_norm, X_test_norm, y_test_norm,
      param_grid={
          'width': [[X_train.shape[1], 2, 1]],
          'grid': [30],
          'k': [3],
          'mult_arity': [0],
          'steps': [50],
          'opt': ['LBFGS'],
          'lr': [1e-1, 1, 5, 10],
          'update_grid': [True, False],
          'lamb': [1e-4, 1e-3, 1e-2, 1e-1],
          'lamb_coef': [1],
          'lamb_entropy': [1],
          'prune': [True],
          'pruning_node_th': [0.01],
          'pruning_edge_th': [3e-2],
          'symbolic': [True],
          'sym_weight_simple': [0.1, 0.5], #, 0.9, 0.99],
          'sym_a_range': [(-20, 20)],
      },
      seeds=[0, 17, 42],      # run each config with multiple seeds
      n_jobs=1,          # number of parallel worker processes
      use_cuda=False,     # set False to force CPU
      save_heading=save_heading,
  )
best = out['best']
print('Best configuration:')
print(json.dumps(best, indent=2))

res, model, _, _ = evaluate_params(
    X_train, y_train, X_val, y_val, best['params'], X_test, y_test, 0, scaler_y, device.type,
    save_heading=save_heading
)
torch.save(model.state_dict(), f"{save_heading}_model.pt")
