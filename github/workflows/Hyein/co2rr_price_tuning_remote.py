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

dir_current = os.getcwd()
save_heading = os.path.join(dir_current, "github", "workflows", "Hyein", "multkan_sweep_autosave",
                            "CO2RR_MSP_" + datetime.datetime.now().strftime('%Y%m%d_%H%M'))
filepath = os.path.join(dir_current, "github", "workflows", "TaeWoong", "25.01.14_CO2RR_GSA.xlsx")

xls = pd.ExcelFile(filepath)
df_in  = pd.read_excel(xls, sheet_name='Input')
df_out = pd.read_excel(xls, sheet_name='Output')

df_in_final, df_out_final = remove_outliers_iqr(df_in, df_out)

removed_count = len(df_in) - len(df_in_final)  # 몇 개 지웠는지 세기
print(f"이상치 제거 후 데이터 수: {len(df_in_final)} 개 ({removed_count} 개 제거됨)")
print("--- 이상치 제거 완료 ---\n")

name_X = [
    "Current density (mA/cm2)",
    "Faradaic efficiency (%)",
    "CO coversion",
    "Voltage (V)",
    "Electricity cost ($/kWh)",
    "Membrain cost ($/m2)",
    "Catpure energy (GJ/ton)",
    "Crossover rate"
]
name_y = "MSP ($/kgCO)" # Required energy_total (MJ/kgCO) # MSP ($/kgCO)
X = df_in_final[name_X].values
y = df_out_final[name_y].values.reshape(-1, 1)

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

y = df_out_final[name_y].values.reshape(-1, 1)
out = sweep_multkan(
      X_train_norm, y_train_norm, X_val_norm, y_val_norm, X_test_norm, y_test_norm,
      param_grid={
          'width': [[X_train.shape[1], X_train.shape[1], 1]],
          'lr': [0.1],
          # 'lr': [0.01, 0.1],
          'update_grid': [True],
          'lamb': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
          'lamb_coef': [0.1],
          'lamb_coefdiff': [0.1],
          'lamb_entropy': [0.1],
          'prune': [True],
          'pruning_th': [0.05],
          # 'symbolic': [True],
          # 'sym_weight_simple': [0, 0.5, 0.9],
      },
      seeds=[60+i for i in range(40)],      # run each config with multiple seeds
      n_jobs=1,          # number of parallel worker processes
      use_cuda=False,     # set False to force CPU,
      scaler_y=scaler_y,
      save_heading=save_heading,
  )

best = out['best']
print('Best configuration:')
print(json.dumps(best, indent=2))
#%%
res, model, _, _ = evaluate_params(
    X_train_norm, y_train_norm, X_val_norm, y_val_norm, best['params'], X_test_norm, y_test_norm, 0, scaler_y, device.type,
    save_heading=save_heading
)
torch.save(model.state_dict(), f"{save_heading}_model.pt")
