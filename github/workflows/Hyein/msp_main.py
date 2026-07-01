"""
MSP main
: the results are slightly twisted compared to the previous version plots.
"""
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from kan.custom_processing import remove_outliers_iqr
import pandas as pd

from kan.experiments.multkan_hparam_sweep_materials import evaluate_params
from kan.custom_processing import plot_activation_and_spline_coefficients
import matplotlib.pyplot as plt

file_name = "20251014_150723_auto_MSP"
root_dir = os.path.join(os.getcwd(), 'github\workflows\Hyein')
df = pd.read_excel(os.path.join(root_dir, "multkan_sweep_autosave", file_name + ".xlsx"), sheet_name='best_avg_by_params')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"This script is running on {device}.")
save_tag = "CO2RR_MSP_" + file_name
save_dir = os.path.join(os.getcwd(), 'github\workflows\Hyein\custom_figures')
save_heading = os.path.join(save_dir, save_tag)

d_opt = df.iloc[0]
d_opt = d_opt.to_dict()
params = {k: v for k, v in d_opt.items() if "param_" in k}
params = {key.replace('param_', ''): value for key, value in params.items()}

dir_current = os.getcwd()
filepath = os.path.join(dir_current, "github\workflows\TaeWoong", "25.01.14_CO2RR_GSA.xlsx")

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

res, model, fit_kwargs, dataset, _ = evaluate_params(
    X_train_norm, y_train_norm, X_val_norm, y_val_norm, params, X_test_norm, y_test_norm,
    0, scaler_y, device.type,
    special_tag=save_tag,
    special_dir=save_dir
)
model.plot()
plt.show()
#%%
from kan.utils import ex_round
sym_fun = ex_round(model.symbolic_formula()[0][0], 4)
with open(os.path.join(save_dir, f"{save_tag}_sym_res.txt"), "w") as f:
    f.write(f"{sym_fun}\n")
#%%
params['grid'] = 30
params['sym_weight_simple'] = 0.8
res_refine, model_refine, fit_kwargs_refine, dataset_refine, _ = evaluate_params(
    X_train_norm, y_train_norm, X_val_norm, y_val_norm, params, X_test_norm, y_test_norm,
    0, scaler_y, device.type,
    special_tag=save_tag + "_refine",
    special_dir=save_dir
)
model_refine.plot()
plt.show()
#%%
X_norm = scaler_X.transform(X)
y_norm = scaler_y.transform(y)

plot_activation_and_spline_coefficients(model, save_heading=save_heading, x=dataset, layers=None)

scores_tot = model.node_scores[0].detach().cpu().numpy()
fig, ax = plt.subplots()
ax.bar(list(range(scores_tot.shape[0])), scores_tot.tolist())
plt.savefig(os.path.join(save_dir, f"{save_tag}_scores_L0.png"))
plt.show()

#%% Draw attribution scores across multi-layer
