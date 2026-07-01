#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from kan.custom_processing import remove_outliers_iqr
import torch
import os
import pandas as pd

data_name = "CrossedBarrel"

root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein')
filepath = os.path.join(root_dir, "data", f"{data_name}.csv")
savepath = os.path.join(root_dir, "nn_models")

filedata = pd.read_csv(filepath)
name_X = filedata.columns[:-1].tolist()
name_y = filedata.columns[-1]
df_in = filedata[name_X]
df_out = filedata[[name_y]]
print(f"TARGET: {name_y}")

df_in_final, df_out_final = remove_outliers_iqr(df_in, df_out)

removed_count = len(df_in) - len(df_in_final)
print(f"# of data after removing outliers: {len(df_in_final)} 개 ({removed_count} 개 제거됨)")

X = df_in_final[name_X].values
y = df_out_final[name_y].values.reshape(-1, 1)

X_temp_denorm, X_test_denorm, y_temp_denorm, y_test_denorm = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_denorm, X_val_denorm, y_train_denorm, y_val_denorm = train_test_split(X_temp_denorm, y_temp_denorm, test_size=0.2,
                                                  random_state=42)
print(f"Train set: {len(X_train_denorm)} ({len(X_train_denorm) / len(X) * 100:.1f}%)")
print(f"Validation set: {len(X_val_denorm)} ({len(X_val_denorm) / len(X) * 100:.1f}%)")
print(f"Test set: {len(X_test_denorm)} ({len(X_test_denorm) / len(X) * 100:.1f}%)")

scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))
X_train_norm = scaler_X.fit_transform(X_train_denorm)
y_train_norm = scaler_y.fit_transform(y_train_denorm)
X_val_norm = scaler_X.transform(X_val_denorm)
X_test_norm = scaler_X.transform(X_test_denorm)
y_val_norm = scaler_y.transform(y_val_denorm)
y_test_norm = scaler_y.transform(y_test_denorm)

#%
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
import numpy as np

# ==========================================
# 1. 탐색할 파라미터 범위 설정 (Dictionary)
# ==========================================
param_distributions = {
    # 은닉층 구조: (노드수, 노드수) -> 2층 구조 위주로 테스트
    'hidden_layer_sizes': [
        (64, 64),          # 기본 2층
        (128, 64),         # 앞이 더 넓은 2층
        (100, 100),        # 넓은 2층
        (64, 32, 16),      # (참고용) 3층 구조도 슬쩍 넣어봄
        (128,)             # (참고용) 아주 넓은 1층
    ],
    # 활성화 함수: tanh는 부드러운 곡선(공학 데이터)에 유리할 수 있음
    'activation': ['relu', 'tanh'],
    # Optimizer: lbfgs는 데이터가 수천 개 이내일 때 수렴 속도와 정확도가 매우 좋음
    'solver': ['adam', 'lbfgs'],
    # 규제(L2) 강도: 높을수록 식을 단순하게 만듦
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    # 학습률 (Adam일 때만 적용됨)
    'learning_rate_init': [0.001, 0.01, 0.0005]
}

# ==========================================
# 2. 기본 모델 및 튜닝 객체 설정
# ==========================================
# max_iter를 넉넉하게 주어 수렴 경고(ConvergenceWarning) 방지
mlp = MLPRegressor(max_iter=100000, random_state=42)

# RandomizedSearchCV 설정
# n_iter=20: 총 20번의 조합을 랜덤으로 뽑아서 테스트 (시간 조절 가능)
# cv=3: 3-Fold 교차 검증 (데이터를 3개로 쪼개서 검증)
search = RandomizedSearchCV(
    estimator=mlp,
    param_distributions=param_distributions,
    n_iter=200,     # 시도할 조합의 개수 (많을수록 좋지만 느려짐)
    cv=3,          # 교차 검증 횟수
    scoring='r2',  # 평가지표: R2 Score
    n_jobs=-1,     # CPU 병렬 처리 (속도 향상)
    verbose=1,     # 진행 상황 출력
    random_state=42
)

# ==========================================
# 3. 튜닝 실행 (Fitting)
# ==========================================
print("🚀 하이퍼파라미터 튜닝 시작... (잠시만 기다려주세요)")
search.fit(X_train_norm, y_train_norm.ravel())

# ==========================================
# 4. 결과 확인
# ==========================================
print("\n" + "="*40)
print(f"🏆 최적 파라미터: {search.best_params_}")
print(f"⭐️ 최고 교차검증 점수 (R2): {search.best_score_:.4f}")
print("="*40)

# 최적 모델로 테스트 데이터 최종 평가
best_model = search.best_estimator_
y_pred = best_model

#%
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV

# ... (앞부분의 데이터 로드, 스케일링, 튜닝 코드는 동일) ...

# 1. RandomizedSearchCV 실행 (가정)
# search.fit(X_train_scaled, y_train)

# 2. 최적 모델 추출
best_model = search.best_estimator_

# 3. 모델 및 스케일러 저장
# 모델 저장 (.pkl 파일)
joblib.dump(best_model, os.path.join(savepath, f'{data_name}_best_mlp_model.pkl'))

# 스케일러 저장 (매우 중요! 나중에 새 데이터도 이걸로 변환해야 함)
# (코드 앞부분에서 정의한 scaler 변수를 저장합니다)
joblib.dump(scaler_y, os.path.join(savepath, f'{data_name}_mlp_scaler_y.pkl'))
joblib.dump(scaler_X, os.path.join(savepath, f'{data_name}_mlp_scaler_X.pkl'))

print("💾 모델과 스케일러가 성공적으로 저장되었습니다!")
print(" - 모델 파일: best_mlp_model.pkl")
print(" - 스케일러 파일: mlp_scaler.pkl")

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol

# ==========================================
# 1. SALib 문제 정의 (Problem Definition)
# ==========================================
# 모델이 학습된 입력 변수의 개수와 범위를 정의합니다.
# 주의: 스케일링된 데이터로 학습했다면, 범위도 그에 맞춰야 합니다.
# 예: MinMaxScaler(-1, 1)을 썼다면 bounds는 [-1, 1] 이어야 합니다.

# 변수 개수 (X_train의 컬럼 수)
n_features = X_train_norm.shape[1]

# 변수 이름 (없으면 그냥 x0, x1... 으로 생성)
feature_names = [f"Feature {i}" for i in range(n_features)]
# 만약 pandas 컬럼 이름이 있다면: feature_names = list(X.columns)

problem = {
    'num_vars': n_features,
    'names': feature_names,
    'bounds': [[-1, 1]] * n_features  # 모든 변수의 범위가 -1 ~ 1 이라고 가정 (스케일링 맞춤)
}

# ==========================================
# 2. 샘플 데이터 생성 (Sample Generation)
# ==========================================
# N은 샘플링 개수입니다. 클수록 정확하지만 계산 시간이 늘어납니다. (보통 1024 이상 권장)
# 총 실행 횟수 = N * (2 * D + 2)  (D는 변수 개수)
N = 1024
X_sobol = saltelli.sample(problem, N, calc_second_order=True)

print(f"생성된 샘플 개수: {X_sobol.shape[0]}개")

# ==========================================
# 3. 모델 예측 실행 (Run Model)
# ==========================================
# 생성된 샘플(X_sobol)을 XGBoost 모델에 넣고 예측값(Y)을 구합니다.
# XGBoost predict는 numpy array를 잘 받으므로 바로 넣으면 됩니다.

Y_sobol = best_model.predict(X_sobol)

# ==========================================
# 4. Sobol 분석 수행 (Analyze)
# ==========================================
# calc_second_order=True면 변수 간의 상호작용(Interaction)까지 분석합니다.
Si = sobol.analyze(problem, Y_sobol, calc_second_order=True)

# ==========================================
# 5. 결과 확인 및 시각화
# ==========================================

# 텍스트로 출력
print("\n[Sobol Analysis Result]")
total_si = Si['ST'] # Total Effect Index (총 영향력)
first_si = Si['S1'] # First Order Index (단독 영향력)

results_df = pd.DataFrame({
    'Feature': feature_names,
    'Total_Effect (ST)': total_si,
    'First_Order (S1)': first_si
}).sort_values(by='Total_Effect (ST)', ascending=False)

print(results_df)

# 막대 그래프 그리기 (상위 10개만)
plt.figure(figsize=(10, 6))
plt.title("Feature Sensitivity (Total Effect Index)")
plt.barh(results_df['Feature'][:10][::-1], results_df['Total_Effect (ST)'][:10][::-1])
plt.xlabel("Total Effect Index (ST)")
plt.savefig(os.path.join(savepath, f"{data_name}_sobol_analysis.png"))
plt.show()
