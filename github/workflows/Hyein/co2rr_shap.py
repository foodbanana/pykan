import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import json
from kan.custom_processing import remove_outliers_iqr
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from scikeras.wrappers import KerasRegressor  # scikeras 권장
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping
import shap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dir_current = os.getcwd()
dir_parent = os.path.dirname(dir_current)
filepath = os.path.join(dir_parent, "TaeWoong", "25.01.14_CO2RR_GSA.xlsx")

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
    "CO conversion",
    "Voltage (V)",
    "Electricity cost ($/kWh)",
    "Membrain cost ($/m2)",
    "Catpure energy (GJ/ton)",
    "Crossover rate"
]
name_y = "Required energy_total (MJ/kgCO)" # Required energy_total (MJ/kgCO) # MSP ($/kgCO)
X = df_in_final[name_X].values
y = df_out_final[name_y].values.reshape(-1, 1)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)  # 0.2 × 0.8 = 0.16 (전체의 16%)

scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))

X_train_norm = scaler_X.fit_transform(X_train) # 훈련 데이터로 스케일러 학습 및 변환 (fit_transform)
y_train_norm = scaler_y.fit_transform(y_train) # X_train의 각 변수(컬럼)별로 최소값은 0, 최대값은 1이 되도록 변환됩니다.

X_val_norm = scaler_X.transform(X_val)
X_test_norm = scaler_X.transform(X_test)

y_val_norm = scaler_y.transform(y_val)
y_test_norm = scaler_y.transform(y_test)

#%%

def create_ann_model(units1=32, units2=16, activation='relu'):
    model = Sequential([
        Input(shape=(8,)),  # 입력 특성 8개
        Dense(units1, activation=activation),  # hidden layer (units1)개
        Dense(units2, activation=activation),  # 입력 특성 8개 (units2)개
        Dense(1, activation='linear')
    ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',  # loss function = MSE -- 이것에 따라 학습의 방향이 결정됨
        metrics=['mean_absolute_error']  # 학습에 영향은 주지 않지만 그냥 평가지표 MAE
    )
    return model


regressor = KerasRegressor(
    # Keras로 만든 딥러닝 모델을 Scikit-learn의 회귀(Regressor) 모델처럼 보이게 포장 --- scikit-learn의 최적의 units 수 등을 찾는 gridsearchCV 같은 기능을 쓰기 가능
    model=create_ann_model,
    verbose=0  # verbose = 0 --- 모델 훈련과정 출력 X // 1 -- 출력  //  2 -- epoch 끝날때마다 출력
)

grid = GridSearchCV(
    estimator=KerasRegressor(model=create_ann_model, verbose=0),
    # Keras로 만든 딥러닝 모델을 Scikit-learn의 회귀(Regressor) 모델처럼 보이게 포장 --- scikit-learn의 최적의 units 수 등을 찾는 gridsearchCV 같은 기능을 쓰기 가능
    # verbose = 0 --- 모델 훈련과정 출력 X // 1 -- 출력  //  2 -- epoch 끝날때마다 출력
    param_grid={
        'model__units1': [16, 32],  # hidden layer 1층 뉴런수
        'model__units2': [8, 16],  # hidden layer 2층 뉴런수
        'model__activation': ['relu', 'tanh'],  # 활성화함수
        'batch_size': [32],  # 배치 사이즈
        'epochs': [100],  # 학습 epoch 수  --- 총 학습 수 = 2*2*2 = 8 이다.
    },
    cv=3,  # cross validation 3번 (3 fold validation)  --- 총 학습 수 = 8*3 = 24
    scoring='neg_mean_squared_error',  # -MSE가 높을수록 좋다고 설정
    n_jobs=-1  # 사용 가능한 CPU 코어 전부 사용하라는 의미
)

grid_result = grid.fit(X_train_norm,
                       y_train_norm)  # GridSearchCV.fit() 실행하기 -- 여러 모델 구조 비교 --- 여러 정보가 grid_result에 할당이 됨

# 6. 결과 출력
print("최적 하이퍼파라미터:", grid_result.best_params_)
print("최적 평균 검증 MSE:", -grid_result.best_score_)  # 아까 score = -MSE로 정의함

# 7. 최적 파라미터 추출 / best_params에 저장되어 있는 최적 정보들 활용
optimal_units1 = grid_result.best_params_['model__units1']  # best_params[키] = 밸류값 출력
optimal_units2 = grid_result.best_params_['model__units2']
optimal_activation = grid_result.best_params_['model__activation']
optimal_batch_size = grid_result.best_params_['batch_size']

# GridSearchCV.fit() 을 실행했을 때 과정
cv_results = pd.DataFrame(grid_result.cv_results_)  # 결과표를 pandas dataframe으로 변환 --- 그러면 보기 편함

# 결과 저장 경로 설정 (동일 폴더 내 figures 하위 폴더)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ 이 없는 환경(예: 인터랙티브) 대비
    script_dir = os.getcwd()
output_dir = os.path.join(script_dir, 'mlp_results')
os.makedirs(output_dir, exist_ok=True)

# 상세 결과 저장 (원본 및 정렬본)
cv_results_path = os.path.join(output_dir, 'gridsearch_cv_results.csv')
cv_results_sorted = cv_results[['params', 'mean_test_score', 'std_test_score']].sort_values('mean_test_score', ascending=False)
cv_results_sorted_path = os.path.join(output_dir, 'gridsearch_cv_results_sorted.csv')
cv_results.to_csv(cv_results_path, index=False)
cv_results_sorted.to_csv(cv_results_sorted_path, index=False)

# 최적 하이퍼파라미터 및 점수 저장
best_summary_path = os.path.join(output_dir, 'gridsearch_best_summary.txt')
with open(best_summary_path, 'w', encoding='utf-8') as f:
    f.write('최적 하이퍼파라미터\n')
    f.write(str(grid_result.best_params_) + '\n')
    f.write(f"최적 평균 검증 MSE: {-grid_result.best_score_}\n")
    f.write(f"최적 모델 구조: 8-{optimal_units1}-{optimal_units2}-1, activation: {optimal_activation}\n")

# JSON으로도 저장하여 파이썬에서 쉽게 로드 가능하도록 함
best_json_path = os.path.join(output_dir, 'gridsearch_best_params.json')
best_payload = {
    "best_params": grid_result.best_params_,
    "best_score_mse": float(-grid_result.best_score_),  # 긍정 MSE 값
    "model": {
        "input_dim": 8,
        "units1": int(optimal_units1),
        "units2": int(optimal_units2),
        "activation": str(optimal_activation),
        "output_dim": 1
    },
    "note": "scoring은 'neg_mean_squared_error' 였습니다. best_score_mse는 양의 MSE 값입니다 (작을수록 좋음)."
}
with open(best_json_path, 'w', encoding='utf-8') as jf:
    json.dump(best_payload, jf, ensure_ascii=False, indent=2)

print("\n=== GridSearchCV 상세 결과 ===")
print(cv_results_sorted)  # grid_result.cv_results_ 의 값 중 3개의 변수만 뽑아서 보기
print(f"\n[저장 완료] CV 결과: {cv_results_path}")
print(f"[저장 완료] CV 정렬 결과: {cv_results_sorted_path}")
print(f"[저장 완료] 최적 요약: {best_summary_path}")
print(f"[저장 완료] 최적 하이퍼파라미터(JSON): {best_json_path}")
# .sort_values =pandas.DataFrame을 특정 열의 값을 기준으로 재배치
#  mean_test_score을 기준으로 ascending=False: 내림차순 정렬 -- 성능이 가장 좋은 것이 위로 간다


# 9. 최적 모델 구조 보여주기
print(f"최적 모델 구조:8- {optimal_units1}-{optimal_units2}-1, activation: {optimal_activation}")

# 7. (선택) 결과 table을 보기 좋게 정리
# import pandas as pd
# cv_results = pd.DataFrame(grid_result.cv_results_)
# display(cv_results.sort_values('mean_test_score', ascending=False))

#%%
def create_optimal_model():
    model = Sequential([
        Input(shape=(8,)),
        Dense(optimal_units1, activation=optimal_activation),
        Dense(optimal_units2, activation=optimal_activation),
        Dense(1, activation='linear')
    ])

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    return model


ann_model = create_optimal_model()  # 100 epoch 기준 최적 모델 구조 생성

print("--- 모델 학습을 시작합니다 ---")

early_stopping = EarlyStopping(  # 최적 모델 학습 (Training)
    monitor='val_loss',  # 검증 손실 기준 = val_loss = validation set의 RMSE
    patience=20,  # 개선되지 않는 epoch 20회 동안 기다리기(8~20 추천)
    restore_best_weights=True  # 가장 좋은 가중치로 복원
)

history = ann_model.fit(
    X_train_norm,
    y_train_norm,
    epochs=1000,  # 충분히 큰 값으로 설정
    batch_size=optimal_batch_size,  # # GridSearchCV 결과 활용 --- 아까 32라고 정의
    validation_data=(X_val_norm, y_val_norm),
    callbacks=[early_stopping],  # callback 중 1개인 Early_stopping 이용/ overfitting 뜨기 전에 earlystopping 으로 끊기
    verbose=0  # 각 epoch마다 진행상황 표현
)
print("\n--- 모델 학습 완료 ---")

# 3. 모델 성능 평가 (Evaluation)
print("\n--- 테스트 데이터 이용 모델 성능 평가 ---")
test_loss, test_mae = ann_model.evaluate(X_test_norm, y_test_norm,
                                         verbose=0)  # evaluate 함수 이용 / verbose = 0 이라 평과과정 출력 X / model을 compile 할 떄 정의한 loss 와 metrics 출력
print(f"테스트 데이터 손실 (MSE): {test_loss:.4f}")
print(f"테스트 데이터 평균 절대 오차 (MAE): {test_mae:.4f}")

# 예측한 값 출력
y_pred_norm = ann_model.predict(X_test_norm)  # predict(입력변수) 함수는 모델의 출력값(예측값) 출력
Y_pred = scaler_y.inverse_transform(y_pred_norm)  # 역변환을 통해 Y_pred를 구함
Y_test_true = y_test  # 가독성을 위해...

# 상관계수 R^2 출력
r2 = r2_score(Y_test_true, Y_pred)  # r2_score 함수를 이용해 결정계수 R^2을 구함
print(f"테스트 데이터 결정계수 (R²): {r2:.4f}")  # r2를 소수점 뒤에 4자리까지만 출력

# best_epoch 출력 + 최소 val_loss 출력
best_epoch = np.argmin(
    history.history['val_loss']) + 1  # 가장 val_loss 가 낮았던 epoch 수를 찾기 // np.argmin 함수는 리스트에서 가장 작은 값의 인덱스(위치) 출력
print(
    f"최적의 epoch: {best_epoch}")  # 위에 것을 이어 쓰자면 model.fit.history이다. 뒤의 .history를 통해 모든 훈련과정의 기록(log)를 출력 // 그 중 val_loss 만 뽑아서 보기 -- 그 중 가장 작은 값의 인덱스 출력 np.argmin
print(
    f"최소 val_loss: {history.history['val_loss'][best_epoch - 1]:.5f}")  # .history에서 np.argmin까지 하면 index 번호에 자동으로 +1을 해서 인간이 보기 편하도록 출력을 해줌 --- -1 해줘서 index 번호 넣어주기

# 5. 학습 과정 시각화 (Loss, MAE, R²)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21,
                                                   5))  # plt.subplots(1, 3) = 1행 3열로 전체 그래프 부분 나누기  // figsize는 전체 그래프 크기 // 전체 그래프 구간을 나눠서 각각 할당해서 그릴 때 subplot 이용

# ax1 = 첫번쨰 그림판 --  Loss 그래프 그리기
ax1.plot(history.history['loss'],
         label='Training Loss')  # 참고로 history = model.fit() 이라고 앞에서 정의를 해놓음 // 범례(legend)를 Training loss 라 이름지음
ax1.plot(history.history['val_loss'], label='Validation Loss')  # model.fit이 진행될때마다 loss 와 val_loss를 그리기
ax1.set_title('Loss (MSE)')  # set_title = 그래프 제목
ax1.set_xlabel('Epoch');
ax1.set_ylabel('Loss');
ax1.legend();
ax1.grid(True)  # .legend() --- 아까 label로 이름 붙여놓은 애들 표시 // .grid(True) -- 격자무늬 추가

# ax2 = MAE
ax2.plot(history.history['mean_absolute_error'], label='Training MAE')  # 아까 model.compile 할 떄 MAE 미리 선언함
ax2.plot(history.history['val_mean_absolute_error'],
         label='Validation MAE')  # Keras는 자동으로 validation set의 키 값 앞에 접두사 val_을 붙여준다
ax2.set_title('Mean Absolute Error (MAE)')
ax2.set_xlabel('Epoch');
ax2.set_ylabel('MAE');
ax2.legend();
ax2.grid(True)

# ax3 = R² 수기계산(별도 그래프)
ax3.plot([r2] * len(history.history['loss']), label=f'Test R² = {r2:.3f}')  # r2를 총 epoch 수만큼 만듦 -- 그래프 직선형으로 그리려고
ax3.set_title('Test R-squared')
ax3.set_xlabel('Epoch');
ax3.set_ylabel('R²');
ax3.legend();
ax3.grid(True)
ax3.set_ylim(bottom=0, top=1)

plt.tight_layout()  # layout 자동으로 깔끔하게 그래프 보여줌
plt.show()  # 화면에 그래프 출력

# 6. 실제값-예측값 산점도
plt.figure(figsize=(7, 7))
plt.scatter(Y_test_true, Y_pred, alpha=0.5)  # alpha는 투명도를 의미 .scatter()을 통해 산점도 그리기 . x 값은 실제값 y 값은 예측값

# y = x 기준선 그리기
plt.plot([Y_test_true.min(), Y_test_true.max()],  # 선의 시작과 끝점 x좌표
         [Y_test_true.min(), Y_test_true.max()], 'r--', lw=2,
         label='Perfect')  # 선의 시작과 끝점 y좌표 --- x좌표와 같으므로 y = x 그래프를 그린다  # lw = linewidth = 선 굵기
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title(f"Actual vs. Predicted (R² = {r2:.3f})")
plt.legend();
plt.grid(True);
plt.axis('equal')  # plt.axis('equal')을 통해 y=x에
plt.show()

# 7. 잔차 플롯 (residuals)
residuals = (Y_test_true.flatten() - Y_pred.flatten())  # residuals = 실제값 - 예측값
plt.figure(figsize=(8, 6))  # 그래프 크기
plt.scatter(Y_pred, residuals, alpha=0.6)  # x축 값 = 예측값, y축 값 = (실제값- 예측값)
plt.axhline(0, color='red', linestyle='--')  # axhline = 수평선(Axis Horizontal Line)**을 그리는 함수
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.grid(True)  # 뒤에 격자 표시
plt.show()

# 모델 성능 요약
print(f"\n=== 최종 모델 성능 요약 ===")
print(f"모델 구조: 8-{optimal_units1}-{optimal_units2}-1")
print(f"활성화 함수: {optimal_activation}")
print(f"최적 epoch: {best_epoch}")
print(f"테스트 R²: {r2:.4f}")
print(f"테스트 MAE: {test_mae:.4f}")

#%%
# 1. SHAP Explainer가 사용할 배경 데이터(background data)를 준비합니다.
#    훈련 데이터셋(X_train_scaled)의 특성을 대표하도록 100개의 데이터를 요약(kmeans)합니다.
#    이는 계산 효율성을 위해 필수적인 과정입니다.
print("배경 데이터셋을 생성합니다. (시간이 조금 걸릴 수 있습니다)")
background_data = shap.kmeans(X_train_norm, 100)
print("배경 데이터셋 생성 완료!")

# 2. MLP 모델(ann_model)을 해석할 KernelExplainer를 생성합니다.
#    - 첫 번째 인자: 모델의 예측 함수(ann_model.predict)를 전달합니다.
#    - 두 번째 인자: 위에서 만든 배경 데이터를 전달합니다.
print("SHAP KernelExplainer를 생성합니다...")
explainer = shap.KernelExplainer(ann_model.predict, background_data)
print("Explainer 생성 완료!")

# 생성된 explainer 객체 확인
print("\nExplainer 객체 정보:")
print(explainer)

print("SHAP 값 계산을 시작합니다. (시간이 몇 분 정도 소요될 수 있습니다)")
# 테스트 데이터셋(X_test_scaled)에 대한 SHAP 값을 계산합니다.
shap_values = explainer.shap_values(X_test_norm, silent = True)  # 학습과정 안보고 싶으면 여기다가 , silent = True 붙이기 // 학습과정 보고 싶으면 그냥 silent = True 지우기
print("SHAP 값 계산 완료!")

# 계산된 SHAP 값의 형태(shape) 확인
print(f"\n계산된 SHAP 값의 형태: {shap_values.shape}")

#%%
shap_values_corrected = np.squeeze(shap_values)

# 2. 수정된 배열을 사용해 SHAP Explanation 객체를 다시 생성합니다.
print("\n2. 수정된 데이터로 SHAP Explanation 객체를 생성합니다...")
shap_explanation_corrected = shap.Explanation(
    values=shap_values_corrected,
    base_values=explainer.expected_value,
    data=X_test_norm,
    feature_names=name_X
)
print("   => 객체 생성 완료.")

shap.summary_plot(shap_explanation_corrected)
plt.title(name_y)
plt.show()

shap.plots.waterfall(shap_explanation_corrected[0])   #[0]을 바꿔가면서 보기
plt.title(name_y)
plt.show()

shap.dependence_plot(
    "CO conversion",  # <--- 'conversion'이 아니라 'coversion'으로 오타를 맞춰줍니다.
    shap_explanation_corrected.values,
    X_test_norm,
    feature_names=name_X,
    interaction_index="auto"
)
plt.title(name_y)
plt.show()

shap.plots.bar(shap_explanation_corrected)
plt.title(name_y)
plt.show()
