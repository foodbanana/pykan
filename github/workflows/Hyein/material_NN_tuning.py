import argparse
import os
import joblib
import json  # <--- [NEW] Import json library
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from kan.custom_processing import remove_outliers_iqr
import matplotlib.pyplot as plt

def main():
    # ==========================================
    # 2. Argument Parsing & Setup
    # ==========================================
    parser = argparse.ArgumentParser(description="Tune MLP for a specific dataset.")
    parser.add_argument("data_name", type=str, nargs='?', default="P3HT",
                        help="The name of the dataset (default: P3HT)")
    parser.add_argument("rand_seed", type=int, nargs='?', default=None,
                        help="The random seed (default: None=42)")

    args = parser.parse_args()
    data_name = args.data_name
    rand_seed = args.rand_seed

    print(f"🚀 Starting MLP Tuning for: '{data_name}' with seed={rand_seed}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   - Device: {device}")

    # Output Directory
    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein')
    filepath = os.path.join(root_dir, "data", f"{data_name}.csv")
    if rand_seed is not None:
        savepath = os.path.join(root_dir, "material_nn_models", data_name + f"_seed_{rand_seed}")
    else:
        savepath = os.path.join(root_dir, "material_nn_models", data_name)
        rand_seed = 42
    os.makedirs(savepath, exist_ok=True)

    # Check if file exists
    if not os.path.exists(filepath):
        print(f"❌ Error: Data file not found at {filepath}")
        return

    filedata = pd.read_csv(filepath)
    name_X = filedata.columns[:-1].tolist()
    name_y = filedata.columns[-1]
    df_in = filedata[name_X]
    df_out = filedata[[name_y]]
    print(f"TARGET: {name_y}")

    # TODO: Data가 너무 많이 지워지긴 함
    df_in_final, df_out_final = remove_outliers_iqr(df_in, df_out)

    removed_count = len(df_in) - len(df_in_final)
    print(f"# of data after removing outliers: {len(df_in_final)} ({removed_count} removed)")

    X = df_in_final[name_X].values
    y = df_out_final[name_y].values.reshape(-1, 1)

    X_temp_denorm, X_test_denorm, y_temp_denorm, y_test_denorm = train_test_split(X, y, test_size=0.2, random_state=rand_seed)
    X_train_denorm, X_val_denorm, y_train_denorm, y_val_denorm = train_test_split(X_temp_denorm, y_temp_denorm,
                                                                                  test_size=0.2, random_state=rand_seed)
    print(f"Train/Validation/Test : {len(X_train_denorm)} / {len(X_val_denorm)} / {len(X_test_denorm)}")

    feat_names = name_X
    nx = len(name_X)

    scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
    scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))

    X_train_norm = scaler_X.fit_transform(X_train_denorm)
    y_train_norm = scaler_y.fit_transform(y_train_denorm)

    X_test_norm = scaler_X.transform(X_test_denorm)
    y_test_norm = scaler_y.transform(y_test_denorm)

    # ==========================================
    # 4. Hyperparameter Tuning (RandomizedSearchCV)
    # ==========================================
    param_distributions = {
        'hidden_layer_sizes': [
            (128,),
            (64, 64),
            (128, 64),
            (100, 100),
            (64, 32, 16),
            (64, 32, 16, 16),
        ],
        'activation': ['relu', 'tanh'],  # tanh is often good for smooth analytical functions
        'solver': ['adam', 'lbfgs'],  # lbfgs is great for noise-free/low-noise math functions
        'alpha': [1.0, 10.],
        'learning_rate_init': [0.0005, 0.001, 0.01, 0.1]
    }

    mlp = MLPRegressor(max_iter=10000, random_state=rand_seed)  # Increased max_iter for convergence

    search = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=param_distributions,
        n_iter=50,  # Tried 50 combinations
        cv=3,  # Internal Cross Validation
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        random_state=rand_seed
    )

    print("\n🏎️  Starting Randomized Hyperparameter Search...")
    # RAVEL IS REQUIRED: y must be (N,) not (N,1) for sklearn
    search.fit(X_train_norm, y_train_norm.ravel())

    # ==========================================
    # 5. Evaluation & Reporting
    # ==========================================
    best_model = search.best_estimator_

    print("\n" + "=" * 40)
    print(f"🏆 Best Parameters: {search.best_params_}")
    print("=" * 40)

    # Evaluate on TEST Set (Normalized scale)
    y_pred_test_norm = best_model.predict(X_test_norm)
    r2_test = r2_score(y_test_norm, y_pred_test_norm)

    y_pred_test_denorm = scaler_y.inverse_transform(y_pred_test_norm.reshape([-1, 1]))

    # Plotting
    plt.figure(figsize=(4, 4))
    plt.scatter(y_test_denorm, y_pred_test_denorm, alpha=0.6, color='skyblue', edgecolors='k', s=30, label='Test Data')

    # Add Identity Line (Perfect Prediction)
    min_val = min(min(y_test_denorm), min(y_pred_test_denorm))
    max_val = max(max(y_test_denorm), max(y_pred_test_denorm))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')

    plt.xlabel(f'Actual {name_y}')
    plt.ylabel(f'Predicted {name_y}')
    plt.title(f'Parity Plot: {data_name} ($R^2 = {r2_test:.3f}$)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the plot
    parity_path = os.path.join(savepath, f"{data_name}_mlp_parity_at_training_plot.png")
    plt.savefig(parity_path, dpi=300, bbox_inches='tight')
    plt.close()


    print(f"📊 [Test Set]       R2 Score: {r2_test:.4f}")

    if r2_test < 0.8:
        print("⚠️  Warning: R2 is low. Try increasing 'n_iter' or checking 'noise_level'.")

    # ==========================================
    # 6. Save Models & Scalers
    # ==========================================
    joblib.dump(best_model, os.path.join(savepath, f'{data_name}_best_mlp_model.pkl'))
    joblib.dump(scaler_X, os.path.join(savepath, f'{data_name}_mlp_scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(savepath, f'{data_name}_mlp_scaler_y.pkl'))

    metrics_data = {
            "dataset": data_name,
            "test_r2": r2_test,
            "best_params": search.best_params_
        }

    json_path = os.path.join(savepath, f'{data_name}_metrics.json')
    with open(json_path, "w") as f:
        json.dump(metrics_data, f, indent=4)  # indent=4 makes it pretty and readable

if __name__ == "__main__":
    main()
