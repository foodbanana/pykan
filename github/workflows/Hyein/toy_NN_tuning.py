import argparse
import os
import joblib
import json  # <--- [NEW] Import json library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from github.workflows.Hyein.toy_analytic_SHAP_Sobol import FUNCTION_ZOO


def generate_synthetic_data(func_config, n_samples=2000, noise_level=0.05):
    """
    Generates X and y data based on the analytical function.
    Adds random noise to make it a proper regression task.
    """
    bounds = func_config['bounds']
    func = func_config['func']
    n_features = len(bounds)

    # 1. Generate Uniform Random Inputs (X)
    X = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(n_samples, n_features)
    )

    # 2. Calculate True Output (y)
    # apply_along_axis allows the lambda to process row by row
    y = np.apply_along_axis(func, 1, X).reshape(-1, 1)

    # 3. Add Gaussian Noise (Simulating real-world imperfection)
    # noise_level: percentage of standard deviation to add as noise
    noise = np.random.normal(0, np.std(y) * noise_level, size=y.shape)
    y_noisy = y + noise

    return X, y_noisy


def main():
    # ==========================================
    # 2. Argument Parsing & Setup
    # ==========================================
    parser = argparse.ArgumentParser(description="Tune MLP for Analytical Functions.")
    parser.add_argument("func_name", type=str, nargs='?', default="original",
                        choices=FUNCTION_ZOO.keys(),
                        help="Choose a function: " + ", ".join(FUNCTION_ZOO.keys()))

    args = parser.parse_args()
    data_name = args.func_name

    print(f"🚀 Starting MLP Tuning for Analytical Case: '{data_name}'")

    # Directory Setup
    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein', 'analytical_results', data_name)
    savepath = os.path.join(root_dir, "nn_models")
    os.makedirs(savepath, exist_ok=True)

    # ==========================================
    # 3. Data Generation & Preprocessing
    # ==========================================
    config = FUNCTION_ZOO[data_name]

    # Generate 2000 samples (enough for reliable splitting)
    print("🎲 Generating synthetic data...")
    X, y = generate_synthetic_data(config, n_samples=2000, noise_level=0.01)  # 1% noise

    # Step 1: Split off 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"   - Train set: {len(X_train)} samples")
    print(f"   - Test set : {len(X_test)} samples")

    # Scaling (As per your request: 0.1 ~ 0.9 range)
    scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
    scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))

    X_train_norm = scaler_X.fit_transform(X_train)
    y_train_norm = scaler_y.fit_transform(y_train)

    X_test_norm = scaler_X.transform(X_test)
    y_test_norm = scaler_y.transform(y_test)

    # ==========================================
    # 4. Hyperparameter Tuning (RandomizedSearchCV)
    # ==========================================
    param_distributions = {
        'hidden_layer_sizes': [
            (64, 64),
            (128, 64),
            (100, 100),
            (64, 32, 16),
            (128,)
        ],
        'activation': ['relu', 'tanh'],  # tanh is often good for smooth analytical functions
        'solver': ['adam', 'lbfgs'],  # lbfgs is great for noise-free/low-noise math functions
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate_init': [0.001, 0.01, 0.0005]
    }

    mlp = MLPRegressor(max_iter=10000, random_state=42)  # Increased max_iter for convergence

    search = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=param_distributions,
        n_iter=50,  # Tried 50 combinations
        cv=3,  # Internal Cross Validation
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        random_state=42
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

    print(f"📊 [Test Set]       R2 Score: {r2_test:.4f}")

    if r2_test < 0.9:
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
