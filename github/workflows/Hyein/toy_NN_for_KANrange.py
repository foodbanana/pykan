import argparse
import os

# [FIX] Set this BEFORE importing numpy/sklearn to fix the MKL memory leak warning
os.environ["OMP_NUM_THREADS"] = "1"

import joblib
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from SALib.sample import sobol as sample_sobol
from SALib.analyze import sobol as analyze_sobol

from github.workflows.Hyein.toy_analytic_SHAP_Sobol import FUNCTION_ZOO
from github.workflows.Hyein.toy_NN_SHAP_Sobol import plot_custom_bars


def tune_and_analyze_subset(X_sub, y_sub, feat_names, scaler_X, save_dir, range_label):
    """
    1. Tunes an MLP on the provided subset (X_sub, y_sub).
    2. Runs SHAP analysis.
    3. Runs Sobol analysis.
    4. Saves all results to save_dir.
    """
    print(f"\n   ⚙️  Tuning MLP for range: {range_label} (Samples: {len(X_sub)})")

    # ---------------------------------------------------------
    # A. Hyperparameter Tuning (MLP)
    # ---------------------------------------------------------
    if len(X_sub) < 10:
        print("      ⚠️ Skipping: Not enough data points to train/tune.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)

    # If dataset is very small, reduce complexity
    n_iter_search = 30 if len(X_train) > 50 else 20
    cv_folds = 3 if len(X_train) > 20 else 2

    param_distributions = {
        'hidden_layer_sizes': [(64, 64), (128, 64), (100, 100), (64, 32, 16), (128,)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'lbfgs'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate_init': [0.001, 0.01, 0.0005]
    }

    mlp = MLPRegressor(max_iter=5000, random_state=42)

    search = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=param_distributions,
        n_iter=n_iter_search,
        cv=cv_folds,
        scoring='r2',
        n_jobs=-1,
        verbose=0,
        random_state=42
    )

    try:
        search.fit(X_train, y_train.ravel())
    except ValueError as e:
        print(f"      ❌ Tuning failed (likely too few samples for CV): {e}")
        return

    best_model = search.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"      🏆 Best R2 on local test set: {r2:.4f}")

    # Save Model
    joblib.dump(best_model, os.path.join(save_dir, "best_mlp_model.pkl"))

    # Save Metrics
    metrics = {"range": range_label, "test_r2": r2, "best_params": search.best_params_, "n_samples": len(X_sub)}
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # ---------------------------------------------------------
    # B. SHAP Analysis
    # ---------------------------------------------------------
    print("      🔍 Running SHAP...")

    # [FIXED SECTION] Use random sampling instead of KMeans
    # KMeans crashes if unique points < 50. Random sampling is robust.
    num_background = 50
    if len(X_train) > num_background:
        # Randomly sample 50 points from X_train
        idx = np.random.choice(len(X_train), num_background, replace=False)
        background = X_train[idx]
    else:
        background = X_train

    explainer = shap.KernelExplainer(best_model.predict, background)

    # Limit test samples for speed (max 100)
    shap_test_data = X_test if len(X_test) < 100 else X_test[:100]
    shap_values = explainer.shap_values(shap_test_data)

    # Calculate Global Importance (Mean Abs)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({'Feature': feat_names, 'Mean_Abs_SHAP': mean_abs_shap})

    # Save CSV
    shap_df.to_csv(os.path.join(save_dir, "shap_importance.csv"), index=False)

    # Plot (Custom Style)
    plot_custom_bars(
        names=shap_df['Feature'],
        values=shap_df['Mean_Abs_SHAP'],
        title=f"SHAP Importance ({range_label})",
        ylabel="mean(|SHAP value|)",
        savepath=os.path.join(save_dir, "shap_bar_plot.png"),
        color='thistle'
    )

    # ---------------------------------------------------------
    # C. Sobol Analysis
    # ---------------------------------------------------------
    print("      🎲 Running Sobol...")
    # Define bounds based on the Scaler
    s_min, s_max = scaler_X.feature_range
    n_vars = len(feat_names)

    problem = {
        'num_vars': n_vars,
        'names': feat_names,
        'bounds': [[s_min, s_max]] * n_vars
    }

    # Generate Sobol samples
    # Warning: 1024 * (2D + 2) samples. For high dims, this is large.
    # If D is large, reduce N (e.g., to 512)
    X_sobol = sample_sobol.sample(problem, 1024, calc_second_order=True)
    Y_sobol = best_model.predict(X_sobol)
    Si = analyze_sobol.analyze(problem, Y_sobol, calc_second_order=True)

    # Save CSV
    sobol_df = pd.DataFrame({
        'Feature': feat_names,
        'Total_Effect (ST)': Si['ST'],
        'First_Order (S1)': Si['S1']
    })
    sobol_df.to_csv(os.path.join(save_dir, "sobol_indices.csv"), index=False)

    # Plot (Custom Style)
    plot_custom_bars(
        names=sobol_df['Feature'],
        values=sobol_df['First_Order (S1)'],
        title=f"Sobol Sensitivity ({range_label})",
        ylabel="First Order Index (S1)",
        savepath=os.path.join(save_dir, "sobol_plot.png"),
        color='bisque'
    )


def main():
    parser = argparse.ArgumentParser(description="Tune KAN for Analytical Functions.")
    parser.add_argument("func_name", type=str, nargs='?', default="exponential",
                        choices=FUNCTION_ZOO.keys(),
                        help="Choose a function from the ZOO.")

    args = parser.parse_args()
    case_name = args.func_name

    # 1. Setup Paths
    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein', 'analytical_results', case_name)
    split_data_path = os.path.join(root_dir, "kan_models", f"{case_name}_range_split_data.pkl")

    if not os.path.exists(split_data_path):
        print(f"❌ Error: Split data not found at {split_data_path}")
        print("   Please run the KAN range splitting script first.")
        return

    print(f"📂 Loading split data from: {split_data_path}")
    split_data = joblib.load(split_data_path)

    # Unpack Data
    dataset = split_data['dataset']  # Tensors
    masks = split_data['masks']  # List of Tensors (boolean)
    labels = split_data['labels']  # List of strings
    feat_names = split_data['feature_names']
    scaler_X = split_data['scaler_X']

    # 2. Iterate through Ranges
    print(f"\n🚀 Found {len(masks)} ranges. Starting analysis...")

    for i, (mask, label) in enumerate(zip(masks, labels)):
        # Create safe folder name
        safe_label = label.replace("<", "lt").replace(">", "gt").replace("=", "eq").replace(" ", "")
        range_dir = os.path.join(root_dir, f"range_{i}_{safe_label}")
        os.makedirs(range_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Processing Range {i}: {label}")
        print(f"Output Directory: {range_dir}")
        print(f"{'=' * 60}")

        # Extract Data for this Range
        # Convert Tensor -> Numpy for Scikit-Learn
        if torch.is_tensor(mask):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask

        if mask_np.sum() == 0:
            print("   ⚠️ Empty range. Skipping.")
            continue

        # Get the subset of data
        X_full = dataset['train_input'].cpu().numpy()
        y_full = dataset['train_label'].cpu().numpy()

        X_sub = X_full[mask_np]
        y_sub = y_full[mask_np]

        # Run Pipeline
        tune_and_analyze_subset(X_sub, y_sub, feat_names, scaler_X, range_dir, label)

    print(f"\n✅ All ranges processed. Results saved in {root_dir}")


if __name__ == "__main__":
    main()