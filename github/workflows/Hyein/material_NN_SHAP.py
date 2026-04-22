import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
import joblib
import torch
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from SALib.sample import sobol as sample_sobol
from SALib.analyze import sobol as analyze_sobol
from sklearn.metrics import pairwise_distances_argmin_min

# Assuming this custom module exists in your environment
from kan.custom_processing import remove_outliers_iqr
from github.workflows.Hyein.toy_NN_SHAP_Sobol import plot_custom_bars

def main():
    # ==========================================
    # 0. Argument Parsing
    # ==========================================
    parser = argparse.ArgumentParser(description="Run SHAP and Sobol analysis for a specific dataset.")
    parser.add_argument("data_name", type=str, nargs='?', default="CO2RRLCA",
                        help="The name of the dataset (default: CO2RRLCA)")
    parser.add_argument("rand_seed", type=int, nargs='?', default=None)

    # Optional: Add flags for specific settings if needed
    # parser.add_argument("--nsamples", type=int, default=100, help="Number of samples for SHAP")

    args = parser.parse_args()
    data_name = args.data_name
    rand_seed = args.rand_seed

    print(f"🚀 Starting analysis for: {data_name}")

    # ==========================================
    # 1. Data Loading & Preprocessing
    # ==========================================
    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein')
    filepath = os.path.join(root_dir, "data", f"{data_name}.csv")
    if rand_seed is not None:
        savepath = os.path.join(root_dir, "material_nn_models", data_name + f"_seed_{rand_seed}")
    else:
        savepath = os.path.join(root_dir, "material_nn_models", data_name)


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

    df_in_final, df_out_final = remove_outliers_iqr(df_in, df_out)

    removed_count = len(df_in) - len(df_in_final)
    print(f"# of data after removing outliers: {len(df_in_final)} ({removed_count} removed)")

    X = df_in_final[name_X].values
    y = df_out_final[name_y].values.reshape(-1, 1)

    X_temp_denorm, X_test_denorm, y_temp_denorm, y_test_denorm = train_test_split(X, y, test_size=0.2, random_state=42)

    feature_names = name_X
    num_train_data = X_temp_denorm.shape[0]

    # ==========================================
    # 2. Load Models & Scalers
    # ==========================================
    model_path = os.path.join(savepath, f'{data_name}_best_mlp_model.pkl')

    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return

    loaded_model = joblib.load(model_path)
    scaler_X = joblib.load(os.path.join(savepath, f'{data_name}_mlp_scaler_X.pkl'))
    scaler_y = joblib.load(os.path.join(savepath, f'{data_name}_mlp_scaler_y.pkl'))

    # Apply scaling
    X_temp_norm = scaler_X.transform(X_temp_denorm)
    X_test_norm = scaler_X.transform(X_test_denorm)

    # ==========================================
    # 2.5 Parity Plot (Actual vs Predicted)
    # ==========================================
    from sklearn.metrics import r2_score

    # Predict on test set
    y_pred_norm = loaded_model.predict(X_test_norm)

    # Inverse transform to original units
    y_test_denorm_flatten = y_test_denorm.reshape(-1, 1).flatten()
    y_pred_denorm = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()

    # Calculate R2
    r2 = r2_score(y_test_denorm_flatten, y_pred_denorm)

    # Plotting
    plt.figure(figsize=(4, 4))
    plt.scatter(y_test_denorm_flatten, y_pred_denorm, alpha=0.6, color='skyblue', edgecolors='k', s=30, label='Test Data')

    # Add Identity Line (Perfect Prediction)
    min_val = min(min(y_test_denorm_flatten), min(y_pred_denorm))
    max_val = max(max(y_test_denorm_flatten), max(y_pred_denorm))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')

    plt.xlabel(f'Actual {name_y}')
    plt.ylabel(f'Predicted {name_y}')
    plt.title(f'Parity Plot: {data_name} ($R^2 = {r2:.3f}$)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the plot
    parity_path = os.path.join(savepath, f"{data_name}_mlp_parity_plot.png")
    plt.savefig(parity_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Parity plot saved to: {parity_path} (R2: {r2:.4f})")

    # ==========================================
    # 3. SHAP Analysis
    # ==========================================
    num_data = len(X_temp_norm)
    num_shap_sample = 1000

    num_unique_rows = np.unique(X_temp_norm, axis=0).shape[0]
    n_clusters = min(num_shap_sample, num_unique_rows)

    if n_clusters < num_data:
        X_train_summary = shap.kmeans(X_temp_norm, n_clusters)
    else:
        X_train_summary = X_temp_norm

    explainer = shap.KernelExplainer(loaded_model.predict, X_train_summary)

    shap_values = explainer.shap_values(X_test_norm)

    # [NEW] Save SHAP Values to CSV
    # 1. Raw SHAP values (useful for reproducing dot plots)
    shap_raw_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_raw_path = os.path.join(savepath, f'{data_name}_shap_values_raw.csv')
    shap_raw_df.to_csv(shap_raw_path, index=False)

    # 2. SHAP Summary/Importance (Mean Absolute Value - mirrors the Bar Plot)
    mean_shap = np.array(shap_values).mean(axis=0)
    abs_mean_shap = np.abs(mean_shap)
    mean_avg_shap = np.abs(shap_values.mean(axis=0))

    shap_summary_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_SHAP': mean_shap,
        'Abs_Mean_SHAP': abs_mean_shap,
        'Mean_Abs_SHAP': mean_avg_shap,
    })

    shap_summary_path = os.path.join(savepath, f'{data_name}_shap_importance.csv')
    shap_summary_df.to_csv(shap_summary_path, index=False)
    print("[SHAP Analysis Result]")
    print(shap_summary_df.sort_values(by='Mean_Abs_SHAP', ascending=False))

    # Plot 1: Bar Plot
    plot_custom_bars(
        names=shap_summary_df['Feature'],
        values=shap_summary_df['Mean_Abs_SHAP'],
        title=f"SHAP Global Importance (MLP) - {data_name}",
        ylabel="mean(|SHAP value|)",
        savepath=os.path.join(savepath, f"{data_name}_mlp_shap_bar_plot.png"),
        color='thistle'
    )

    # Plot 2: Dot Plot
    shap.summary_plot(
        shap_values,
        X_test_norm,
        feature_names=feature_names,
        show=False  # <--- Crucial change
    )

    # 3. Save the figure BEFORE calling plt.show() or plt.close()
    save_file = os.path.join(savepath, f'{data_name}_shap_dot_plot.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')

    # 4. Now you can close it to free up memory
    plt.close()

    # ==========================================
    # 4. SALib (Sobol) Analysis
    # ==========================================
    n_features = X_temp_norm.shape[1]

    # [Dynamic Bounds Adjustment]
    # We use the actual feature_range from the loaded scaler.
    # If scaler was (-1, 1), it uses that. If (0, 1), it uses that.
    scaler_min, scaler_max = scaler_X.feature_range
    print(f"ℹ️  Sobol Bounds set to scaler range: [{scaler_min}, {scaler_max}]")

    problem = {
        'num_vars': n_features,
        'names': feature_names,
        'bounds': [[scaler_min, scaler_max]] * n_features
    }

    power2 = 1
    while power2 < num_train_data:
        power2 *= 2
    N = power2
    X_sobol = sample_sobol.sample(problem, N, calc_second_order=True)

    print(f"⏳ Running Sobol analysis on {X_sobol.shape[0]} samples...")
    Y_sobol = loaded_model.predict(X_sobol)

    Si = analyze_sobol.analyze(problem, Y_sobol, calc_second_order=True)

    # Results
    print("\n[Sobol Analysis Result]")
    results_df = pd.DataFrame({
        'Feature': feature_names,
        'Total_Effect (ST)': Si['ST'],
        'First_Order (S1)': Si['S1']
    })

    print(results_df.sort_values(by='First_Order (S1)', ascending=False))

    # [NEW] Save Sobol Indices to CSV
    sobol_csv_path = os.path.join(savepath, f"{data_name}_sobol_indices.csv")
    results_df.to_csv(sobol_csv_path, index=False)

    # Plot
    plot_custom_bars(
        names=results_df['Feature'],
        values=results_df['First_Order (S1)'],
        title=f"Sobol Sensitivity (MLP) - {data_name}",
        ylabel="First Order Index (S1)",
        savepath=os.path.join(savepath, f"{data_name}_mlp_sobol_plot.png"),
        color='bisque'
    )

if __name__ == "__main__":
    main()
