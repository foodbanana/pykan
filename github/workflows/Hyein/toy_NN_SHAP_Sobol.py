import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from SALib.sample import sobol as saltelli
from SALib.analyze import sobol
from github.workflows.Hyein.toy_analytic_SHAP_Sobol import FUNCTION_ZOO, plot_custom_bars

def main():
    # ==========================================
    # 2. Argument Parsing & Setup
    # ==========================================
    parser = argparse.ArgumentParser(description="Analyze MLP models trained on analytical functions.")
    parser.add_argument("func_name", type=str, nargs='?', default="original",
                        choices=FUNCTION_ZOO.keys(),
                        help="Choose a function: " + ", ".join(FUNCTION_ZOO.keys()))

    args = parser.parse_args()
    case_name = args.func_name

    print(f"🚀 Running MLP Analysis for case: '{case_name}'")

    # Paths
    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein', "analytical_results", case_name)
    model_dir = os.path.join(root_dir, "nn_models")

    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"❌ Error: Model directory not found at {model_dir}")
        print("   Please run the training script (tune_analytical_mlp.py) first.")
        return

    # ==========================================
    # 3. Load Model and Scalers
    # ==========================================
    try:
        model_path = os.path.join(model_dir, f'{case_name}_best_mlp_model.pkl')
        scaler_X_path = os.path.join(model_dir, f'{case_name}_mlp_scaler_X.pkl')
        scaler_y_path = os.path.join(model_dir, f'{case_name}_mlp_scaler_y.pkl')

        mlp_model = joblib.load(model_path)
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        print(f"✅ Loaded model and scalers from {model_dir}")
    except FileNotFoundError as e:
        print(f"❌ Error loading files: {e}")
        return

    # ==========================================
    # 4. Define Prediction Wrapper
    # ==========================================
    def mlp_wrapper(X_raw):
        # 1. Scale Input: Physical -> Normalized
        X_norm = scaler_X.transform(X_raw)
        # 2. Predict
        y_pred_norm = mlp_model.predict(X_norm)
        # 3. Inverse Scale Output: Normalized -> Physical
        y_pred_raw = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
        return y_pred_raw

    # Load Config
    config = FUNCTION_ZOO[case_name]
    bounds = config["bounds"]
    feature_names = config["names"]
    n_features = len(bounds)

    # Output Directory for Analysis
    analysis_savepath = os.path.join(root_dir, "nn_analysis")
    os.makedirs(analysis_savepath, exist_ok=True)

    # ==========================================
    # 5. Sobol Analysis
    # ==========================================
    print("\n[1/2] 🎲 Running Sobol Analysis on MLP...")

    problem = {
        'num_vars': n_features,
        'names': feature_names,
        'bounds': bounds
    }

    # Generate samples (Physical Domain)
    X_sobol = saltelli.sample(problem, 1024, calc_second_order=True)

    # Predict using Wrapper
    Y_sobol = mlp_wrapper(X_sobol)

    # Analyze
    Si = sobol.analyze(problem, Y_sobol, calc_second_order=True)

    # Create DataFrame (PRESERVE ORDER: Do not sort)
    results_df = pd.DataFrame({
        'Feature': feature_names,
        'Total_Effect (ST)': Si['ST'],
        'First_Order (S1)': Si['S1']
    })

    print(results_df)
    results_df.to_csv(os.path.join(analysis_savepath, "mlp_sobol_indices.csv"), index=False)

    # Plot using Custom Style
    plot_custom_bars(
        names=results_df['Feature'],
        values=results_df['First_Order (S1)'],
        title=f"Sobol Sensitivity (MLP) - {case_name}",
        ylabel="First Order Index (S1)",
        savepath=os.path.join(analysis_savepath, "mlp_sobol_plot.png"),
        color='bisque'
    )

    # ==========================================
    # 6. SHAP Analysis
    # ==========================================
    print("\n[2/2] 🔍 Running SHAP Analysis on MLP...")

    # 1. Background (Physical Domain)
    X_representative = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(2000, n_features)
    )
    # Summarize to 100 weighted points
    X_bg = shap.kmeans(X_representative, 100)

    # 2. Test Data (Physical Domain)
    X_test = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(500, n_features)
    )

    # Use the wrapper for the explainer
    explainer = shap.KernelExplainer(mlp_wrapper, X_bg)
    shap_values = explainer.shap_values(X_test)

    # Save Raw SHAP
    pd.DataFrame(shap_values, columns=feature_names).to_csv(
        os.path.join(analysis_savepath, "mlp_shap_raw.csv"), index=False
    )

    # Calculate Mean Absolute SHAP (Global Importance)
    mean_shap = np.array(shap_values).mean(axis=0)
    abs_mean_shap = np.abs(mean_shap)

    # Create DataFrame (PRESERVE ORDER: Do not sort)
    shap_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_SHAP': mean_shap,
        'Abs_Mean_SHAP': abs_mean_shap
    })

    shap_importance_df.to_csv(os.path.join(analysis_savepath, "mlp_shap_mean_abs.csv"), index=False)

    # Plot 1: Dot Plot (Standard SHAP library plot)
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(analysis_savepath, "mlp_shap_dot_plot.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Custom Bar Plot (Replaces shap.summary_plot(plot_type="bar"))
    plot_custom_bars(
        names=shap_importance_df['Feature'],
        values=shap_importance_df['Abs_Mean_SHAP'],
        title=f"SHAP Global Importance (MLP) - {case_name}",
        ylabel="mean(|SHAP value|)",
        savepath=os.path.join(analysis_savepath, "mlp_shap_bar_plot.png"),
        color='thistle'
    )

    print(f"✅ Analysis Complete! Results saved in: {analysis_savepath}")


if __name__ == "__main__":
    main()