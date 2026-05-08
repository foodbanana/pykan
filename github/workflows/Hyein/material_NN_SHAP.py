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
from sklearn.metrics import r2_score

# Assuming this custom module exists in your environment
from kan.custom_processing import remove_outliers_iqr
from github.workflows.Hyein.toy_NN_SHAP_Sobol import plot_custom_bars


# ==============================================================================
# Owen (2013) Pick-Freeze Bootstrap Sobol Estimator
# ------------------------------------------------------------------------------
# Reference:
#   Owen, A. B. (2013). Variance components and generalized Sobol' indices.
#   SIAM/ASA Journal on Uncertainty Quantification, 1(1), 19-41.
#   https://doi.org/10.1137/120876782
#
# Method (Pick-Freeze):
#   Given data matrix X (n x p) and model predictions f(X), the first-order
#   Sobol index for feature i is estimated as:
#
#       S_i  = [ (1/n) * sum_j f(A_j) * f(BA_j^(i)) - f0^2 ] / Var(Y)
#       ST_i = 1 - [ (1/n) * sum_j f(B_j) * f(AB_j^(i)) - f0^2 ] / Var(Y)
#
#   where:
#     A, B    : two independent bootstrap draws from X
#     BA^(i)  : B with column i replaced by column i from A  ("pick-freeze")
#     AB^(i)  : A with column i replaced by column i from B
#     f0      : grand mean of f(X)
#
#   Bootstrap resampling is applied K times to obtain mean estimates and
#   95% confidence intervals (2.5th / 97.5th percentile across resamples).
# ==============================================================================

def owen_sobol_bootstrap(model, X, n_bootstrap=500, seed=42):
    """
    Compute first-order (S1) and total-effect (ST) Sobol indices from real
    data using the Owen (2013) pick-freeze estimator with bootstrap CIs.

    Parameters
    ----------
    model       : fitted model with a .predict(X) method
    X           : np.ndarray, shape (n_samples, n_features) — normalized inputs
    n_bootstrap : int, number of bootstrap resamples (default 500)
    seed        : int, random seed for reproducibility

    Returns
    -------
    results : dict with keys
        'S1'      — np.ndarray (n_features,)   mean first-order indices
        'ST'      — np.ndarray (n_features,)   mean total-effect indices
        'S1_ci'   — np.ndarray (n_features, 2) lower/upper 95% CI for S1
        'ST_ci'   — np.ndarray (n_features, 2) lower/upper 95% CI for ST
        'S1_boot' — np.ndarray (n_bootstrap, n_features) all S1 resamples
        'ST_boot' — np.ndarray (n_bootstrap, n_features) all ST resamples
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape

    S1_boot = np.zeros((n_bootstrap, p))
    ST_boot = np.zeros((n_bootstrap, p))

    for b in range(n_bootstrap):
        # Draw two independent bootstrap samples of size n
        idx_A = rng.integers(0, n, size=n)
        idx_B = rng.integers(0, n, size=n)
        A = X[idx_A]   # (n, p)
        B = X[idx_B]   # (n, p)

        # Base predictions
        fA = model.predict(A)   # (n,)
        fB = model.predict(B)   # (n,)

        f0   = 0.5 * (fA.mean() + fB.mean())   # grand mean
        VarY = np.concatenate([fA, fB]).var()   # pooled variance

        if VarY < 1e-12:
            # Degenerate: model output is constant, all indices = 0
            S1_boot[b, :] = 0.0
            ST_boot[b, :] = 0.0
            continue

        for i in range(p):
            # Pick-freeze matrices
            # AB_i : A with column i replaced by B[:, i]  -> used for ST
            # BA_i : B with column i replaced by A[:, i]  -> used for S1
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]

            BA_i = B.copy()
            BA_i[:, i] = A[:, i]

            fAB_i = model.predict(AB_i)   # (n,)
            fBA_i = model.predict(BA_i)   # (n,)

            # Owen (2013) Eq. 5 & 6
            # S1_i  =      Cov(f(A),  f(BA_i)) / Var(Y)
            # ST_i  = 1 -  Cov(f(B),  f(AB_i)) / Var(Y)
            # Numerically: Cov(u, v) = mean(u*v) - mean(u)*mean(v) ~ mean(u*v) - f0^2
            cov_S1 = np.mean(fA * fBA_i) - f0 ** 2
            cov_ST = np.mean(fB * fAB_i) - f0 ** 2

            S1_boot[b, i] = cov_S1 / VarY
            ST_boot[b, i] = 1.0 - cov_ST / VarY

    # Aggregate over bootstrap resamples
    S1_mean = S1_boot.mean(axis=0)
    ST_mean = ST_boot.mean(axis=0)
    S1_ci   = np.percentile(S1_boot, [2.5, 97.5], axis=0).T   # (p, 2)
    ST_ci   = np.percentile(ST_boot, [2.5, 97.5], axis=0).T   # (p, 2)

    return {
        'S1':      S1_mean,
        'ST':      ST_mean,
        'S1_ci':   S1_ci,
        'ST_ci':   ST_ci,
        'S1_boot': S1_boot,
        'ST_boot': ST_boot,
    }


def plot_sobol_with_ci(feature_names, S1, ST, S1_ci, ST_ci, title, savepath):
    """
    Side-by-side bar chart of S1 and ST indices with 95% bootstrap CI error bars.
    """
    p     = len(feature_names)
    x     = np.arange(p)
    width = 0.35

    # Error bar lengths (must be non-negative)
    S1_err = np.array([S1 - S1_ci[:, 0], S1_ci[:, 1] - S1]).clip(min=0)
    ST_err = np.array([ST - ST_ci[:, 0], ST_ci[:, 1] - ST]).clip(min=0)

    fig, ax = plt.subplots(figsize=(max(6, p * 1.2), 4))

    ax.bar(x - width / 2, S1, width,
           label='First-Order (S1)', color='bisque',
           edgecolor='k', linewidth=0.7,
           yerr=S1_err, capsize=4, error_kw=dict(elinewidth=1.2))
    ax.bar(x + width / 2, ST, width,
           label='Total-Effect (ST)', color='lightsteelblue',
           edgecolor='k', linewidth=0.7,
           yerr=ST_err, capsize=4, error_kw=dict(elinewidth=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Sobol Index')
    ax.set_title(title)
    ax.legend()
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_ylim(bottom=min(0, (S1 - S1_err[0]).min() - 0.05))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Sobol CI plot saved to: {savepath}")


def main():
    # ==========================================
    # 0. Argument Parsing
    # ==========================================
    parser = argparse.ArgumentParser(description="Run SHAP and Sobol analysis for a specific dataset.")
    parser.add_argument("data_name", type=str, nargs='?', default="ITHH4500",
                        help="The name of the dataset (default: ITHH4500)")
    parser.add_argument("rand_seed", type=int, nargs='?', default=None,
                        help="The random seed (default: None=42)")
    parser.add_argument("--n_bootstrap", type=int, default=4000,
                        help="Number of bootstrap resamples for Owen Sobol estimator (default: 1000)")

    args = parser.parse_args()
    data_name   = args.data_name
    rand_seed   = args.rand_seed
    n_bootstrap = args.n_bootstrap

    print(f"Starting analysis for: {data_name} with seed={rand_seed}")

    # ==========================================
    # 1. Data Loading & Preprocessing
    # ==========================================
    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein')
    filepath = os.path.join(root_dir, "data", f"{data_name}.csv")
    if rand_seed is not None:
        savepath = os.path.join(root_dir, "material_nn_models", data_name + f"_seed_{rand_seed}")
    else:
        savepath = os.path.join(root_dir, "material_nn_models", data_name)
        rand_seed = 42

    if not os.path.exists(filepath):
        print(f"Error: Data file not found at {filepath}")
        return

    filedata = pd.read_csv(filepath)
    name_X   = filedata.columns[:-1].tolist()
    name_y   = filedata.columns[-1]
    df_in    = filedata[name_X]
    df_out   = filedata[[name_y]]
    print(f"TARGET: {name_y}")

    df_in_final, df_out_final = remove_outliers_iqr(df_in, df_out)
    removed_count = len(df_in) - len(df_in_final)
    print(f"# of data after removing outliers: {len(df_in_final)} ({removed_count} removed)")

    X = df_in_final[name_X].values
    y = df_out_final[name_y].values.reshape(-1, 1)

    X_temp_denorm, X_test_denorm, y_temp_denorm, y_test_denorm = train_test_split(
        X, y, test_size=0.2, random_state=rand_seed
    )

    feature_names  = name_X
    num_train_data = X_temp_denorm.shape[0]

    # ==========================================
    # 2. Load Models & Scalers
    # ==========================================
    model_path = os.path.join(savepath, f'{data_name}_best_mlp_model.pkl')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    loaded_model = joblib.load(model_path)
    scaler_X     = joblib.load(os.path.join(savepath, f'{data_name}_mlp_scaler_X.pkl'))
    scaler_y     = joblib.load(os.path.join(savepath, f'{data_name}_mlp_scaler_y.pkl'))

    X_temp_norm = scaler_X.transform(X_temp_denorm)
    X_test_norm = scaler_X.transform(X_test_denorm)

    # ==========================================
    # 2.5 Parity Plot (Actual vs Predicted)
    # ==========================================
    y_pred_norm        = loaded_model.predict(X_test_norm)
    y_test_denorm_flat = y_test_denorm.reshape(-1, 1).flatten()
    y_pred_denorm      = scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    r2                 = r2_score(y_test_denorm_flat, y_pred_denorm)

    plt.figure(figsize=(4, 4))
    plt.scatter(y_test_denorm_flat, y_pred_denorm,
                alpha=0.6, color='skyblue', edgecolors='k', s=30, label='Test Data')
    min_val = min(y_test_denorm_flat.min(), y_pred_denorm.min())
    max_val = max(y_test_denorm_flat.max(), y_pred_denorm.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    plt.xlabel(f'Actual {name_y}')
    plt.ylabel(f'Predicted {name_y}')
    plt.title(f'Parity Plot: {data_name} ($R^2 = {r2:.3f}$)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    parity_path = os.path.join(savepath, f"{data_name}_mlp_parity_plot.png")
    plt.savefig(parity_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Parity plot saved to: {parity_path} (R2: {r2:.4f})")

    # ==========================================
    # 2.6 Input vs Output Plot
    # ==========================================
    pred_y_norm_temp = loaded_model.predict(X_temp_norm)
    try:
        pred_y_temp = scaler_y.inverse_transform(pred_y_norm_temp.reshape(-1, 1)).flatten()
    except ValueError:
        pred_y_temp = pred_y_norm_temp.flatten()

    y_temp_denorm_flat = y_temp_denorm.flatten()
    n_features         = X_temp_denorm.shape[1]
    n_cols             = 2
    n_rows             = (n_features + n_cols - 1) // n_cols

    fig_io, axs_io = plt.subplots(n_rows, n_cols,
                                   figsize=(4 * n_cols, 3 * n_rows),
                                   constrained_layout=True)
    if n_features == 1:
        axs_io = np.array([axs_io])
    elif n_rows == 1:
        axs_io = np.array(axs_io)
    axs_io = axs_io.flatten()

    for i in range(n_features):
        ax = axs_io[i]
        ax.scatter(X_temp_denorm[:, i], y_temp_denorm_flat,
                   alpha=0.5, c='gray', s=15, label='Ground Truth')
        ax.scatter(X_temp_denorm[:, i], pred_y_temp,
                   alpha=0.5, c='red',  s=15, label='Prediction')
        ax.set_xlabel(feature_names[i] if i < len(feature_names) else f"Feature {i}")
        ax.set_ylabel(f"Output {name_y}")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    for i in range(n_features, len(axs_io)):
        axs_io[i].axis('off')

    plt.suptitle(f"Input vs Output Analysis: {data_name}", fontsize=14)
    input_vs_output_path = os.path.join(savepath, f"{data_name}_mlp_input_vs_output.png")
    plt.savefig(input_vs_output_path, dpi=300)
    plt.close()
    print(f"Input vs Output plot saved to: {input_vs_output_path}")

    # ==========================================
    # 3. SHAP Analysis
    # ==========================================
    n_bg_set        = 100
    num_unique_rows = np.unique(X_temp_norm, axis=0).shape[0]
    n_bg            = min(n_bg_set, num_unique_rows)
    print(f"N of Background Data Points: {n_bg}")

    X_bg   = shap.kmeans(X_temp_norm, n_bg) if n_bg < num_train_data else X_temp_norm
    X_eval = X_test_norm

    explainer   = shap.KernelExplainer(loaded_model.predict, X_bg)
    shap_values = explainer.shap_values(X_eval)

    shap_raw_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_raw_df.to_csv(os.path.join(savepath, f'{data_name}_shap_values_raw.csv'), index=False)

    mean_shap     = np.array(shap_values).mean(axis=0)
    abs_mean_shap = np.abs(mean_shap)
    mean_avg_shap = np.abs(shap_values.mean(axis=0))

    shap_summary_df = pd.DataFrame({
        'Feature':       feature_names,
        'Mean_SHAP':     mean_shap,
        'Abs_Mean_SHAP': abs_mean_shap,
        'Mean_Abs_SHAP': mean_avg_shap,
    })
    shap_summary_df.to_csv(
        os.path.join(savepath, f'{data_name}_shap_importance.csv'), index=False
    )
    print("[SHAP Analysis Result]")
    print(shap_summary_df.sort_values(by='Mean_Abs_SHAP', ascending=False))

    plot_custom_bars(
        names=shap_summary_df['Feature'],
        values=shap_summary_df['Mean_Abs_SHAP'],
        title=f"SHAP Global Importance (MLP) - {data_name}",
        ylabel="mean(|SHAP value|)",
        savepath=os.path.join(savepath, f"{data_name}_mlp_shap_bar_plot.png"),
        color='thistle'
    )

    shap.summary_plot(shap_values, X_eval, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(savepath, f'{data_name}_shap_dot_plot.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # 4. Owen (2013) Bootstrap Sobol Analysis
    # ==========================================
    # Uses the pick-freeze estimator directly on X_temp_norm (real training data).
    # No synthetic Saltelli sampling — sensitivity indices are estimated over
    # the empirical input distribution, which is the key advantage over SALib.
    #
    # SALib imports are no longer needed and have been removed.
    #
    # Runtime note:
    #   Each bootstrap iteration calls model.predict() 2*(p+1) times.
    #   Total predict calls = n_bootstrap * 2 * (p + 1)
    #   e.g., n_bootstrap=500, p=10 → ~11,000 calls (typically <1 min for MLP).
    #   Reduce n_bootstrap to 200 for a faster run during development.
    # ==========================================
    n_features = X_temp_norm.shape[1]

    print(f"\nRunning Owen (2013) Bootstrap Sobol analysis "
          f"({n_bootstrap} resamples, n={num_train_data} training points, "
          f"p={n_features} features)...")

    sobol_results = owen_sobol_bootstrap(
        model=loaded_model,
        X=X_temp_norm,
        n_bootstrap=n_bootstrap,
        seed=rand_seed
    )

    S1    = sobol_results['S1']
    ST    = sobol_results['ST']
    S1_ci = sobol_results['S1_ci']   # (p, 2): [lower, upper]
    ST_ci = sobol_results['ST_ci']

    # Console summary
    print("\n[Owen Sobol Analysis Result]")
    results_df = pd.DataFrame({
        'Feature':     feature_names,
        'S1':          S1,
        'S1_CI_lower': S1_ci[:, 0],
        'S1_CI_upper': S1_ci[:, 1],
        'ST':          ST,
        'ST_CI_lower': ST_ci[:, 0],
        'ST_CI_upper': ST_ci[:, 1],
    })
    print(results_df.sort_values(by='S1', ascending=False).to_string(index=False))

    # Save mean indices to CSV
    sobol_csv_path = os.path.join(savepath, f"{data_name}_sobol_indices_owen.csv")
    results_df.to_csv(sobol_csv_path, index=False)
    print(f"Sobol indices saved to: {sobol_csv_path}")

    # Save all bootstrap resamples (useful for diagnostic plots or meta-analysis)
    boot_df = pd.DataFrame(
        np.hstack([sobol_results['S1_boot'], sobol_results['ST_boot']]),
        columns=[f"S1_{f}" for f in feature_names] + [f"ST_{f}" for f in feature_names]
    )
    boot_csv_path = os.path.join(savepath, f"{data_name}_sobol_bootstrap_samples_owen.csv")
    boot_df.to_csv(boot_csv_path, index=False)
    print(f"Bootstrap samples saved to: {boot_csv_path}")

    # Plot 1: S1 only bar chart (mirrors original style via plot_custom_bars)
    plot_custom_bars(
        names=results_df['Feature'],
        values=results_df['S1'],
        title=f"Sobol Sensitivity — Owen Bootstrap (MLP) - {data_name}",
        ylabel="First Order Index (S1)",
        savepath=os.path.join(savepath, f"{data_name}_mlp_sobol_S1_plot.png"),
        color='bisque'
    )

    # Plot 2: S1 + ST side-by-side with 95% CI error bars
    plot_sobol_with_ci(
        feature_names=feature_names,
        S1=S1, ST=ST,
        S1_ci=S1_ci, ST_ci=ST_ci,
        title=f"Sobol Indices with 95% CI — Owen Bootstrap (MLP) - {data_name}",
        savepath=os.path.join(savepath, f"{data_name}_mlp_sobol_S1_ST_ci_plot.png")
    )


if __name__ == "__main__":
    main()