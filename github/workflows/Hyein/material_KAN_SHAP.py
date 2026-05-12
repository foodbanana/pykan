import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"
import joblib
import torch
import shap
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# Assuming this custom module exists in your environment
from kan.custom_processing import remove_outliers_iqr
from github.workflows.Hyein.toy_NN_SHAP_Sobol import plot_custom_bars

# KAN wrapper used to load the pretrained KAN checkpoint
# (same import path used in material_KAN_analyze.py)
from github.workflows.Hyein.toy_KAN_sweep import KANRegressor


# ==============================================================================
# YAML tuple constructor — required because KANRegressor.load_model() reads a
# config YAML produced with PyYAML's !!python/tuple tag. Mirrors the setup in
# material_KAN_analyze.py.
# ==============================================================================
def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))


yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.SafeLoader)
try:
    yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.Loader)
except AttributeError:
    pass


# ==============================================================================
# KAN -> sklearn-style predictor wrapper
# ------------------------------------------------------------------------------
# The SHAP KernelExplainer and the Owen Sobol estimator both assume the model
# has a `.predict(X_numpy) -> 1D numpy` interface (the sklearn MLP contract).
# A trained KAN, in contrast, is an nn.Module that takes a torch.Tensor and
# returns a torch.Tensor of shape (n, 1).
#
# This wrapper bridges the two so that the entire downstream pipeline
# (parity plot, SHAP, Sobol bootstrap) runs unchanged against a KAN model.
# ==============================================================================
class KANPredictor:
    """sklearn-style wrapper around a trained KAN model."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        # Put the model in eval mode if supported (no-op for plain modules).
        try:
            self.model.eval()
        except Exception:
            pass

    def predict(self, X):
        """X: (n, p) numpy array of NORMALISED inputs. Returns (n,) numpy array."""
        if isinstance(X, torch.Tensor):
            X_t = X.to(device=self.device, dtype=torch.float32)
        else:
            X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            y = self.model(X_t)
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        return np.asarray(y).reshape(-1)


# ==============================================================================
# Owen (2013) Pick-Freeze Bootstrap Sobol Estimator
# ------------------------------------------------------------------------------
# Reference:
#   Owen, A. B. (2013). Variance components and generalized Sobol' indices.
#   SIAM/ASA Journal on Uncertainty Quantification, 1(1), 19-41.
#   https://doi.org/10.1137/120876782
#
# First-order (S1) and total-effect (ST) — two bootstrap samples A, B:
#
#   S_i  = [ mean(f(A) * f(BA^(i))) - f0^2 ] / Var(Y)
#   ST_i = 1 - [ mean(f(B) * f(AB^(i))) - f0^2 ] / Var(Y)
#
#   where:
#     A, B    : two independent bootstrap draws from X
#     BA^(i)  : B with column i replaced by column i from A  ("pick-freeze")
#     AB^(i)  : A with column i replaced by column i from B
#     f0      : grand mean of f(X)
#
# Second-order (S2) — requires a third independent bootstrap sample C
# (Owen 2013, Eq. 17). The closed-form estimator is:
#
#   S2_ij = [ mean(f(A) * f(C_ij)) - mean(f(A) * f(CA^(i)))
#                                   - mean(f(A) * f(CA^(j))) + f0^2 ] / Var(Y)
#
#   where C_ij : C with columns i AND j both replaced from A (double pick-freeze)
#         CA^(i): C with only column i replaced from A  (single pick-freeze)
#         CA^(j): C with only column j replaced from A  (single pick-freeze)
#
#   This isolates the pure interaction variance between i and j by subtracting
#   the two main effects that would otherwise be double-counted.
#
#   Cost scales as O(p^2): p*(p-1)/2 pairs, each requiring one extra predict
#   call per bootstrap iteration (CA^(i) and CA^(j) are reused across pairs).
#
# Bootstrap resampling is applied K times to obtain mean estimates and
# 95% confidence intervals (2.5th / 97.5th percentile across resamples).
# ==============================================================================

def owen_sobol_bootstrap(model, X, n_bootstrap=500, seed=42):
    """
    Compute first-order (S1), total-effect (ST), and second-order (S2) Sobol
    indices from real data using the Owen (2013) pick-freeze estimator with
    bootstrap CIs.

    Parameters
    ----------
    model       : fitted model with a .predict(X) method
    X           : np.ndarray, shape (n_samples, n_features) — normalized inputs
    n_bootstrap : int, number of bootstrap resamples (default 500)
    seed        : int, random seed for reproducibility

    Returns
    -------
    results : dict with keys
        'S1'      — np.ndarray (p,)       mean first-order indices
        'ST'      — np.ndarray (p,)       mean total-effect indices
        'S2'      — np.ndarray (p, p)     mean second-order indices (symmetric,
                                          diagonal = 0, upper triangle filled)
        'S1_ci'   — np.ndarray (p, 2)     lower/upper 95% CI for S1
        'ST_ci'   — np.ndarray (p, 2)     lower/upper 95% CI for ST
        'S2_ci'   — np.ndarray (p, p, 2)  lower/upper 95% CI for S2
        'S1_boot' — np.ndarray (K, p)     all S1 bootstrap resamples
        'ST_boot' — np.ndarray (K, p)     all ST bootstrap resamples
        'S2_boot' — np.ndarray (K, p, p)  all S2 bootstrap resamples
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape

    # Pair-count warning
    n_pairs = p * (p - 1) // 2
    if p > 15:
        print(f"⚠️  Warning: {p} features → {n_pairs} pairs. "
              f"S2 computation will be slow. Consider reducing features.")
    else:
        print(f"ℹ️  Computing S2 for {n_pairs} feature pairs.")

    S1_boot = np.zeros((n_bootstrap, p))
    ST_boot = np.zeros((n_bootstrap, p))
    S2_boot = np.zeros((n_bootstrap, p, p))   # upper triangle used; rest = 0

    for b in range(n_bootstrap):
        # ----------------------------------------------------------------
        # Draw THREE independent bootstrap samples A, B, C
        #   A, B  — used for S1 and ST (unchanged from before)
        #   C     — third sample needed for S2 pick-freeze
        # ----------------------------------------------------------------
        idx_A = rng.integers(0, n, size=n)
        idx_B = rng.integers(0, n, size=n)
        idx_C = rng.integers(0, n, size=n)
        A = X[idx_A]
        B = X[idx_B]
        C = X[idx_C]

        # Base predictions
        fA = model.predict(A)
        fB = model.predict(B)

        f0   = 0.5 * (fA.mean() + fB.mean())
        VarY = np.concatenate([fA, fB]).var()

        if VarY < 1e-12:
            # Degenerate: constant output — all indices remain 0
            continue

        # ----------------------------------------------------------------
        # S1 / ST loop — identical to before, but also pre-compute
        # fCA_i for every i (C with column i from A), storing for S2 reuse
        # ----------------------------------------------------------------
        fCA = {}   # fCA[i] = model.predict(C with col i from A)

        for i in range(p):
            AB_i = A.copy();  AB_i[:, i] = B[:, i]
            BA_i = B.copy();  BA_i[:, i] = A[:, i]
            CA_i = C.copy();  CA_i[:, i] = A[:, i]

            fAB_i    = model.predict(AB_i)
            fBA_i    = model.predict(BA_i)
            fCA_i    = model.predict(CA_i)
            fCA[i]   = fCA_i

            cov_S1 = np.mean(fA * fBA_i) - f0 ** 2
            cov_ST = np.mean(fB * fAB_i) - f0 ** 2

            S1_boot[b, i] = cov_S1 / VarY
            ST_boot[b, i] = 1.0 - cov_ST / VarY

        # ----------------------------------------------------------------
        # S2 loop — Owen (2013) Eq. 17
        #
        #   S2_ij = [ mean(fA * f(C_ij))
        #             - mean(fA * fCA[i])
        #             - mean(fA * fCA[j])
        #             + f0^2 ] / VarY
        #
        # where C_ij = C with BOTH columns i and j replaced from A.
        # fCA[i] and fCA[j] are already computed above, so the only new
        # predict call per pair is f(C_ij).
        # ----------------------------------------------------------------
        for i in range(p):
            for j in range(i + 1, p):
                C_ij = C.copy()
                C_ij[:, i] = A[:, i]
                C_ij[:, j] = A[:, j]

                fC_ij = model.predict(C_ij)

                s2 = (np.mean(fA * fC_ij)
                      - np.mean(fA * fCA[i])
                      - np.mean(fA * fCA[j])
                      + f0 ** 2) / VarY

                S2_boot[b, i, j] = s2
                S2_boot[b, j, i] = s2   # keep matrix symmetric

    # ----------------------------------------------------------------
    # Aggregate over bootstrap resamples
    # ----------------------------------------------------------------
    S1_mean = S1_boot.mean(axis=0)
    ST_mean = ST_boot.mean(axis=0)
    S2_mean = S2_boot.mean(axis=0)

    S1_ci = np.percentile(S1_boot, [2.5, 97.5], axis=0).T          # (p, 2)
    ST_ci = np.percentile(ST_boot, [2.5, 97.5], axis=0).T          # (p, 2)
    S2_ci = np.percentile(S2_boot, [2.5, 97.5], axis=0)            # (2, p, p)
    S2_ci = np.moveaxis(S2_ci, 0, -1)                               # (p, p, 2)

    return {
        'S1':      S1_mean,
        'ST':      ST_mean,
        'S2':      S2_mean,
        'S1_ci':   S1_ci,
        'ST_ci':   ST_ci,
        'S2_ci':   S2_ci,
        'S1_boot': S1_boot,
        'ST_boot': ST_boot,
        'S2_boot': S2_boot,
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


def plot_sobol_s2_heatmap(feature_names, S2, S2_ci, title, savepath):
    """
    Heatmap of second-order Sobol indices S2_ij with 95% CI annotation.

    Upper triangle: S2 value (color-mapped).
    Lower triangle: mirrored for visual symmetry (same values).
    Diagonal: masked (self-interaction is undefined).
    Each cell is annotated with "mean\\n[lo, hi]" from the bootstrap CI.
    """
    import matplotlib.ticker as ticker
    from matplotlib.colors import TwoSlopeNorm

    p = len(feature_names)

    # Build display matrix: use full symmetric S2
    S2_display = S2.copy()
    np.fill_diagonal(S2_display, np.nan)   # mask diagonal

    # Color norm: centre at 0 so negative values (estimation noise) show clearly
    vmax = np.nanmax(np.abs(S2_display))
    vmax = max(vmax, 1e-6)   # guard against all-zero case
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(5, p * 0.9), max(4, p * 0.8)))
    im = ax.imshow(S2_display, cmap='RdBu_r', norm=norm, aspect='auto')

    # Annotate each off-diagonal cell with value and 95% CI
    for i in range(p):
        for j in range(p):
            if i == j:
                ax.text(j, i, '—', ha='center', va='center',
                        fontsize=8, color='gray')
                continue
            val  = S2[i, j]
            lo   = S2_ci[i, j, 0]
            hi   = S2_ci[i, j, 1]
            cell_text = f"{val:.3f}\n[{lo:.3f}, {hi:.3f}]"
            # Choose text colour for contrast against background
            text_color = 'white' if abs(val) > 0.4 * vmax else 'black'
            ax.text(j, i, cell_text, ha='center', va='center',
                    fontsize=7, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('S2 index', fontsize=9)

    ax.set_xticks(range(p))
    ax.set_yticks(range(p))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.set_title(title, fontsize=10, pad=10)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ S2 heatmap saved to: {savepath}")


def main():
    # ==========================================
    # 0. Argument Parsing
    # ==========================================
    parser = argparse.ArgumentParser(description="Run SHAP and Sobol analysis for a specific dataset (KAN model).")
    parser.add_argument("data_name", type=str, nargs='?', default="ITH4500",
                        help="The name of the dataset (default: ITH4500)")
    parser.add_argument("rand_seed", type=int, nargs='?', default=None,
                        help="The random seed (default: None=42)")
    parser.add_argument("--n_bootstrap", type=int, default=200,
                        help="Number of bootstrap resamples for Owen Sobol estimator (default: 200)")

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

    # ---- KAN models live under material_kan_models/ ----
    if rand_seed is not None:
        savepath = os.path.join(root_dir, "material_kan_models", data_name + f"_seed_{rand_seed}")
    else:
        savepath = os.path.join(root_dir, "material_kan_models", data_name)
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
    # 2. Load Pretrained KAN Model & Scalers
    # ------------------------------------------
    # Mirrors the loading logic in material_KAN_analyze.py:
    #   * KAN checkpoint:  <savepath>/<data_name>_best_kan_model   (no extension)
    #   * Scaler X / y :   <savepath>/<data_name>_mlp_scaler_X.pkl
    #                      <savepath>/<data_name>_mlp_scaler_y.pkl
    #
    # The KAN nn.Module is then wrapped in KANPredictor, which gives it a
    # sklearn-style .predict(numpy_array) interface so SHAP and the Owen
    # Sobol bootstrap downstream can be reused verbatim.
    # ==========================================
    ckpt_path     = os.path.join(savepath, f'{data_name}_best_kan_model')
    scaler_x_path = os.path.join(savepath, f'{data_name}_mlp_scaler_X.pkl')
    scaler_y_path = os.path.join(savepath, f'{data_name}_mlp_scaler_y.pkl')

    if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
        print(f"❌ Error: Scaler files not found in {savepath}")
        return

    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model_wrapper = KANRegressor(device=device)
    try:
        model_wrapper.load_model(ckpt_path)
        print("✅ KAN model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load KAN model from {ckpt_path}: {e}")
        return

    # sklearn-style wrapper for downstream SHAP & Sobol
    loaded_model = KANPredictor(model_wrapper.model, device=device)

    # Use scaler.transform (NOT fit_transform) — the scalers were fitted at
    # training time and we want consistent inputs to the pretrained model.
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
    parity_path = os.path.join(savepath, f"{data_name}_kan_parity_plot.png")
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
    input_vs_output_path = os.path.join(savepath, f"{data_name}_kan_input_vs_output.png")
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
        title=f"SHAP Global Importance (KAN) - {data_name}",
        ylabel="mean(|SHAP value|)",
        savepath=os.path.join(savepath, f"{data_name}_kan_shap_bar_plot.png"),
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
    # the empirical input distribution.
    #
    # Runtime note (with S2):
    #   Each bootstrap iteration calls model.predict() 2*(p+1) + p*(p-1)/2 times.
    #   e.g., n_bootstrap=500, p=10 → S1/ST: ~11,000 calls + S2: ~22,500 calls.
    #   KAN forward passes are heavier than the sklearn MLP, so consider
    #   reducing n_bootstrap to 200 for a faster run during development.
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
    S2    = sobol_results['S2']       # (p, p) symmetric matrix
    S1_ci = sobol_results['S1_ci']    # (p, 2)
    ST_ci = sobol_results['ST_ci']    # (p, 2)
    S2_ci = sobol_results['S2_ci']    # (p, p, 2)

    # Per-feature S2 sum: sum_j S2_ij for each feature i
    # Diagonal is 0, so summing the full row gives sum over all j != i
    S2_sum = S2.sum(axis=1)           # (p,)

    # ---- Console summary ----
    print("\n[Owen Sobol Analysis Result]")
    results_df = pd.DataFrame({
        'Feature':     feature_names,
        'S1':          S1,
        'S1_CI_lower': S1_ci[:, 0],
        'S1_CI_upper': S1_ci[:, 1],
        'ST':          ST,
        'ST_CI_lower': ST_ci[:, 0],
        'ST_CI_upper': ST_ci[:, 1],
        'S2_sum':      S2_sum,        # sum_j S2_ij for each feature i
    })
    print(results_df.sort_values(by='S1', ascending=False).to_string(index=False))

    # ---- Save S1/ST/S2_sum indices CSV ----
    sobol_csv_path = os.path.join(savepath, f"{data_name}_sobol_indices_owen.csv")
    results_df.to_csv(sobol_csv_path, index=False)
    print(f"Sobol indices saved to: {sobol_csv_path}")

    # ---- Save full S2 matrix to CSV ----
    s2_matrix_df = pd.DataFrame(S2, index=feature_names, columns=feature_names)
    s2_csv_path  = os.path.join(savepath, f"{data_name}_sobol_S2_matrix_owen.csv")
    s2_matrix_df.to_csv(s2_csv_path)
    print(f"S2 matrix saved to: {s2_csv_path}")

    # ---- Save all bootstrap resamples ----
    boot_df = pd.DataFrame(
        np.hstack([sobol_results['S1_boot'], sobol_results['ST_boot']]),
        columns=[f"S1_{f}" for f in feature_names] + [f"ST_{f}" for f in feature_names]
    )
    boot_csv_path = os.path.join(savepath, f"{data_name}_sobol_bootstrap_samples_owen.csv")
    boot_df.to_csv(boot_csv_path, index=False)
    print(f"Bootstrap samples saved to: {boot_csv_path}")

    # ---- Plot 1: S1 only bar chart ----
    plot_custom_bars(
        names=results_df['Feature'],
        values=results_df['S1'],
        title=f"Sobol Sensitivity — Owen Bootstrap (KAN) - {data_name}",
        ylabel="First Order Index (S1)",
        savepath=os.path.join(savepath, f"{data_name}_kan_sobol_S1_plot.png"),
        color='bisque'
    )

    # ---- Plot 2: S1 + ST side-by-side with 95% CI error bars ----
    plot_sobol_with_ci(
        feature_names=feature_names,
        S1=S1, ST=ST,
        S1_ci=S1_ci, ST_ci=ST_ci,
        title=f"Sobol Indices with 95% CI — Owen Bootstrap (KAN) - {data_name}",
        savepath=os.path.join(savepath, f"{data_name}_kan_sobol_S1_ST_ci_plot.png")
    )

    # ---- Plot 3: S2 heatmap ----
    plot_sobol_s2_heatmap(
        feature_names=feature_names,
        S2=S2,
        S2_ci=S2_ci,
        title=f"Second-Order Sobol Indices S2 — Owen Bootstrap (KAN) - {data_name}",
        savepath=os.path.join(savepath, f"{data_name}_kan_sobol_S2_heatmap.png")
    )


if __name__ == "__main__":
    main()