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

# ----------------------------------------------------------------------------
# SALib imports — module layout changed between versions.
# Try the modern path first; fall back to the legacy one.
# ----------------------------------------------------------------------------
try:
    from SALib.sample.sobol import sample as saltelli_sample
except ImportError:
    from SALib.sample.saltelli import sample as saltelli_sample
from SALib.analyze.sobol import analyze as sobol_analyze
import math

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
# The SHAP KernelExplainer and SALib's Sobol analyser both consume a callable
# that takes a (n, p) numpy array and returns a 1D numpy array of length n.
# This wrapper bridges the KAN nn.Module to that contract, so SHAP and SALib
# can each be reused as-is.
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
# SALib Sobol' Analysis (Saltelli sampling)
# ------------------------------------------------------------------------------
# Reference:
#   Saltelli, A. et al. (2010). Variance based sensitivity analysis of model
#   output. Design and estimator for the total sensitivity index.
#   Computer Physics Communications, 181(2), 259-270.
#
#   Sobol', I. M. (2001). Global sensitivity indices for nonlinear mathematical
#   models and their Monte Carlo estimates. Mathematics and Computers in
#   Simulation, 55(1-3), 271-280.
#
# Sampling design:
#   * Inputs are assumed to be normalised to [0, 1] (MinMaxScaler convention)
#   * Saltelli low-discrepancy sequence is used to construct paired matrices
#   * Total evaluations:
#       N * (2D + 2)  if calc_second_order=True   (default)
#       N * (D + 2)   if calc_second_order=False
#     where N = base sample size, D = number of features.
#
# Returned indices (SALib `analyze` output):
#   S1, S1_conf  — first-order indices and 95% confidence half-widths
#   ST, ST_conf  — total-effect indices and 95% confidence half-widths
#   S2, S2_conf  — (D, D) upper-triangular matrices, NaN elsewhere
# ==============================================================================

def salib_sobol_analysis(model, n_features, feature_names,
                         n_samples=1024, calc_second_order=True, seed=42,
                         bounds_lo=0.0, bounds_hi=1.0):
    """
    Run SALib Sobol' analysis on a model whose inputs live in [bounds_lo, bounds_hi]^p.

    Parameters
    ----------
    model            : object with .predict(X) -> 1D numpy
    n_features       : int, number of input features
    feature_names    : list[str], length n_features
    n_samples        : int, Saltelli base sample size N
                       (total predict calls = N * (2D + 2) with S2, N * (D + 2) without)
    calc_second_order: bool, whether to compute S2 indices
    seed             : int, RNG seed for sampling and CI bootstrap
    bounds_lo/hi     : float, per-feature bounds (default [0, 1] for MinMax-scaled data)

    Returns
    -------
    dict with keys
        'S1', 'ST'           — np.ndarray (p,)
        'S1_conf', 'ST_conf' — np.ndarray (p,)  symmetric 95% half-widths
        'S1_ci', 'ST_ci'     — np.ndarray (p, 2) lower/upper bounds
        'S2'                 — np.ndarray (p, p) symmetric; diagonal = 0
        'S2_conf'            — np.ndarray (p, p) symmetric; diagonal = 0
        'S2_ci'              — np.ndarray (p, p, 2)
        'param_values'       — np.ndarray (N*(2D+2), p) raw Saltelli design
        'Y'                  — np.ndarray (N*(2D+2),)   model outputs at design
        'problem'            — dict, the SALib problem definition
    """
    problem = {
        'num_vars': n_features,
        'names':    list(feature_names),
        'bounds':   [[bounds_lo, bounds_hi]] * n_features,
    }

    # ---- Saltelli sampling ---------------------------------------------------
    param_values = saltelli_sample(
        problem, n_samples,
        calc_second_order=calc_second_order,
        seed=seed,
    )
    print(f"ℹ️  Saltelli design: {param_values.shape[0]} evaluations "
          f"(N={n_samples}, D={n_features}, S2={'on' if calc_second_order else 'off'})")

    # ---- Model evaluation ----------------------------------------------------
    Y = model.predict(param_values)

    # ---- Sobol' analysis -----------------------------------------------------
    Si = sobol_analyze(
        problem, Y,
        calc_second_order=calc_second_order,
        seed=seed,
        print_to_console=False,
    )

    S1      = np.asarray(Si['S1'])
    ST      = np.asarray(Si['ST'])
    S1_conf = np.asarray(Si['S1_conf'])
    ST_conf = np.asarray(Si['ST_conf'])

    S1_ci = np.stack([S1 - S1_conf, S1 + S1_conf], axis=1)
    ST_ci = np.stack([ST - ST_conf, ST + ST_conf], axis=1)

    # ---- S2 matrix (symmetrise; diagonal -> 0) -------------------------------
    if calc_second_order:
        S2_raw       = np.asarray(Si['S2'],      dtype=float)
        S2_conf_raw  = np.asarray(Si['S2_conf'], dtype=float)
        S2           = np.zeros_like(S2_raw)
        S2_conf      = np.zeros_like(S2_conf_raw)
        for i in range(n_features):
            for j in range(i + 1, n_features):
                v   = S2_raw[i, j]
                vc  = S2_conf_raw[i, j]
                if not np.isnan(v):
                    S2[i, j]      = v;  S2[j, i]      = v
                    S2_conf[i, j] = vc; S2_conf[j, i] = vc
        S2_ci = np.stack([S2 - S2_conf, S2 + S2_conf], axis=-1)
    else:
        S2      = np.zeros((n_features, n_features))
        S2_conf = np.zeros_like(S2)
        S2_ci   = np.zeros((n_features, n_features, 2))

    return {
        'S1':      S1,
        'ST':      ST,
        'S1_conf': S1_conf,
        'ST_conf': ST_conf,
        'S1_ci':   S1_ci,
        'ST_ci':   ST_ci,
        'S2':      S2,
        'S2_conf': S2_conf,
        'S2_ci':   S2_ci,
        'param_values': param_values,
        'Y':            Y,
        'problem':      problem,
    }


def plot_sobol_with_ci(feature_names, S1, ST, S1_ci, ST_ci, title, savepath):
    """
    Side-by-side bar chart of S1 and ST indices with 95% CI error bars.
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
    """
    from matplotlib.colors import TwoSlopeNorm

    p = len(feature_names)

    S2_display = S2.copy().astype(float)
    np.fill_diagonal(S2_display, np.nan)   # mask diagonal

    vmax = np.nanmax(np.abs(S2_display))
    vmax = max(vmax, 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(5, p * 0.9), max(4, p * 0.8)))
    im = ax.imshow(S2_display, cmap='RdBu_r', norm=norm, aspect='auto')

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


# ==============================================================================
# Saltelli Design Coverage Check (post-hoc)
# ------------------------------------------------------------------------------
# Saltelli sampling fills [0, 1]^p uniformly with a low-discrepancy sequence.
# When the training data lives on a strict subset of the unit cube (typical
# for tabular engineering data), a significant share of the design points
# land in regions the KAN has never seen during training. In those regions,
# the model is extrapolating, and the resulting Sobol' indices partially
# reflect that extrapolation rather than data-supported sensitivity.
#
# This routine quantifies "how far off the data manifold the Saltelli design
# strays" with four complementary measures:
#
#   1. Per-feature marginal coverage — fraction of design points whose
#      value for each feature is inside the empirical [min, max] of the
#      training data.
#   2. Bounding-box coverage — fraction inside ALL per-feature ranges
#      simultaneously (the loosest convex envelope).
#   3. Convex-hull coverage (p <= 8 only) — fraction inside the convex
#      hull of the training points. Tighter than the bounding box; skipped
#      in higher dimensions because Delaunay triangulation becomes
#      computationally infeasible.
#   4. Nearest-neighbour distance ratio — median NN distance from design
#      points to training data, divided by the median NN distance within
#      the training data itself. Values near 1.0 mean the design is as
#      tight as the data; large values indicate it has drifted off the
#      manifold.
# ==============================================================================

def check_design_coverage(X_train, X_design, feature_names, savepath, data_name):
    """Diagnose how much of the Saltelli design lies inside the training-data support."""
    from sklearn.neighbors import NearestNeighbors

    n_train, p = X_train.shape
    n_design   = X_design.shape[0]

    # ---- 1. Per-feature marginal coverage --------------------------------
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    in_each      = (X_design >= x_min) & (X_design <= x_max)   # (n_design, p)
    per_feat_cov = in_each.mean(axis=0)
    bbox_cov     = float(in_each.all(axis=1).mean())

    # ---- 2. Convex-hull coverage (low-D only) ----------------------------
    hull_cov = None
    if p <= 8:
        try:
            from scipy.spatial import Delaunay
            hull = Delaunay(X_train)
            hull_cov = float((hull.find_simplex(X_design) >= 0).mean())
        except Exception as e:
            print(f"   Convex-hull check skipped: {e}")
    else:
        print(f"   Convex-hull check skipped: p={p} > 8 "
              f"(Delaunay triangulation infeasible in higher dim).")

    # ---- 3. NN distance distribution -------------------------------------
    nn1          = NearestNeighbors(n_neighbors=1).fit(X_train)
    d_design, _  = nn1.kneighbors(X_design)
    d_design     = d_design.ravel()

    nn2          = NearestNeighbors(n_neighbors=2).fit(X_train)
    d_train, _   = nn2.kneighbors(X_train)
    d_train      = d_train[:, 1]    # nearest neighbour excluding self

    nn_ratio = float(np.median(d_design) / (np.median(d_train) + 1e-12))

    # ---- Console summary -------------------------------------------------
    print("\n[Saltelli Design Coverage Check]")
    print(f"   n_train  = {n_train}")
    print(f"   n_design = {n_design}")
    print(f"   Bounding-box coverage:                          {bbox_cov:.3%}")
    if hull_cov is not None:
        print(f"   Convex-hull coverage:                           {hull_cov:.3%}")
    print(f"   Per-feature coverage  (min / median / max):     "
          f"{per_feat_cov.min():.3f} / {np.median(per_feat_cov):.3f} / {per_feat_cov.max():.3f}")
    print(f"   NN-distance ratio (median design / median train): {nn_ratio:.2f}")

    # ---- Save CSVs -------------------------------------------------------
    per_feat_df = pd.DataFrame({
        'Feature':                feature_names,
        'Training_min':           x_min,
        'Training_max':           x_max,
        'Saltelli_in_range_frac': per_feat_cov,
    })
    csv_path = os.path.join(savepath, f"{data_name}_design_coverage_per_feature.csv")
    per_feat_df.to_csv(csv_path, index=False)
    print(f"   Per-feature coverage saved to: {csv_path}")

    summary = {
        'n_train':                n_train,
        'n_design':               n_design,
        'bounding_box_coverage':  bbox_cov,
        'convex_hull_coverage':   hull_cov if hull_cov is not None else np.nan,
        'nn_distance_ratio':      nn_ratio,
        'median_train_nn_dist':   float(np.median(d_train)),
        'median_design_nn_dist':  float(np.median(d_design)),
    }
    summary_df  = pd.DataFrame([summary])
    summary_csv = os.path.join(savepath, f"{data_name}_design_coverage_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"   Summary saved to: {summary_csv}")

    # ---- Plot ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # (a) Per-feature coverage bar chart
    xpos = np.arange(p)
    ax1.bar(xpos, per_feat_cov, color='lightsteelblue', edgecolor='k', linewidth=0.7)
    ax1.axhline(bbox_cov, color='red', ls='--', lw=1,
                label=f'Bounding-box: {bbox_cov:.1%}')
    if hull_cov is not None:
        ax1.axhline(hull_cov, color='darkgreen', ls=':', lw=1.2,
                    label=f'Convex hull: {hull_cov:.1%}')
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(feature_names, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel("Fraction within training [min, max]")
    ax1.set_ylim([0, 1.05])
    ax1.legend(fontsize=8, loc='lower right')
    ax1.set_title("Per-feature coverage")
    ax1.grid(axis='y', alpha=0.3)

    # (b) Nearest-neighbour distance histogram
    upper = max(np.percentile(d_design, 99), np.percentile(d_train, 99))
    bins  = np.linspace(0, upper, 40)
    ax2.hist(d_train,  bins=bins, alpha=0.6, color='gray',
             label=f'Training NN (n={n_train})',    density=True)
    ax2.hist(d_design, bins=bins, alpha=0.6, color='bisque',
             label=f'Saltelli design NN (n={n_design})', density=True)
    ax2.axvline(np.median(d_train),  color='gray',   ls='--', lw=1)
    ax2.axvline(np.median(d_design), color='orange', ls='--', lw=1)
    ax2.set_xlabel("Nearest-neighbour distance to training set")
    ax2.set_ylabel("Density")
    ax2.set_title(f"NN-distance distribution (ratio = {nn_ratio:.2f})")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.suptitle(f"Saltelli Design Coverage Check — {data_name}", fontsize=11)
    plt.tight_layout()
    fig_path = os.path.join(savepath, f"{data_name}_design_coverage.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Coverage plot saved to: {fig_path}")

    return summary


def next_power_of_two(N):
    """
    Return (n, 2**n) such that 2**n is the smallest power of two strictly
    greater than N.

    Parameters
    ----------
    N : int or float

    Returns
    -------
    (n, value) : tuple[int, int]
        n     — exponent
        value — 2 ** n
    """
    if N < 1:
        # For N <= 0, the smallest 2^n > N is 2^0 = 1.
        return 0, 1

    # log2 is robust for both int and float inputs; flooring then adding 1
    # gives the smallest integer n with 2^n > N (strictly).
    log2_N = math.log2(N)
    n = int(math.floor(log2_N)) + 1
    return n, 1 << n


def main():
    # ==========================================
    # 0. Argument Parsing
    # ==========================================
    parser = argparse.ArgumentParser(description="Run SHAP and SALib Sobol analysis for a specific dataset (KAN model).")
    parser.add_argument("data_name", type=str, nargs='?', default="ITH4500",
                        help="The name of the dataset (default: ITH4500)")
    parser.add_argument("rand_seed", type=int, nargs='?', default=None,
                        help="The random seed (default: None=42)")
    parser.add_argument("--with_s2", action='store_true',
                        help="Also compute second-order S2 indices (slower; off by default).")

    args = parser.parse_args()
    data_name         = args.data_name
    rand_seed         = args.rand_seed
    calc_second_order = args.with_s2

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

    # sklearn-style wrapper for downstream SHAP & SALib
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
    parity_path = os.path.join(savepath, f"{data_name}_kan_SALib_parity_plot.png")
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
    input_vs_output_path = os.path.join(savepath, f"{data_name}_kan_SALib_input_vs_output.png")
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
    # 4. SALib (Saltelli) Sobol' Analysis
    # ------------------------------------------
    # Replaces the Owen pick-freeze bootstrap. Inputs are uniformly sampled
    # from [0, 1]^p using Saltelli's low-discrepancy design, then evaluated on
    # the KAN. Note: this samples beyond the empirical data manifold; if the
    # KAN extrapolates poorly outside the training distribution, indices may
    # reflect that extrapolation rather than data-supported sensitivity.
    # ==========================================
    n_features = X_temp_norm.shape[1]
    _, n_LGN_power_two = next_power_of_two(X_test_norm.shape[0])
    n_samples = min(n_LGN_power_two, 2048)

    print(f"\nRunning SALib Sobol' analysis "
          f"(Saltelli sampling: N={n_samples}, "
          f"S2={'on' if calc_second_order else 'off'}, "
          f"p={n_features} features)...")

    sobol_results = salib_sobol_analysis(
        model=loaded_model,
        n_features=n_features,
        feature_names=feature_names,
        n_samples=n_samples,
        calc_second_order=calc_second_order,
        seed=rand_seed,
        bounds_lo=0.0,
        bounds_hi=1.0,
    )

    S1    = sobol_results['S1']
    ST    = sobol_results['ST']
    S2    = sobol_results['S2']       # (p, p) symmetric
    S1_ci = sobol_results['S1_ci']    # (p, 2)
    ST_ci = sobol_results['ST_ci']    # (p, 2)
    S2_ci = sobol_results['S2_ci']    # (p, p, 2)

    # Per-feature S2 sum: sum_j S2_ij for each feature i
    S2_sum = S2.sum(axis=1)           # (p,)

    # ---- Console summary ----
    print("\n[SALib Sobol' Analysis Result]")
    results_df = pd.DataFrame({
        'Feature':     feature_names,
        'S1':          S1,
        'S1_conf':     sobol_results['S1_conf'],
        'S1_CI_lower': S1_ci[:, 0],
        'S1_CI_upper': S1_ci[:, 1],
        'ST':          ST,
        'ST_conf':     sobol_results['ST_conf'],
        'ST_CI_lower': ST_ci[:, 0],
        'ST_CI_upper': ST_ci[:, 1],
        'S2_sum':      S2_sum,
    })
    print(results_df.sort_values(by='S1', ascending=False).to_string(index=False))

    # ---- Save S1/ST/S2_sum indices CSV ----
    sobol_csv_path = os.path.join(savepath, f"{data_name}_sobol_indices_salib.csv")
    results_df.to_csv(sobol_csv_path, index=False)
    print(f"Sobol indices saved to: {sobol_csv_path}")

    # ---- Save full S2 matrix to CSV ----
    s2_matrix_df = pd.DataFrame(S2, index=feature_names, columns=feature_names)
    s2_csv_path  = os.path.join(savepath, f"{data_name}_sobol_S2_matrix_salib.csv")
    s2_matrix_df.to_csv(s2_csv_path)
    print(f"S2 matrix saved to: {s2_csv_path}")

    # ---- Save raw Saltelli design + outputs (handy for debugging / reuse) ----
    raw_df = pd.DataFrame(sobol_results['param_values'], columns=feature_names)
    raw_df['Y'] = sobol_results['Y']
    raw_csv_path = os.path.join(savepath, f"{data_name}_saltelli_design.csv")
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"Saltelli design + outputs saved to: {raw_csv_path}")

    # ==========================================
    # 4.5 Post-hoc: Saltelli Design Coverage Check
    # ------------------------------------------
    # How far off the training-data manifold does the Saltelli design stray?
    # If coverage is low, the Sobol' indices above partly reflect the KAN's
    # extrapolation behaviour rather than its behaviour on plausible inputs;
    # report this diagnostic alongside the indices.
    # ==========================================
    # check_design_coverage(
    #     X_train      = X_temp_norm,
    #     X_design     = sobol_results['param_values'],
    #     feature_names= feature_names,
    #     savepath     = savepath,
    #     data_name    = data_name,
    # )

    # ---- Plot 1: S1 only bar chart ----
    plot_custom_bars(
        names=results_df['Feature'],
        values=results_df['S1'],
        title=f"Sobol Sensitivity — SALib Saltelli (KAN) - {data_name}",
        ylabel="First Order Index (S1)",
        savepath=os.path.join(savepath, f"{data_name}_kan_sobol_S1_plot.png"),
        color='bisque'
    )

    # ---- Plot 2: S1 + ST side-by-side with 95% CI error bars ----
    plot_sobol_with_ci(
        feature_names=feature_names,
        S1=S1, ST=ST,
        S1_ci=S1_ci, ST_ci=ST_ci,
        title=f"Sobol Indices with 95% CI — SALib Saltelli (KAN) - {data_name}",
        savepath=os.path.join(savepath, f"{data_name}_kan_sobol_S1_ST_ci_plot.png")
    )

    # ---- Plot 3: S2 heatmap ----
    if calc_second_order:
        plot_sobol_s2_heatmap(
            feature_names=feature_names,
            S2=S2,
            S2_ci=S2_ci,
            title=f"Second-Order Sobol Indices S2 — SALib Saltelli (KAN) - {data_name}",
            savepath=os.path.join(savepath, f"{data_name}_kan_sobol_S2_heatmap.png")
        )


if __name__ == "__main__":
    main()