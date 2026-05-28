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
from github.workflows.Hyein.toy_KAN_sweep import FUNCTION_ZOO


# ==========================================
# 0. Helper Functions
# ==========================================
def plot_custom_bars(names, values, title, ylabel, savepath, color="skyblue", show=False):
    """
    Helper function to draw vertical bar plots.
    """
    fig, ax = plt.subplots(figsize=(max(4, len(names) * 1.2), 5))

    # Create Vertical Bars
    bars = ax.bar(names, values, color=color, edgecolor='black', width=0.7)

    # Add number labels on top of bars
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10)

    # Formatting
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='center', fontsize=10)

    # Adjust Y-limit
    if len(values) > 0:
        ax.set_ylim(0, max(values) * 1.15)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def run_analysis_suite(model_func, bounds, feature_names, save_dir, suffix, title_suffix=""):
    """
    Runs both Sobol and SHAP analysis for a specific set of bounds.
    """
    n_features = len(bounds)
    print(f"\n   ⚙️ Running Analysis Suite {suffix}...")

    # ------------------------------------------------
    # 1. Sobol Analysis
    # ------------------------------------------------
    problem = {
        'num_vars': n_features,
        'names': feature_names,
        'bounds': bounds
    }

    # Generate samples (shared with SHAP below)
    X_sobol = saltelli.sample(problem, 512, calc_second_order=True, seed=42)
    Y_sobol = model_func(X_sobol)

    try:
        Si = sobol.analyze(problem, Y_sobol, calc_second_order=True)

        # Save CSV
        results_df = pd.DataFrame({
            'Feature': feature_names,
            'Total_Effect (ST)': Si['ST'],
            'First_Order (S1)': Si['S1']
        })  # Preserve order for consistency

        results_df.to_csv(os.path.join(save_dir, f"sobol_indices{suffix}.csv"), index=False)

        # Plot
        plot_custom_bars(
            names=results_df['Feature'],
            values=results_df['First_Order (S1)'],
            title=f"Sobol Sensitivity {title_suffix}",
            ylabel="First Order Index (S1)",
            savepath=os.path.join(save_dir, f"sobol_plot{suffix}.png"),
            color='bisque'
        )
    except Exception as e:
        print(f"      ⚠️ Sobol Analysis skipped due to error (likely range too small/constant): {e}")

    # ------------------------------------------------
    # 2. SHAP Analysis
    # ------------------------------------------------
    # Use the same Sobol samples for consistency
    X_bg = shap.kmeans(X_sobol, 100)
    X_test = X_sobol

    explainer = shap.KernelExplainer(model_func, X_bg)
    # Silence shap warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        shap_values = explainer.shap_values(X_test, silent=True)

    # Save Mean Abs SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    })
    shap_importance_df.to_csv(os.path.join(save_dir, f"shap_mean_abs{suffix}.csv"), index=False)

    # Plot Bar
    plot_custom_bars(
        names=shap_importance_df['Feature'],
        values=shap_importance_df['Mean_Abs_SHAP'],
        title=f"SHAP Importance {title_suffix}",
        ylabel="mean(|SHAP value|)",
        savepath=os.path.join(save_dir, f"shap_bar_plot{suffix}.png"),
        color='thistle'
    )
    print(f"      ✅ Completed {suffix}")


def main():
    parser = argparse.ArgumentParser(description="Analyze analytical functions.")
    parser.add_argument("func_name", type=str, nargs='?', default="exponential",
                        choices=FUNCTION_ZOO.keys(),
                        help="Choose a function: " + ", ".join(FUNCTION_ZOO.keys()))

    args = parser.parse_args()
    case_name = args.func_name

    print(f"🚀 Running Analysis for case: '{case_name}'")

    config = FUNCTION_ZOO[case_name]
    feature_names = config["names"]
    n_features = len(config["bounds"])

    # Setup Output Path
    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein', "analytical_results", case_name)
    os.makedirs(root_dir, exist_ok=True)

    # Load scalers saved by toy_KAN_sweep.py
    kan_dir = os.path.join(root_dir, "kan_models")
    scaler_X = joblib.load(os.path.join(kan_dir, f'{case_name}_mlp_scaler_X.pkl'))
    scaler_y = joblib.load(os.path.join(kan_dir, f'{case_name}_mlp_scaler_y.pkl'))

    # model_func operates in scaled space: takes X in [0.1, 0.9], returns scaled y
    raw_func = lambda X: np.apply_along_axis(config["func"], 1, X)
    model_func = lambda X_scaled: scaler_y.transform(
        raw_func(scaler_X.inverse_transform(X_scaled)).reshape(-1, 1)
    ).flatten()

    # All features share the same scaled bounds
    base_bounds = [[0.1, 0.9]] * n_features

    # ==========================================
    # 2. Run Global Analysis
    # ==========================================
    print("\n🌍 [1/2] Running GLOBAL Analysis...")
    run_analysis_suite(
        model_func=model_func,
        bounds=base_bounds,
        feature_names=feature_names,
        save_dir=root_dir,
        suffix="_global",
        title_suffix="(Global)"
    )

    # ==========================================
    # 3. Run Range-Based Analysis (if configured)
    # ==========================================
    mask_idx = config.get("mask_idx")
    mask_divs = config.get("mask_division")

    if mask_idx is not None and mask_divs:
        print(f"\n✂️ [2/2] Running RANGE Analysis (Split by Feature {mask_idx}: {feature_names[mask_idx]})...")

        # Scale raw division points into [0.1, 0.9] space using scaler_X
        def scale_point(val):
            dummy = np.zeros((1, n_features))
            dummy[0, mask_idx] = val
            return scaler_X.transform(dummy)[0, mask_idx]

        feat_min, feat_max = 0.1, 0.9
        valid_divs = sorted([scale_point(d) for d in mask_divs
                             if feat_min < scale_point(d) < feat_max])
        split_points = [feat_min] + valid_divs + [feat_max]

        print(f"   Splitting points (scaled): {[f'{p:.3f}' for p in split_points]}")

        for i in range(len(split_points) - 1):
            lb, ub = split_points[i], split_points[i + 1]
            range_label = f"range_{i}_{lb:.2f}_to_{ub:.2f}"

            current_bounds = [list(b) for b in base_bounds]
            current_bounds[mask_idx] = [lb, ub]

            print(f"   🔹 Processing Range {i}: {feature_names[mask_idx]} in [{lb:.2f}, {ub:.2f}]")

            run_analysis_suite(
                model_func=model_func,
                bounds=current_bounds,
                feature_names=feature_names,
                save_dir=root_dir,
                suffix=f"_{range_label}",
                title_suffix=f"\n({feature_names[mask_idx]}: {lb:.2f} ~ {ub:.2f})"
            )

    else:
        print("\nℹ️  No 'mask_idx' or 'mask_division' defined. Skipping range analysis.")

    print(f"\n✅ All analysis complete. Results saved in: {root_dir}")


if __name__ == "__main__":
    main()