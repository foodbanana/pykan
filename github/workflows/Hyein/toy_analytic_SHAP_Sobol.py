import os
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from SALib.sample import sobol as saltelli
from SALib.analyze import sobol
from github.workflows.Hyein.toy_7_log_sum_factory import LOG_SUM_ZOO
from github.workflows.Hyein.toy_8_convex_factory import CONVEX_ZOO


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

    # Generate samples (Physical Domain)
    # Note: Sobol requires 2^n samples. 1024 is standard base.
    try:
        X_sobol = saltelli.sample(problem, 1024, calc_second_order=True)
        Y_sobol = model_func(X_sobol)
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
    # 1. Background (Random uniform within CURRENT bounds)
    X_representative = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(1024, n_features)
    )
    # Summarize to 100 weighted points
    X_bg = shap.kmeans(X_representative, 100)

    # 2. Test Data (Random uniform within CURRENT bounds)
    X_test = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(256, n_features)
    )

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


# ==========================================
# 1. Function Zoo & Main
# ==========================================
STANDARD_ZOO = {
    "original": {
        "func": lambda x: np.sin(2 * x[0]) + 5 * x[1],
        "bounds": [[-np.pi, np.pi], [-1, 1]],
        "names": ["Angle (x0)", "Linear (x1)"],
        "mask_idx": None,
        "mask_division": []
    },
    "mult_periodic": {
        "func": lambda x: x[1] * np.sin(2 * x[0]),
        "bounds": [[-np.pi, np.pi], [-1, 1]],
        "names": ["Angle (x0)", "Multiplier (x1)"],
        "mask_idx": None,
        "mask_division": []
    },
    "exponential": {
        "func": lambda x: np.exp(-2 * x[0]) + x[1],
        "bounds": [[-1, 1], [-1, 1]],
        "names": ["Exponent (x0)", "Linear (x1)"],
        "mask_idx": 0,
        "mask_division": [0.4]
    },
    "logarithm": {
        "func": lambda x: np.log(20 * (x[0] + 1.2)) + x[1],
        "bounds": [[-1, 1], [-1, 1]],
        "names": ["Log (x0)", "Linear (x1)"],
        "mask_idx": 0,
        "mask_division": [-0.8]
    },
    "convolution": {
        "func": lambda x: x[0] ** 2 / (x[1] + 1.08) / 1.8,
        "bounds": [[-1, 1], [-1, 1]],
        "names": ["Convex (x0)", "Denominator (x1)"],
        "mask_idx": None,
        "mask_division": []
    },
    "multiplication": {
        "func": lambda x: x[0] **4 * x[1],
        "bounds": [[-1, 1], [-1, 1]],
        "names": ["Square (x0)", "Linear (x1)"],
        "mask_idx": None,
        "mask_division": []
    },
    "conditional": {
        "func": lambda x: x[0]*2 + x[1] if x[0] < 0 else x[1],
        "bounds": [[-1, 1], [-1, 1]],
        "names": ["Conditional (x0)", "Linear (x1)"],
        "mask_idx": None,
        "mask_division": []
    },
}
FUNCTION_ZOO = {**STANDARD_ZOO, **LOG_SUM_ZOO, **CONVEX_ZOO}


def main():
    parser = argparse.ArgumentParser(description="Analyze analytical functions.")
    parser.add_argument("func_name", type=str, nargs='?', default="exponential",
                        choices=FUNCTION_ZOO.keys(),
                        help="Choose a function: " + ", ".join(FUNCTION_ZOO.keys()))

    args = parser.parse_args()
    case_name = args.func_name

    print(f"🚀 Running Analysis for case: '{case_name}'")

    # Load config
    config = FUNCTION_ZOO[case_name]
    model_func = lambda X: np.apply_along_axis(config["func"], 1, X)

    base_bounds = config["bounds"]
    feature_names = config["names"]

    # Setup Output Path
    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein', "analytical_results", case_name)
    os.makedirs(root_dir, exist_ok=True)

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

        # Get the global bounds for the split feature
        feat_min, feat_max = base_bounds[mask_idx]

        # Create full list of split points: [min, div1, div2, ..., max]
        # Sort and filter divisions to ensure they are within bounds
        valid_divs = sorted([d for d in mask_divs if feat_min < d < feat_max])
        split_points = [feat_min] + valid_divs + [feat_max]

        print(f"   Splitting points: {split_points}")

        for i in range(len(split_points) - 1):
            lb, ub = split_points[i], split_points[i + 1]
            range_label = f"range_{i}_{lb:.2f}_to_{ub:.2f}"

            # Create New Bounds for this range
            # Copy base bounds, then update the specific feature's bounds
            current_bounds = [list(b) for b in base_bounds]  # Deep copy
            current_bounds[mask_idx] = [lb, ub]

            print(f"   🔹 Processing Range {i}: {feature_names[mask_idx]} in [{lb:.2f}, {ub:.2f}]")

            # Create sub-directory for neatness (optional, currently saving in root)
            # save_subdir = os.path.join(root_dir, range_label)
            # os.makedirs(save_subdir, exist_ok=True)

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