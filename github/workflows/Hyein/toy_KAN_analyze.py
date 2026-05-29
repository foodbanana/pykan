import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import yaml  # <--- [NEW] Import YAML

# ==========================================
# [FIX] Register Python Tuple for YAML Loading
# ==========================================
# This fixes the "could not determine a constructor for tag:yaml.org,2002:python/tuple" error
def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))

yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.SafeLoader)
# Depending on PyYAML version/method used by KAN, we might need to register it for the default Loader too
try:
    yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.Loader)
except AttributeError:
    pass # yaml.Loader might not exist in some setups, safe to ignore if SafeLoader is used

# ==========================================
# Import your wrapper and function ZOO
from SALib.sample import sobol as saltelli
from github.workflows.Hyein.toy_KAN_sweep import KANRegressor, FUNCTION_ZOO
from kan.experiments.analysis import find_indices_sign_revert


SA_RC = {
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'figure.autolayout': True,
    'axes.facecolor': 'white',
    'axes.edgecolor': '#444444',
    'axes.linewidth': 0.8,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.labelsize': 12,
    'axes.labelcolor': 'black',
    'axes.titlelocation': 'center',
    'axes.grid': False,
    'xtick.labelsize': 12,
    'xtick.color': 'black',
    'xtick.direction': 'out',
    'ytick.labelsize': 10,
    'ytick.color': 'black',
    'ytick.direction': 'out',
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue LT Pro', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'font.weight': '300',
    'axes.labelweight': '500',
    'text.color': 'black',
    'patch.edgecolor': 'black',
    'patch.linewidth': 0.7,
    'patch.force_edgecolor': True,
    'legend.fontsize': 8,
    'legend.title_fontsize': 9,
    'legend.framealpha': 0.0,
    'legend.edgecolor': '#444444',
    'lines.linewidth': 0.7,
    'lines.markersize': 3,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
}


def main():
    parser = argparse.ArgumentParser(description="Tune KAN for Analytical Functions.")
    parser.add_argument("func_name", type=str, nargs='?', default="log2",
                        choices=FUNCTION_ZOO.keys(),
                        help="Choose a function from the ZOO.")

    args = parser.parse_args()
    data_name = args.func_name
    plt.rcParams.update(SA_RC)
    # ==========================================
    # 1. Setup Paths & Load Model/Scalers
    # ==========================================
    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein', 'analytical_results', data_name)
    savepath = os.path.join(root_dir, "kan_models")

    ckpt_path = os.path.join(savepath, f'{data_name}_best_kan_model')
    scaler_x_path = os.path.join(savepath, f'{data_name}_scaler_X.pkl')
    scaler_y_path = os.path.join(savepath, f'{data_name}_scaler_y.pkl')

    print(f"📂 Loading results from: {savepath}")

    # A. Load Scalers
    if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
        print("❌ Error: Scalers not found.")
        return

    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    # B. Initialize Wrapper & Load Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_wrapper = KANRegressor(device=device)

    try:
        model_wrapper.load_model(ckpt_path)
        model = model_wrapper.model  # Access the actual MultKAN object
        print("✅ KAN Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # ==========================================
    # 2. Regenerate Data
    # ==========================================
    print("\n🎲 Regenerating Train data for analysis...")
    config = FUNCTION_ZOO[data_name]
    target_func = config["func"]
    bounds = config["bounds"]
    feat_names = config["names"]
    nx = len(bounds)

    X_raw = np.random.uniform(low=[b[0] for b in bounds], high=[b[1] for b in bounds], size=(1000, nx))
    y_raw = np.apply_along_axis(target_func, 1, X_raw).reshape(-1, 1)
    # noise = np.random.normal(0, np.std(y_raw) * 0.05, size=y_raw.shape)
    # y_raw = y_raw + noise

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

    # Normalize Inputs (Critical for range analysis 0.1 ~ 0.9)
    X_train_norm = scaler_X.transform(X_train)

    # Create dataset dict (needed for forward pass logic sometimes)
    dataset = {
        'train_input': torch.tensor(X_train_norm, dtype=torch.float32, device=device),
        'train_label': torch.tensor(y_train, dtype=torch.float32, device=device).reshape(-1, 1)
        # Label scaling optional here
    }

    # ==========================================
    # 2.5 [NEW] Plot Input vs Output (Ground Truth vs Prediction)
    # ==========================================

    pred_y_norm = model(dataset['train_input']).detach().cpu().numpy()
    try:
        pred_y = scaler_y.inverse_transform(pred_y_norm)
    except ValueError:
        # Fallback if dimensions mismatch or scaler wasn't fitted on 2D
        pred_y = pred_y_norm

    n_features = X_train.shape[1]
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols

    fig_io, axs_io = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), constrained_layout=True)
    axs_io = axs_io.flatten()

    for i in range(n_features):
        ax = axs_io[i]

        # Plot Ground Truth (Gray)
        # X_train is the raw input (before normalization), y_train is raw output
        ax.scatter(X_train[:, i], y_train, alpha=0.5, c='gray', s=15, label='Ground Truth')

        # Plot Prediction (Red)
        ax.scatter(X_train[:, i], pred_y, alpha=0.5, c='red', s=15, label='Prediction')

        feature_label = feat_names[i] if feat_names and i < len(feat_names) else f"Feature {i}"
        ax.set_xlabel(feature_label)
        ax.set_ylabel("Output y")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_features, len(axs_io)):
        axs_io[i].axis('off')

    # Save & Show
    plot_path_io = os.path.join(savepath, f"{data_name}_input_vs_output.png")
    plt.savefig(plot_path_io, dpi=300)
    # plt.show()

    # Run forward pass once to populate internals (splines, activations)
    model.forward(dataset['train_input'])
    scores_tot = model.feature_score.detach().cpu().numpy()  # Global scores
    #
    # fig_tot, ax_tot = plt.subplots()
    #
    # positions = range(len(scores_tot))
    # bars = ax_tot.bar(positions, scores_tot, color='skyblue', edgecolor='black')
    # ax_tot.bar_label(bars, fmt='%.2f', padding=3)
    # ax_tot.set_xticks(list(positions))  # Set positions first
    # ax_tot.set_xticklabels(feat_names, rotation=15, ha='center')  # Then set text labels
    # ax_tot.set_ylabel("Global Attribution Score")
    # ax_tot.set_title(f"Feature Importance: {data_name}")
    #
    # # Save & Show
    # plot_path_tot = os.path.join(savepath, f"{data_name}_scores_global.png")
    # plt.tight_layout()
    # plt.savefig(plot_path_tot, dpi=300)
    # plt.show()

    with plt.rc_context({'figure.autolayout': False}):
        model.plot()
        plt.savefig(os.path.join(savepath, f"{data_name}_model.png"))
        plt.close()

    # ==========================================
    # 3. Inflection Point Analysis (Layer 0)
    # ==========================================
    print("\n🔍 Analyzing Inflection Points in Layer 0...")
    l = 0
    act = model.act_fun[l]
    ni, no = act.coef.shape[:2]
    coef = act.coef.tolist()
    depth = len(model.act_fun)
    # Pre-allocate indexed by original feature; downstream code uses inflection_points_per_input[mask_idx]
    inflection_points_per_input = [None] * ni
    sort_order_act = np.argsort(scores_tot)[::-1]
    feat_colors = [plt.get_cmap('RdYlBu')(x) for x in np.linspace(0.1, 0.9, ni)]

    fig_eval, axs_eval = plt.subplots(nrows=no, ncols=ni, squeeze=False,
                                      figsize=(max(4 * ni, 6), max(3 * no, 3.5)),
                                      constrained_layout=True)
    fig_spline, axs_spline = plt.subplots(nrows=no, ncols=ni, squeeze=False,
                                          figsize=(max(4 * ni, 6), max(3 * no, 3.5)),
                                          constrained_layout=True)

    for col_pos, i in enumerate(sort_order_act):
        knot_points_actual = act.grid[i, model.k - 1:-2].cpu().detach().numpy()
        feature_inflections_all = []
        for j in range(no):
            ax = axs_eval[j, col_pos]
            ax2 = axs_spline[j, col_pos]
            inputs = model.spline_preacts[l][:, j, i].cpu().detach().numpy()
            outputs = model.spline_postacts[l][:, j, i].cpu().detach().numpy()
            coef_node = coef[i][j]
            knot_indices = np.arange(len(coef_node))

            rank = np.argsort(inputs)
            ax.plot(inputs[rank], outputs[rank], marker='o',
                    color=feat_colors[col_pos], label='Activation')

            slope = [x - y for x, y in zip(coef_node[1:], coef_node[:-1])]
            slope_2nd = [(x - y) * 10 for x, y in zip(slope[1:], slope[:-1])]

            ax2.plot(knot_indices, coef_node, marker='o',
                     color=feat_colors[col_pos], label='Coefficients')

            slope_indices = knot_indices[:-1] + 0.5
            ax2.bar(slope_indices, slope, width=0.3, align='center',
                    hatch='///', edgecolor='dimgray', facecolor='none', label='Slope')

            if depth == 1:
                ax2.bar(slope_indices[1:] - 0.3, slope_2nd, width=0.3, align='center',
                        hatch='xx', edgecolor='steelblue', facecolor='none', label='2nd Slope')

            ax2.set_xticks(knot_indices)
            ax2.set_xticklabels([f"{val:.2f}" for val in knot_points_actual], rotation=45, fontsize=9)

            if depth == 1:
                idx_revert = find_indices_sign_revert(slope_2nd)
                idx_revert = [ir + 1 for ir in idx_revert]
            elif depth == 2:
                idx_revert = find_indices_sign_revert(slope)
            else:
                idx_revert = []

            if idx_revert:
                first_vline = True
                for ir in idx_revert:
                    inflection_val = knot_points_actual[ir]
                    feature_inflections_all.append(inflection_val)
                    label_to_add = "Inflection" if first_vline else "_"
                    ax2.axvline(x=ir, color='green', linestyle='--', alpha=0.7, label=label_to_add)
                    ax.axvline(x=inflection_val, color='green', linestyle='--', alpha=0.7, label=label_to_add)
                    first_vline = False

            ax.set_xlabel(f"{feat_names[i]}")
            ax.set_ylabel(f"node ({l+1}, {j})")
            ax2.set_xlabel(f"{feat_names[i]}")
            ax2.set_ylabel(f"node ({l+1}, {j})")
            ax2.axhline(0, color='dimgray', linestyle='--', alpha=0.4)
            ax2.legend(loc='best')

        feature_inflections = sorted(set(feature_inflections_all))
        inflection_points_per_input[i] = feature_inflections

    fig_eval.savefig(os.path.join(savepath, f"{data_name}_activations_values_L{l}.png"), dpi=300)
    fig_eval.savefig(os.path.join(savepath, f"{data_name}_activations_values_L{l}.svg"), format='svg')
    fig_eval.savefig(os.path.join(savepath, f"{data_name}_activations_values_L{l}.eps"), format='eps')
    fig_spline.savefig(os.path.join(savepath, f"{data_name}_activations_L{l}.png"), dpi=300)
    fig_spline.savefig(os.path.join(savepath, f"{data_name}_activations_L{l}.svg"), format='svg')
    fig_spline.savefig(os.path.join(savepath, f"{data_name}_activations_L{l}.eps"), format='eps')
    plt.close(fig_eval)
    plt.close(fig_spline)
    print(f"📊 Activation analysis saved to: {savepath}")

    # ==========================================
    # 3.5 Attribution Trajectory across Grid Intervals
    # ==========================================
    print("\n📈 Computing Attribution Trajectory across grid intervals...")

    sort_order_global = sort_order_act  # same ordering; feat_colors already defined in section 3
    rank_of_feat = {int(orig): rank for rank, orig in enumerate(sort_order_global)}

    n_cols_traj = min(ni, 3)
    n_rows_traj = (ni + n_cols_traj - 1) // n_cols_traj

    fig_traj, axs_traj = plt.subplots(n_rows_traj, n_cols_traj, squeeze=False,
                                      figsize=(4 * n_cols_traj, 3 * n_rows_traj),
                                      constrained_layout=True)
    axs_traj_flat = axs_traj.flatten()

    for col_pos, split_feat_idx in enumerate(sort_order_global):
        split_feat_idx = int(split_feat_idx)
        ax = axs_traj_flat[col_pos]
        knots = act.grid[split_feat_idx, model.k - 1:-2].cpu().detach().numpy()

        interval_scores = []
        interval_centers = []

        for lb, ub in zip(knots[:-1], knots[1:]):
            mask = (dataset['train_input'][:, split_feat_idx] > lb) & \
                   (dataset['train_input'][:, split_feat_idx] <= ub)
            if torch.any(mask):
                x_slice = dataset['train_input'][mask, :]
                x_std = torch.std(x_slice, dim=0).detach().cpu().numpy()
                model.forward(x_slice)
                score = model.feature_score.detach().cpu().numpy().copy()
                interval_scores.append(score / (x_std + 1e-6))
                interval_centers.append(float((lb + ub) / 2))

        if len(interval_scores) < 2:
            ax.set_visible(False)
            continue

        scores_arr = np.array(interval_scores)  # (n_valid_intervals, ni)
        x_pos = np.array(interval_centers)

        # --- Primary axis: line plots per feature ---
        for orig_idx in sort_order_global:
            orig_idx = int(orig_idx)
            rank = rank_of_feat[orig_idx]
            ax.plot(x_pos, scores_arr[:, orig_idx], marker='o',
                    color=feat_colors[rank],
                    label=f"x{rank}: {feat_names[orig_idx]}")

        for ip in (inflection_points_per_input[split_feat_idx] or []):
            ax.axvline(x=ip, color='green', linestyle='--', alpha=0.7, linewidth=1.2)

        ax.set_xlabel(f"{feat_names[split_feat_idx]}")
        ax.set_ylabel("Normalized Attribution Score")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)

        # --- Secondary axis: relative importance as bar plot ---
        if ni >= 2:
            g1_idx = int(sort_order_global[0])
            g2_idx = int(sort_order_global[1])
            log_ratio = np.log10(
                (scores_arr[:, g1_idx] + 1e-9) / (scores_arr[:, g2_idx] + 1e-9)
            )
            ax2 = ax.twinx()
            ax2.bar(x_pos, log_ratio, width=(x_pos[1] - x_pos[0]) * 0.4 if len(x_pos) > 1 else 0.05,
                    color='dimgray', alpha=0.25, zorder=1,
                    label=r'$\mathcal{R}(x_0,x_1)$')
            ax2.axhline(0, color='dimgray', linestyle='--', alpha=0.4)
            ax2.set_ylabel(r'Relative Importance  $\mathcal{R}(x_0,x_1)$')
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc='best')
        else:
            ax.legend(loc='best')

    for k in range(ni, len(axs_traj_flat)):
        axs_traj_flat[k].set_visible(False)

    traj_base = os.path.join(savepath, f"{data_name}_attribution_trajectory")
    fig_traj.savefig(traj_base + ".png", dpi=300)
    fig_traj.savefig(traj_base + ".svg", format='svg')
    fig_traj.savefig(traj_base + ".eps", format='eps')
    plt.close(fig_traj)
    print(f"📊 Attribution trajectory saved to: {traj_base}.png/svg/eps")

    # ==========================================
    # 4. Range-Based Attribution Scoring (Iterative Search)
    # ==========================================

    # Sort features by global score (Highest -> Lowest)
    sorted_feat_indices = np.argsort(scores_tot)[::-1]

    selected_mask_idx = None
    selected_split_points = None
    masks = []
    labels = []

    print("\n🔍 Searching for a feature that splits data into valid ranges...")

    for mask_idx in sorted_feat_indices:
        feat_name = feat_names[mask_idx]
        print(f"   Checking Feature {mask_idx} ({feat_name})...", end=" ")

        # Get valid inflection points for this feature (within 0.1~0.9 range)
        raw_ips = inflection_points_per_input[mask_idx]
        valid_ips = [ip for ip in raw_ips if ip is not None and 0.1 < ip < 0.9]

        # Remove duplicates and sort
        unique_ips = sorted(list(set([round(ip, 3) for ip in valid_ips])))

        # If no inflection points, we can't split "into both areas"
        if len(unique_ips) == 0:
            print("Skipping (No inflection points in 0.1-0.9).")
            continue

        # Define Intervals: [0.1, ip1, ip2, ..., 0.9]
        mask_interval = [0.1] + unique_ips + [0.9]

        # Create Candidate Masks
        x_mask_data = dataset['train_input'][:, mask_idx]
        candidate_masks = [((x_mask_data > lb) & (x_mask_data <= ub))
                           for lb, ub in zip(mask_interval[:-1], mask_interval[1:])]

        # Check if "mask exists in both areas"
        # (Meaning: At least 2 intervals have samples)
        non_empty_count = sum([1 for m in candidate_masks if torch.any(m)])

        if non_empty_count >= 2:
            print(f"✅ Selected! (Found {non_empty_count} active intervals)")
            selected_mask_idx = mask_idx
            selected_split_points = mask_interval
            masks = candidate_masks
            labels = [f'{lb:.2f} < x{mask_idx} <= {ub:.2f}' for lb, ub in zip(mask_interval[:-1], mask_interval[1:])]
            break
        else:
            print(f"Skipping (Data only exists in {non_empty_count} interval).")

    # Fallback: If loop finishes without success, pick the top feature anyway (to prevent crash)
    if selected_mask_idx is None:
        print("⚠️ Warning: No feature provided a valid split. Defaulting to top feature.")
        selected_mask_idx = sorted_feat_indices[0]
        # Re-generate masks for the top feature (even if empty/single)
        # ... (simplified logic just to ensure variables exist)
        x_mask_data = dataset['train_input'][:, selected_mask_idx]
        masks = [(x_mask_data > -np.inf)]  # Dummy mask
        labels = ["All Range"]

    # Now calculate scores for the chosen masks
    print(f"\n✂️ Slicing data based on Feature {selected_mask_idx}...")
    scores_interval_norm = []

    # Compute Scores per Interval
    for i, mask in enumerate(masks):
        if torch.any(mask):
            x_tensor_masked = dataset['train_input'][mask, :]

            # Standard deviation of input in this slice (used for normalization)
            x_std = torch.std(x_tensor_masked, dim=0).detach().cpu().numpy()

            # Forward pass on masked data to get local attribution
            model.forward(x_tensor_masked)
            score_masked = model.feature_score.detach().cpu().numpy()

            # Normalize score
            score_norm = score_masked / (x_std + 1e-6)
            scores_interval_norm.append(score_norm)
            print(f"   Interval {labels[i]}: {mask.sum().item()} samples")
        else:
            scores_interval_norm.append(np.zeros(scores_tot.shape))
            print(f"   Interval {labels[i]}: 0 samples (Skipping)")

    # ==========================================
    # 4.5 [NEW] Save Range Split Data for NN Training
    # ==========================================
    split_data_savepath = os.path.join(savepath, f"{data_name}_range_split_data.pkl")

    split_data = {
        'dataset': dataset,
        'masks': masks,
        'labels': labels,
        'selected_mask_idx': selected_mask_idx,
        'selected_mask_name': feat_names[selected_mask_idx],
        'split_points': selected_split_points,  # interval boundaries in [0.1, 0.9] space
        'inflection_points_per_input': inflection_points_per_input,  # per-feature inflection points
        'feature_names': feat_names,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }

    joblib.dump(split_data, split_data_savepath)

    # Save KAN interval scores as CSV for cross-method comparison
    kan_scores_df = pd.DataFrame(scores_interval_norm, columns=feat_names)
    kan_scores_df.insert(0, 'Interval_Label', labels)
    kan_scores_df.to_csv(os.path.join(savepath, f"{data_name}_kan_interval_scores.csv"), index=False)

    # ==========================================
    # 5. Plot Range-Based Scores
    # ==========================================
    width = 0.2
    n_features = scores_tot.shape[0]
    n_intervals = len(scores_interval_norm)

    fig, ax = plt.subplots(figsize=(max(8, n_intervals * 2), 5))

    # X-axis: Intervals
    x_positions = np.arange(n_intervals)

    # We want to show bars for EACH feature within each interval group
    # But usually, we want to see how feature importance changes across intervals.
    # Let's group by Interval on X-axis.

    max_score = max([max(s) for s in scores_interval_norm]) if scores_interval_norm else 1.0

    for feat_idx in range(n_features):
        # Extract score of this feature across all intervals
        feat_scores = [s[feat_idx] for s in scores_interval_norm]

        # Offset bars
        offset = (feat_idx - n_features / 2) * width + width / 2
        bars = ax.bar(x_positions + offset, feat_scores, width, label=f"{feat_names[feat_idx]}")
        # ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=15, ha='center', fontsize=9)
    ax.set_ylabel("Normalized Attribution Score")
    ax.set_title(f"Feature Importance per Range (sliced by {feat_names[mask_idx]})")
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.set_ylim(0, max_score * 1.2)
    plt.tight_layout()

    plot_path_score = os.path.join(savepath, f"{data_name}_scores_interval_x{mask_idx}.png")
    plt.savefig(plot_path_score)
    # plt.show()
    print(f"📊 Range-based score plot saved to: {plot_path_score}")


    # ==========================================
    # 6. Attribution Scoring on Saltelli Dataset
    # (mirrors toy_KAN_sweep.py lines 339-373)
    # ==========================================
    print("\n🧂 [6] Computing Attribution Scores on Saltelli Dataset...")

    problem = {
        'num_vars': nx,
        'names': feat_names,
        'bounds': bounds
    }
    X_saltelli_raw = saltelli.sample(problem, 512, calc_second_order=True, seed=42)
    X_saltelli_norm = scaler_X.transform(X_saltelli_raw)
    X_saltelli_tensor = torch.tensor(X_saltelli_norm, dtype=torch.float32, device=device)

    model.forward(X_saltelli_tensor)
    scores_saltelli = model.feature_score.detach().cpu().numpy()

    if len(scores_saltelli.shape) > 1:
        scores_saltelli = scores_saltelli.flatten()

    fig_s, ax_s = plt.subplots()
    positions = range(len(scores_saltelli))
    bars = ax_s.bar(positions, scores_saltelli, color='skyblue', edgecolor='black')
    ax_s.bar_label(bars, fmt='%.2f', padding=3)
    ax_s.set_xticks(list(positions))
    ax_s.set_xticklabels(feat_names, rotation=15, ha='center')
    ax_s.set_ylabel("Global Attribution Score")
    ax_s.set_title(f"Feature Importance (Saltelli): {data_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, f"{data_name}_scores_global_saltelli.png"), dpi=300)
    # plt.show()

    df_scores_saltelli = pd.DataFrame({
        'Feature': feat_names,
        'Global_Attribution_Score': scores_saltelli
    }).sort_values(by='Global_Attribution_Score', ascending=False)
    df_scores_saltelli.to_csv(
        os.path.join(savepath, f"{data_name}_global_attribution_scores_saltelli.csv"), index=False
    )
    print(f"📊 Saltelli attribution scores saved to: {savepath}")


if __name__ == "__main__":
    main()