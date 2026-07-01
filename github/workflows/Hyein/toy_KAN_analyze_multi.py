import argparse
import os
import json
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools  # <--- [IMPORTANT] Added for multi-feature combinations
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
from github.workflows.Hyein.toy_KAN_sweep import KANRegressor
from github.workflows.Hyein.toy_analytic_SHAP_Sobol import FUNCTION_ZOO
from kan.experiments.analysis import find_index_sign_revert


def main():
    parser = argparse.ArgumentParser(description="Tune KAN for Analytical Functions.")
    parser.add_argument("func_name", type=str, nargs='?', default="exponential",
                        choices=FUNCTION_ZOO.keys(),
                        help="Choose a function from the ZOO.")

    args = parser.parse_args()
    data_name = args.func_name
    # ==========================================
    # 1. Setup Paths & Load Model/Scalers
    # ==========================================
    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein', 'analytical_results', data_name)
    savepath = os.path.join(root_dir, "kan_models")

    ckpt_path = os.path.join(savepath, f'{data_name}_best_kan_model')
    scaler_x_path = os.path.join(savepath, f'{data_name}_mlp_scaler_X.pkl')
    scaler_y_path = os.path.join(savepath, f'{data_name}_mlp_scaler_y.pkl')

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
        ax.set_title(f"{feature_label} vs Output")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_features, len(axs_io)):
        axs_io[i].axis('off')

    plt.suptitle(f"Input vs Output Analysis: {data_name}", fontsize=14)

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

    model.plot()
    plt.savefig(os.path.join(savepath, f"{data_name}_model.png"))

    # ==========================================
    # 3. Inflection Point Analysis (Layer 0)
    # ==========================================
    print("\n🔍 Analyzing Inflection Points in Layer 0...")

    depth = len(model.act_fun)
    l = 0  # Analyze Layer 0
    act = model.act_fun[l]
    ni, no = act.coef.shape[:2]
    coef = act.coef.tolist()

    inflection_points_per_input = []  # Store list of inflection points for each input feature

    fig, axs = plt.subplots(nrows=no, ncols=ni, squeeze=False,
                            figsize=(max(2.5 * ni, 6), max(2.5 * no, 3.5)),
                            constrained_layout=True)

    for i in range(ni):  # For each input feature
        feature_inflections = []
        for j in range(no):  # For each output node of the layer
            ax = axs[j, i]

            # 1. Get Data
            # Note: spline_preacts might not be populated unless update_grid_from_samples or similar was called during training/forward
            # We assume model has tracked data. If not, we might need model.forward(dataset['train_input']) again.
            inputs = model.spline_preacts[l][:, j, i].cpu().detach().numpy()
            outputs = model.spline_postacts[l][:, j, i].cpu().detach().numpy()

            coef_node = coef[i][j]
            num_knot = act.grid.shape[1]
            spline_radius = int((num_knot - len(coef_node)) / 2)

            # 2. Plot Activations
            rank = np.argsort(inputs)
            ax.plot(inputs[rank], outputs[rank], marker='o', ms=2, lw=1, label='Activations')

            # 3. Plot Coefficients & Slope
            ax2 = ax.twinx()
            # Plot coefficients (control points)
            ax2.scatter(act.grid[i, spline_radius:-spline_radius].cpu(), coef_node,
                        s=20, color='white', edgecolor='k', label='Coefficients')

            # Calculate Slope
            slope = [x - y for x, y in zip(coef_node[1:], coef_node[:-1])]
            slope_2nd = [x - y for x, y in zip(slope[1:], slope[:-1])]
            bar_width = (act.grid[i, 1:] - act.grid[i, :-1]).mean().item() / 2  # Approx width

            # Plot Slope
            ax2.bar(act.grid[i, spline_radius:-(spline_radius + 1)].cpu(), slope,
                    width=bar_width, align='center', color='r', alpha=0.3, label='Slope')
            if depth < 2:
                ax2.bar(act.grid[i, spline_radius:-(spline_radius + 2)].cpu() + bar_width/3, slope_2nd,
                        width=bar_width, align='center', color='g', alpha=0.3, label='2nd Slope')

            ax.set_title(f'in {i} -> out {j}', fontsize=9)

            # 4. Find Inflection
            if depth == 1:
                idx_revert = find_index_sign_revert(slope_2nd)
            elif depth == 2:
                idx_revert = find_index_sign_revert(slope)
            if idx_revert is not None:
                inflection_val = act.grid[i, spline_radius + idx_revert].item()
                feature_inflections.append(inflection_val)
                # Mark on plot
                ax.axvline(x=inflection_val, color='g', linestyle='--', alpha=0.7)

        inflection_points_per_input.append(feature_inflections)

    plt.suptitle(f"Layer {l} Activation Analysis: {data_name}", fontsize=12)
    plot_path_act = os.path.join(savepath, f"{data_name}_activations_L{l}.png")
    plt.savefig(plot_path_act)
    # plt.show()
    print(f"📊 Activation analysis saved to: {plot_path_act}")

    # ==========================================
    # 4. Range-Based Attribution Scoring (Iterative Search)
    # ==========================================

    # Settings
    NUM_FEATURES_TO_COMBINE = 2
    MIN_NON_EMPTY_INTERVALS = 2

    # Sort features by global score (Highest -> Lowest)
    sorted_feat_indices = np.argsort(scores_tot)[::-1]

    selected_features_data = []  # List of dicts: {'index', 'masks', 'labels'}

    print(f"\n🔍 Searching for top {NUM_FEATURES_TO_COMBINE} features that split data into valid ranges...")

    for mask_idx in sorted_feat_indices:
        if len(selected_features_data) >= NUM_FEATURES_TO_COMBINE:
            break

        feat_name = feat_names[mask_idx]
        raw_ips = inflection_points_per_input[mask_idx]
        valid_ips = [ip for ip in raw_ips if ip is not None and 0.1 < ip < 0.9]
        unique_ips = sorted(list(set([round(ip, 3) for ip in valid_ips])))

        if len(unique_ips) == 0:
            continue

        # Define Intervals: [0.1, ip1, ip2, ..., 0.9]
        mask_interval = [0.1] + unique_ips + [0.9]
        x_mask_data = dataset['train_input'][:, mask_idx]

        current_feat_masks = []
        current_feat_labels = []

        for lb, ub in zip(mask_interval[:-1], mask_interval[1:]):
            m = ((x_mask_data > lb) & (x_mask_data <= ub))
            current_feat_masks.append(m)
            current_feat_labels.append(f'{lb:.2f}<x{mask_idx}<{ub:.2f}')

        non_empty_count = sum([1 for m in current_feat_masks if torch.any(m)])

        if non_empty_count >= MIN_NON_EMPTY_INTERVALS:
            print(f"   ✅ Selected Feature {mask_idx} ({feat_name}): Found {non_empty_count} active intervals")
            selected_features_data.append({
                'index': mask_idx,
                'masks': current_feat_masks,
                'labels': current_feat_labels
            })

    # Combine Masks (Cartesian Product)
    final_masks = []
    final_labels = []

    if not selected_features_data:
        print("⚠️ Warning: No valid splitting features found. Defaulting to top feature (All Range).")
        fallback_idx = sorted_feat_indices[0]
        x_mask_data = dataset['train_input'][:, fallback_idx]
        final_masks = [(x_mask_data > -np.inf)]
        final_labels = [f"All Range"]
        selected_features_indices = [fallback_idx]
    else:
        print(f"\n🔗 Combining intervals from {len(selected_features_data)} features...")
        selected_features_indices = [d['index'] for d in selected_features_data]

        lists_of_masks = [d['masks'] for d in selected_features_data]
        lists_of_labels = [d['labels'] for d in selected_features_data]

        for combined_mask_tuple, combined_label_tuple in zip(itertools.product(*lists_of_masks),
                                                             itertools.product(*lists_of_labels)):

            combined_m = combined_mask_tuple[0]
            for m in combined_mask_tuple[1:]:
                combined_m = combined_m & m

            combined_l = " & ".join(combined_label_tuple)

            if torch.any(combined_m):
                final_masks.append(combined_m)
                final_labels.append(combined_l)

        print(f"   -> Generated {len(final_masks)} non-empty combined regions.")

    # Calculate scores for final masks
    print(f"\n✂️ Scoring {len(final_masks)} regions...")
    scores_interval_norm = []

    for i, mask in enumerate(final_masks):
        x_tensor_masked = dataset['train_input'][mask, :]
        x_std = torch.std(x_tensor_masked, dim=0).detach().cpu().numpy()

        model.forward(x_tensor_masked)
        score_masked = model.feature_score.detach().cpu().numpy()
        score_norm = score_masked / (x_std + 1e-6)
        scores_interval_norm.append(score_norm)
        print(f"   Region {i + 1}: {mask.sum().item()} samples")

    # ==========================================
    # 4.5 [NEW] Save Range Split Data for NN Training
    # ==========================================
    split_data_savepath = os.path.join(savepath, f"{data_name}_range_split_data.pkl")

    split_data = {
        'dataset': dataset,
        'masks': final_masks,  # UPDATED: Now saving the combined masks
        'labels': final_labels,  # UPDATED: Combined labels
        'selected_features_indices': selected_features_indices,  # UPDATED: List of indices
        'feature_names': feat_names,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    joblib.dump(split_data, split_data_savepath)

    # ==========================================
    # 5. Plot Range-Based Scores
    # ==========================================
    width = 0.08
    n_features = scores_tot.shape[0]
    n_intervals = len(scores_interval_norm)

    fig, ax = plt.subplots(figsize=(max(10, n_intervals * 1.5), 6))
    x_positions = np.arange(n_intervals)
    max_score = max([max(s) for s in scores_interval_norm]) if scores_interval_norm else 1.0

    for feat_idx in range(n_features):
        feat_scores = [s[feat_idx] for s in scores_interval_norm]
        offset = (feat_idx - n_features / 2) * width + width / 2
        bars = ax.bar(x_positions + offset, feat_scores, width, label=f"{feat_names[feat_idx]}")

    ax.set_xticks(x_positions)
    short_labels = [l if len(l) < 30 else l[:15] + "..." + l[-10:] for l in final_labels]
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)

    ax.set_ylabel("Normalized Attribution Score")
    indices_str = "_".join(map(str, selected_features_indices))
    ax.set_title(f"Feature Importance per Range (Features {indices_str})")
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')
    ax.set_ylim(0, max_score * 1.2)
    plt.tight_layout()

    plot_path_score = os.path.join(savepath, f"{data_name}_scores_interval_combined.png")
    plt.savefig(plot_path_score)
    # plt.show()
    print(f"📊 Range-based score plot saved to: {plot_path_score}")


if __name__ == "__main__":
    main()