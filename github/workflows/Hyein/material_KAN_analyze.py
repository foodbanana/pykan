import argparse
import os
import json
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import itertools  # <--- [IMPORTANT] Added for multi-feature combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import yaml
from kan.custom_processing import remove_outliers_iqr


# ==========================================
# [FIX] Register Python Tuple for YAML Loading
# ==========================================
def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))


yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.SafeLoader)
try:
    yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.Loader)
except AttributeError:
    pass

# ==========================================
# Import your wrapper and function ZOO
from github.workflows.Hyein.toy_KAN_sweep import KANRegressor
from kan.experiments.analysis import find_indices_sign_revert


def main():
    parser = argparse.ArgumentParser(description="Run SHAP and Sobol analysis for a specific dataset.")
    parser.add_argument("data_name", type=str, nargs='?', default="CO2HEx10",
                        help="The name of the dataset")
    parser.add_argument("rand_seed", type=int, nargs='?', default=None,
                        help="The random seed (default: None=42)")

    args = parser.parse_args()
    data_name = args.data_name
    rand_seed = args.rand_seed

    if "CO2RR" in data_name:
        data_on_contour = False
    else:
        data_on_contour = True

    # ==========================================
    # 1. Setup Paths & Load Model/Scalers
    # ==========================================
    root_dir = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein')
    filepath = os.path.join(root_dir, "data", f"{data_name}.csv")
    if rand_seed is not None:
        savepath = os.path.join(root_dir, "material_kan_models", data_name + f"_seed_{rand_seed}")
    else:
        savepath = os.path.join(root_dir, "material_kan_models", data_name)
        rand_seed = 42

    ckpt_path = os.path.join(savepath, f'{data_name}_best_kan_model')
    scaler_x_path = os.path.join(savepath, f'{data_name}_mlp_scaler_X.pkl')
    scaler_y_path = os.path.join(savepath, f'{data_name}_mlp_scaler_y.pkl')

    print(f"DATASET: {data_name}")

    if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
        print("❌ Error: Scalers not found.")
        return

    scaler_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_wrapper = KANRegressor(device=device)

    try:
        model_wrapper.load_model(ckpt_path)
        model = model_wrapper.model
        print("✅ KAN Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # ==========================================
    # 2. Regenerate Data
    # ==========================================
    if not os.path.exists(filepath):
        print(f"❌ Error: Data file not found at {filepath}")
        return

    filedata = pd.read_csv(filepath)
    name_X = filedata.columns[:-1].tolist()
    name_y = filedata.columns[-1]
    df_in = filedata[name_X]
    df_out = filedata[[name_y]]

    df_in_final, df_out_final = remove_outliers_iqr(df_in, df_out)

    removed_count = len(df_in) - len(df_in_final)
    print(f"# of data after removing outliers: {len(df_in_final)} ({removed_count} removed)")

    X = df_in_final[name_X].values
    y = df_out_final[name_y].values.reshape(-1, 1)

    X_temp_denorm, X_test_denorm, y_temp_denorm, y_test_denorm = train_test_split(X, y, test_size=0.2, random_state=rand_seed)
    X_train_denorm, X_val_denorm, y_train_denorm, y_val_denorm = train_test_split(X_temp_denorm, y_temp_denorm,
                                                                                  test_size=0.2, random_state=rand_seed)
    print(f"Train/Validation/Test : {len(X_train_denorm)} / {len(X_val_denorm)} / {len(X_test_denorm)}")

    feat_names = name_X

    X_train_norm = scaler_X.fit_transform(X_train_denorm)
    y_train_norm = scaler_y.fit_transform(y_train_denorm)
    X_test_norm = scaler_X.fit_transform(X_test_denorm)
    y_test_norm = scaler_y.fit_transform(y_test_denorm)

    dataset = {
        'train_input': torch.tensor(X_train_norm, dtype=torch.float32, device=device),
        'train_label': torch.tensor(y_train_denorm, dtype=torch.float32, device=device).reshape(-1, 1),
        'test_input': torch.tensor(X_test_norm, dtype=torch.float32, device=device),
        'test_label': torch.tensor(y_test_denorm, dtype=torch.float32, device=device).reshape(-1, 1)
    }

    # ==========================================
    # 2.5 Plot Input vs Output
    # ==========================================
    pred_y_norm = model(dataset['train_input']).detach().cpu().numpy()
    try:
        pred_y = scaler_y.inverse_transform(pred_y_norm)
    except ValueError:
        pred_y = pred_y_norm

    n_features = X_train_denorm.shape[1]
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols

    fig_io, axs_io = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), constrained_layout=True)
    axs_io = axs_io.flatten()

    for i in range(n_features):
        ax = axs_io[i]
        ax.scatter(X_train_denorm[:, i], y_train_denorm, alpha=0.5, c='gray', s=15, label='Ground Truth')
        ax.scatter(X_train_denorm[:, i], pred_y, alpha=0.5, c='red', s=15, label='Prediction')
        feature_label = feat_names[i] if feat_names and i < len(feat_names) else f"Feature {i}"
        ax.set_xlabel(feature_label)
        ax.set_ylabel("Output y")
        # ax.set_title(f"{feature_label} vs Output")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    for i in range(n_features, len(axs_io)):
        axs_io[i].axis('off')
    plt.suptitle(f"Input vs Output Analysis: {data_name}", fontsize=14)
    plt.savefig(os.path.join(savepath, f"{data_name}_input_vs_output.png"), dpi=300)
    # plt.show()

    model.forward(dataset['train_input'])
    scores_tot = model.feature_score.detach().cpu().numpy()
    model.plot()
    plt.savefig(os.path.join(savepath, f"{data_name}_model.png"))

    # ==========================================
    # 3. Inflection Point Analysis (Layer 0)
    # ==========================================
    print("\n🔍 Analyzing Inflection Points in Layer 0...")
    l = 0
    act = model.act_fun[l]
    ni, no = act.coef.shape[:2]
    coef = act.coef.tolist()
    depth = len(model.act_fun)
    inflection_points_per_input = []

    fig_eval, axs_eval = plt.subplots(nrows=no, ncols=ni, squeeze=False, figsize=(max(3 * ni, 6), max(2.5 * no, 3.5)),
                            constrained_layout=True)
    fig_spline, axs_spline = plt.subplots(nrows=no, ncols=ni, squeeze=False, figsize=(max(3 * ni, 6), max(2.5 * no, 3.5)),
                            constrained_layout=True)

    for i in range(ni):
        knot_points_actual = act.grid[i, model.k - 1:-2].cpu().detach().numpy()
        feature_inflections_all = []
        for j in range(no):
            ax = axs_eval[j, i]
            ax2 = axs_spline[j, i]
            inputs = model.spline_preacts[l][:, j, i].cpu().detach().numpy()
            outputs = model.spline_postacts[l][:, j, i].cpu().detach().numpy()
            coef_node = coef[i][j]
            knot_indices = np.arange(len(coef_node))

            rank = np.argsort(inputs)
            ax.plot(inputs[rank], outputs[rank], marker='o', ms=2, lw=1, label='Activation function')

            slope = [x - y for x, y in zip(coef_node[1:], coef_node[:-1])]
            slope_2nd = [(x - y)*10 for x, y in zip(slope[1:], slope[:-1])]

            # Plot Slope
            ax2.plot(knot_indices, coef_node, marker='o', ms=4, lw=1,
                     color='dimgray', markerfacecolor='none', label='Coefficients')

            slope = [x - y for x, y in zip(coef_node[1:], coef_node[:-1])]
            slope_indices = knot_indices[:-1] + 0.5
            ax2.bar(slope_indices, slope,
                    width=0.3,
                    align='center',
                    hatch='xx',  # Red diagonal hatching
                    edgecolor='r',  # The color of the hatching lines
                    facecolor='none',  # No solid color fill
                    linewidth=0.8,  # Outline width
                    label='Slope')

            if depth == 1:
                # 3. Plot 2nd Slope with green 'x' hatching
                ax2.bar(slope_indices[1:] - 0.3, slope_2nd,
                        width=0.3,  # Making it narrower so they can co-exist
                        align='center',
                        hatch='///',  # Green 'x' hatching
                        edgecolor='b',  # The color of the hatching lines
                        facecolor='none',  # No solid color fill
                        linewidth=0.8,  # Outline width
                        label='2nd Slope')

            ax2.set_xticks(knot_indices)
            ax2.set_xticklabels([f"{val:.2f}" for val in knot_points_actual],
                                rotation=45, fontsize=7)

            # Logic for inflection points
            if depth == 1:
                idx_revert = find_indices_sign_revert(slope_2nd)
                idx_revert = [ir + 1 for ir in idx_revert]
            elif depth == 2:
                idx_revert = find_indices_sign_revert(slope)
            else:
                idx_revert = None

            if idx_revert:
                first_vline=True
                for ir in idx_revert:
                    inflection_val = knot_points_actual[ir]
                    feature_inflections_all.append(inflection_val)
                    label_to_add = "Inflection" if first_vline else "_"

                    # Vlines use the index 'ir' for the sequence plot
                    ax2.axvline(x=ir, color='g', linestyle='--', alpha=0.7, lw=1.5, label=label_to_add)
                    # Vlines use the actual value for the activation plot
                    ax.axvline(x=inflection_val, color='g', linestyle='--', alpha=0.7, lw=1.5, label=label_to_add)

                    first_vline = False  # Subsequent lines will be ignored by legend

            ax.set_title(f'in {i} -> out {j}', fontsize=9)
            ax2.set_title(f'in {i} -> out {j}', fontsize=9)
            ax2.legend(loc='best', fontsize=7)

        feature_inflections = sorted(set(feature_inflections_all))
        inflection_points_per_input.append(feature_inflections)

    # Save the first figure (activations)
    fig_eval.savefig(os.path.join(savepath, f"{data_name}_activations_values_L{l}.png"), dpi=300)
    fig_eval.savefig(os.path.join(savepath, f"{data_name}_activations_values_L{l}.svg"), format='svg')

    # Save the second figure (spline coefficients/slopes)
    fig_spline.savefig(os.path.join(savepath, f"{data_name}_activations_L{l}.png"), dpi=300)
    fig_spline.savefig(os.path.join(savepath, f"{data_name}_activations_L{l}.svg"), format='svg')

    # plt.show()
    plt.close(fig_eval)
    plt.close(fig_spline)

    # ==========================================
    # 4. Multi-Feature Range-Based Slicing
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
    # 4.5 Save Range Split Data
    # ==========================================
    split_data_savepath = os.path.join(savepath, f"{data_name}_range_split_data.pkl")
    print(f"\n💾 Saving range split data to: {split_data_savepath}")

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
    n_features_plot = scores_tot.shape[0]
    n_intervals = len(scores_interval_norm)

    # Dynamic figure size based on number of intervals
    fig, ax = plt.subplots(figsize=(max(10, n_intervals * 1.5), 6))
    x_positions = np.arange(n_intervals)
    max_score = max([max(s) for s in scores_interval_norm]) if scores_interval_norm else 1.0

    for feat_idx in range(n_features_plot):
        feat_scores = [s[feat_idx] for s in scores_interval_norm]
        offset = (feat_idx - n_features_plot / 2) * width + width / 2
        ax.bar(x_positions + offset, feat_scores, width, label=f"x{feat_idx}: {feat_names[feat_idx]}")

    ax.set_xticks(x_positions)
    # Truncate labels if too long for the plot
    short_labels = [l if len(l) < 30 else l[:15] + "..." + l[-10:] for l in final_labels]
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)

    ax.set_ylabel("Normalized Attribution Score")
    # Join indices for the filename/title
    indices_str = "_".join(map(str, selected_features_indices))
    ax.set_title(f"Feature Importance per Range (Features {indices_str})")
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')
    ax.set_ylim(0, max_score * 1.2)
    plt.tight_layout()

    plot_path_score = os.path.join(savepath, f"{data_name}_scores_interval_combined.png")
    plt.savefig(plot_path_score)
    # plt.show()
    print(f"📊 Range-based score plot saved to: {plot_path_score}")

    # ==========================================
    # 5.5 Save Range-Based Scores to CSV
    # ==========================================
    # Create a list to store row data
    range_scores_data = []

    for i, (label, scores) in enumerate(zip(final_labels, scores_interval_norm)):
        # Create a dictionary for the row: {Region_Label, Feature_0_Score, ...}
        row = {'Region_Index': i, 'Region_Label': label}
        for feat_idx, score_val in enumerate(scores):
            feat_name = feat_names[feat_idx]
            row[f'x{feat_idx}_{feat_name}'] = score_val
        range_scores_data.append(row)

    # Convert to DataFrame
    df_range_scores = pd.DataFrame(range_scores_data)

    # Save to CSV
    range_score_csv_path = os.path.join(savepath, f'{data_name}_range_attribution_scores.csv')
    df_range_scores.to_csv(range_score_csv_path, index=False)

    print(f"✅ Range-based scores saved to: {range_score_csv_path}")

    # ==========================================
    # 5.9. NEW: Save Individual Plots for Each Range
    # ==========================================
    print(f"⏳ Generating {n_intervals} individual range plots...")

    for i in range(n_intervals):
        # Create a fresh figure for each range
        fig_sep, ax_sep = plt.subplots(figsize=(4, 4))

        # Get scores for this specific interval
        current_interval_scores = scores_interval_norm[i]
        x_pos_sep = np.arange(n_features_plot)

        # Draw bars - using a distinct color map for clarity
        colors = plt.cm.get_cmap('tab20', n_features_plot)
        ax_sep.bar(x_pos_sep, current_interval_scores, width=0.6,
                   color=[colors(j) for j in range(n_features_plot)],
                   edgecolor='black', alpha=0.8)

        # Formatting
        ax_sep.set_xticks(x_pos_sep)
        ax_sep.set_xticklabels(feat_names, rotation=45, ha='right', fontsize=9)
        ax_sep.set_ylabel("Normalized Attribution Score")

        current_label = final_labels[i]
        ax_sep.set_title(f"Feature Importance\nRange: {current_label}", fontsize=11)

        # Set Y-axis limit (using the global max for consistency across all plots)
        ax_sep.set_ylim(0, max_score * 1.2)
        ax_sep.grid(axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()

        # Create a safe filename (removes spaces and special characters)
        clean_label = "".join([c if c.isalnum() else "_" for c in current_label])
        indiv_plot_name = f"{data_name}_score_range_{i}_{clean_label}.png"
        indiv_plot_path = os.path.join(savepath, indiv_plot_name)

        plt.savefig(indiv_plot_path, dpi=300)
        plt.close(fig_sep)  # Close to prevent memory accumulation

    print(f"✅ All individual range plots saved in: {savepath}")

    # ==========================================
    # 6. Global Scores
    # ==========================================
    fig_tot, ax_tot = plt.subplots(figsize=(4, 4))

    positions = range(len(scores_tot))
    bars = ax_tot.bar(positions, scores_tot, color='skyblue', edgecolor='black')
    ax_tot.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
    ax_tot.set_xticks(list(positions))  # Set positions first
    ax_tot.set_xticklabels(feat_names, rotation=45, ha='center', fontsize=9)  # Then set text labels
    ax_tot.set_ylabel("Global Attribution Score", fontsize=12)
    ax_tot.set_title(f"Feature Importance: {data_name}")

    # Adjust Y-limit
    if len(scores_tot) > 0:
        ax.set_ylim(0, max(scores_tot) * 1.15)
    plt.tight_layout()

    # Save & Show
    plot_path_tot = os.path.join(savepath, f"{data_name}_scores_global.png")
    plt.savefig(plot_path_tot, dpi=300)
    # plt.show()

    # Ensure scores_tot is a flat 1D array
    if len(scores_tot.shape) > 1:
        scores_tot = scores_tot.flatten()

    df_scores = pd.DataFrame({
        'Feature': feat_names,
        'Global_Attribution_Score': scores_tot
    })

    # Optional: Sort by importance
    df_scores = df_scores.sort_values(by='Global_Attribution_Score', ascending=False)

    # 3. Save to CSV
    score_csv_path = os.path.join(savepath, f'{data_name}_global_attribution_scores.csv')
    df_scores.to_csv(score_csv_path, index=False)

    # ==========================================
    # 7. Parity Plot
    # ==========================================

    y_pred_test_norm = model.forward(dataset['test_input']).detach().numpy()
    r2_test = r2_score(y_test_norm, y_pred_test_norm)

    plt.figure(figsize=(4, 4))
    y_pred_test = scaler_y.inverse_transform(y_pred_test_norm.reshape(1, -1))
    plt.scatter(y_test_denorm, y_pred_test, alpha=0.6, color='skyblue', edgecolors='k', s=30, label='Test Data')

    # Perfect fit line
    min_val = min(y_test_denorm.min(), y_pred_test.min())
    max_val = max(y_test_denorm.max(), y_pred_test.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')

    plt.title(f"Parity Plot: {data_name} ($R^2 = {r2_test:.3f}$)")
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plot_path = os.path.join(savepath, f"{data_name}_parity_plot.png")
    plt.savefig(plot_path, dpi=300)
    # plt.show()
    print(f"📊 Parity plot saved to: {plot_path}")

    # ==========================================
    # 8. Plot Input vs Output (Colored by Combined Range)
    # ==========================================
    print("\n📈 Plotting Input vs Output (Original Data Colored by Range)...")

    pred_y_norm = model(dataset['train_input']).detach().cpu().numpy()
    try:
        pred_y = scaler_y.inverse_transform(pred_y_norm)
    except ValueError:
        pred_y = pred_y_norm

    cmap = plt.get_cmap('tab10')
    # Use final_masks here
    colors = [cmap(k % 10) for k in range(len(final_masks))]

    n_features = X_train_denorm.shape[1]
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols

    fig_io, axs_io = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), constrained_layout=True)
    axs_io = axs_io.flatten()

    for i in range(n_features):
        ax = axs_io[i]
        # Background: Predictions
        ax.scatter(X_train_denorm[:, i], pred_y, alpha=0.15, c='gray', s=20, label='Prediction')

        # Foreground: Original Data Colored by Slice
        for m_idx, (mask, label) in enumerate(zip(final_masks, final_labels)):
            if torch.is_tensor(mask):
                mask_np = mask.cpu().numpy().astype(bool)
            else:
                mask_np = mask

            if mask_np.sum() > 0:
                x_seg = X_train_denorm[mask_np, i]
                y_seg_original = y_train_denorm[mask_np]

                # Truncate label for legend
                lbl_short = label if len(label) < 20 else label[:8] + "..." + label[-8:]
                ax.scatter(x_seg, y_seg_original, alpha=0.9, s=25,
                           color=colors[m_idx], label=lbl_short)

        feature_label = f"x{i}: {feat_names[i]}" if feat_names and i < len(feat_names) else f"Feature {i}"
        ax.set_xlabel(feature_label)
        ax.set_ylabel("Output y")

        if i == 0:
            ax.legend(loc='upper left', fontsize=6, markerscale=1.5)
        ax.grid(True, alpha=0.3)

    for i in range(n_features, len(axs_io)):
        axs_io[i].axis('off')

    plt.suptitle(f"Input vs Output (Colored by {indices_str}): {data_name}", fontsize=14)
    plot_path_io = os.path.join(savepath, f"{data_name}_in_out_range_combined.png")
    plt.savefig(plot_path_io, dpi=300)
    # plt.show()
    print(f"📊 Colored Input-Output plots saved to: {plot_path_io}")

    # ==========================================
    # 9. Contour Analysis
    # ==========================================
    from sklearn.neighbors import NearestNeighbors
    import torch.optim as optim

    # Toggle this to False if you want to skip optimization entirely
    RUN_ADVANCED_ANALYSIS = True

    print("\n🗺️ Generating Contour Analysis...")

    # 1. Setup Features & Grid
    top_2_idx = np.argsort(scores_tot)[-2:][::-1]
    f1_idx, f2_idx = top_2_idx[0], top_2_idx[1]
    f1_name, f2_name = feat_names[f1_idx], feat_names[f2_idx]
    other_indices = [i for i in range(n_features) if i not in [f1_idx, f2_idx]]

    grid_res = 30

    # x1_min, x1_max = X_train_denorm[:, f1_idx].min(), X_train_denorm[:, f1_idx].max()
    # x2_min, x2_max = X_train_denorm[:, f2_idx].min(), X_train_denorm[:, f2_idx].max()
    X_train_max_denorm = scaler_X.inverse_transform([[0.9] * ni])
    X_train_min_denorm = scaler_X.inverse_transform([[0.1] * ni])
    x1_max, x2_max = X_train_max_denorm[0, f1_idx], X_train_max_denorm[0, f2_idx]
    x1_min, x2_min = X_train_min_denorm[0, f1_idx], X_train_min_denorm[0, f2_idx]

    x1_lin = np.linspace(x1_min, x1_max, grid_res)
    x2_lin = np.linspace(x2_min, x2_max, grid_res)
    X1_mesh, X2_mesh = np.meshgrid(x1_lin, x2_lin)
    grid_coords_denorm = np.stack([X1_mesh.ravel(), X2_mesh.ravel()], axis=-1)

    # Helper: Denormalize inflection points for the overlay
    def get_denorm_ips(feat_idx):
        raw_ips = inflection_points_per_input[feat_idx]
        valid_ips = [ip for ip in raw_ips if ip is not None and 0.05 < ip < 0.95]
        if not valid_ips: return []
        dummy_x = np.zeros((len(valid_ips), n_features))
        dummy_x[:, feat_idx] = valid_ips
        return scaler_X.inverse_transform(dummy_x)[:, feat_idx]

    f1_ips = get_denorm_ips(f1_idx)
    f2_ips = get_denorm_ips(f2_idx)

    # --- MODE 1: Fixed at Mean (Always Run) ---
    mean_input_denorm = np.tile(np.mean(X_train_denorm, axis=0), (grid_res ** 2, 1))
    mean_input_denorm[:, f1_idx] = grid_coords_denorm[:, 0]
    mean_input_denorm[:, f2_idx] = grid_coords_denorm[:, 1]

    with torch.no_grad():
        in_norm = torch.tensor(scaler_X.transform(mean_input_denorm), dtype=torch.float32, device=device)
        Z_mean = scaler_y.inverse_transform(model(in_norm).cpu().numpy()).reshape(grid_res, grid_res)

    # --- Primary Plotting: Single Case (Always saved) ---
    fig, ax = plt.subplots(figsize=(5, 4))
    cp = ax.contourf(X1_mesh, X2_mesh, Z_mean, levels=30, cmap='RdYlBu_r', alpha=0.8)
    cbar = fig.colorbar(cp, ax=ax, label=name_y)
    cbar.set_label(name_y, fontsize=12)
    if data_on_contour:
        ax.scatter(X_train_denorm[:, f1_idx], X_train_denorm[:, f2_idx], c='black', s=8, alpha=0.2)
        ax.set_xlim([x1_min, x1_max])
        ax.set_ylim([x2_min, x2_max])

    for ip in f1_ips: ax.axvline(x=ip, color='green', linestyle='--', alpha=0.5, lw=2.5)
    for ip in f2_ips: ax.axhline(y=ip, color='green', linestyle='--', alpha=0.5, lw=2.5)

    # ax.set_title(f"KAN Prediction Surface (Mean-Fixed)\n{f1_name} vs {f2_name}")
    ax.set_xlabel(f1_name, fontsize=15)
    ax.set_ylabel(f2_name, fontsize=15)
    plt.tight_layout()
    single_plot_path = os.path.join(savepath, f"{data_name}_contour_mean_fixed.png")
    plt.savefig(single_plot_path, dpi=300)
    print(f"✅ Single contour saved to {single_plot_path}")

    # --- ADVANCED ANALYSIS (Modes 2 & 3) ---
    if RUN_ADVANCED_ANALYSIS:
        try:
            print("🚀 Starting Manifold and Optimization Analysis...")

            # --- MODE 2: Data Manifold ---
            nn = NearestNeighbors(n_neighbors=5).fit(X_train_denorm[:, [f1_idx, f2_idx]])
            _, nn_idx = nn.kneighbors(grid_coords_denorm)
            others_man = np.mean(X_train_denorm[nn_idx, :][:, :, other_indices], axis=1)

            man_input_denorm = np.zeros((grid_res ** 2, n_features))
            man_input_denorm[:, f1_idx] = grid_coords_denorm[:, 0]
            man_input_denorm[:, f2_idx] = grid_coords_denorm[:, 1]
            man_input_denorm[:, other_indices] = others_man

            with torch.no_grad():
                in_norm_man = torch.tensor(scaler_X.transform(man_input_denorm), dtype=torch.float32, device=device)
                Z_man = scaler_y.inverse_transform(model(in_norm_man).cpu().numpy()).reshape(grid_res, grid_res)

            # --- MODE 3: Optimization ---
            Z_opt_norm = np.zeros((grid_res, grid_res))
            X_med_norm = np.median(scaler_X.transform(X_train_denorm), axis=0)

            for i in range(grid_res):
                v1_n = (x1_lin - x1_min) / (x1_max - x1_min + 1e-9)
                v2_n = (x2_lin[i] - x2_min) / (x2_max - x2_min + 1e-9)
                x_others = torch.tensor(np.tile(X_med_norm[other_indices], (grid_res, 1)),
                                        dtype=torch.float32, device=device, requires_grad=True)

                opt = optim.LBFGS([x_others], lr=0.1, max_iter=20)

                def closure():
                    opt.zero_grad()
                    row_in = torch.zeros((grid_res, n_features), device=device)
                    row_in[:, f1_idx] = torch.tensor(v1_n, device=device).float()
                    row_in[:, f2_idx] = v2_n
                    row_in[:, other_indices] = x_others
                    loss = model(row_in).sum()
                    loss.backward()
                    return loss

                opt.step(closure)

                with torch.no_grad():
                    row_in = torch.zeros((grid_res, n_features), device=device)
                    row_in[:, f1_idx] = torch.tensor(v1_n, device=device).float()
                    row_in[:, f2_idx] = v2_n
                    row_in[:, other_indices] = x_others
                    Z_opt_norm[i, :] = model(row_in).cpu().numpy().flatten()

            Z_opt = scaler_y.inverse_transform(Z_opt_norm.reshape(-1, 1)).reshape(grid_res, grid_res)

            # --- Triple Plotting ---
            fig, axs = plt.subplots(1, 3, figsize=(24, 7))
            titles = ["Fixed at Mean", "Data Manifold (NN)", "Optimized (Minima)"]
            z_data_list = [Z_mean, Z_man, Z_opt]

            for ax_t, Z_plot, title in zip(axs, z_data_list, titles):
                cp = ax_t.contourf(X1_mesh, X2_mesh, Z_plot, levels=30, cmap='RdYlBu_r', alpha=0.8)
                fig.colorbar(cp, ax=ax_t, label=name_y)
                ax_t.scatter(X_train_denorm[:, f1_idx], X_train_denorm[:, f2_idx], c='black', s=8, alpha=0.2)
                for ip in f1_ips: ax_t.axvline(x=ip, color='green', linestyle='--', alpha=0.5, lw=1.2)
                for ip in f2_ips: ax_t.axhline(y=ip, color='green', linestyle='--', alpha=0.5, lw=1.2)
                ax_t.set_title(f"{title}\n{f1_name} vs {f2_name}")

            plt.tight_layout()
            plt.savefig(os.path.join(savepath, f"{data_name}_triple_contour_analysis.png"), dpi=300)
            print("✅ Triple contour analysis complete.")

        except Exception as e:
            print(f"⚠️ Advanced analysis failed or timed out: {e}")
            print("⏭️ Skipping to next step with Mode 1 results only.")
    # ==========================================
    # [NEW] Save Denormalized IPs to CSV
    # ==========================================
    ip_records = []

    # Add IPs for Feature 1
    for ip in f1_ips:
        ip_records.append({'Feature': f1_name, 'Inflection_Point': ip, 'Type': 'Primary (f1)'})

    # Add IPs for Feature 2
    for ip in f2_ips:
        ip_records.append({'Feature': f2_name, 'Inflection_Point': ip, 'Type': 'Secondary (f2)'})

    # Create DataFrame and Save
    if ip_records:
        df_ips = pd.DataFrame(ip_records)
        ip_csv_path = os.path.join(savepath, f"{data_name}_inflection_points_denorm.csv")
        df_ips.to_csv(ip_csv_path, index=False)
        print(f"📄 Denormalized inflection points saved to: {ip_csv_path}")
    else:
        print("ℹ️ No valid inflection points found within the 0.05-0.95 range to save.")

    # ==========================================
    # 10. Range-Based Attribution Heatmap (Fixed)
    # ==========================================
    print(f"\n🎨 Generating Range-Based Attribution Heatmap for feature: {feat_names[top_2_idx[0]]}")

    target_feat_idx = top_2_idx[0]
    target_feat_name = feat_names[target_feat_idx]

    # Extract attribution scores for the top feature from each interval
    region_attr_values = [scores[target_feat_idx] for scores in scores_interval_norm]

    # We need to classify each grid point into one of your 'final_masks'
    # 1. Prepare the grid in normalized space (same as the model sees)
    grid_coords_norm = scaler_X.transform(mean_input_denorm)
    grid_coords_norm_torch = torch.tensor(grid_coords_norm, dtype=torch.float32, device=device)

    # 2. Initialize the heatmap array
    Z_attr_range_flat = np.zeros(grid_res ** 2)

    # 3. Apply the logical ranges to the grid points
    # We iterate through the logic that created final_masks
    for i, label in enumerate(final_labels):
        # 'label' looks like '0.10<x1<0.45 & 0.20<x2<0.80'
        # We parse the label to apply the same logic to our grid
        # Or, more simply, we repeat the masking logic on our grid_coords_norm_torch

        mask_grid = torch.ones(grid_res ** 2, dtype=torch.bool, device=device)

        # Re-apply the selection criteria for each selected feature
        for feat_d in selected_features_data:
            idx = feat_d['index']
            x_col = grid_coords_norm_torch[:, idx]

            # Identify which specific interval for this feature is mentioned in the label
            # This part ensures we match the grid point to the correct region label
            for sub_mask, sub_label in zip(feat_d['masks'], feat_d['labels']):
                if sub_label in label:
                    # Extract the bounds from the label string (e.g., '0.10<x1<0.45')
                    parts = sub_label.split('<')
                    lb = float(parts[0])
                    ub = float(parts[2])
                    mask_grid = mask_grid & (x_col > lb) & (x_col <= ub)

        if torch.any(mask_grid):
            Z_attr_range_flat[mask_grid.cpu().numpy()] = region_attr_values[i]

    Z_attr_range = Z_attr_range_flat.reshape(grid_res, grid_res)

    # 4. Plotting
    fig_range, ax_range = plt.subplots(figsize=(10, 8))

    # Use pcolormesh for the discrete "tiled" look
    im = ax_range.pcolormesh(X1_mesh, X2_mesh, Z_attr_range, cmap='YlOrRd', shading='auto', alpha=0.8)

    # Overlay original data points
    if data_on_contour:
        ax_range.scatter(X_train_denorm[:, f1_idx], X_train_denorm[:, f2_idx], c='black', s=8, alpha=0.2)
        ax_range.set_xlim([x1_min, x1_max])
        ax_range.set_ylim([x2_min, x2_max])

    # Overlay inflection boundaries
    for ip in f1_ips: ax_range.axvline(x=ip, color='blue', linestyle='--', alpha=0.4, lw=1.5)
    for ip in f2_ips: ax_range.axhline(y=ip, color='blue', linestyle='--', alpha=0.4, lw=1.5)

    plt.colorbar(im, ax=ax_range, label=f'Attribution Score of {target_feat_name}')
    ax_range.set_title(f"Range-Based Attribution Map: {target_feat_name}\n(Tiled by combined feature ranges)")
    ax_range.set_xlabel(f1_name)
    ax_range.set_ylabel(f2_name)

    range_heatmap_path = os.path.join(savepath, f"{data_name}_range_attribution_heatmap.png")
    plt.tight_layout()
    plt.savefig(range_heatmap_path, dpi=300)
    #
    # # ==========================================
    # # 11. Global Top-2 Dominance Map (FIXED COORDINATES)
    # # ==========================================
    # print("\n⚖️ Generating Global Top-2 Dominance Map with aligned coordinates...")
    #
    # # 1. Identify Global Ranks
    # global_ranks = np.argsort(scores_tot)[::-1]
    # g1_idx, g2_idx = global_ranks[0], global_ranks[1]
    # g1_name, g2_name = feat_names[g1_idx], feat_names[g2_idx]
    #
    # # 2. Re-calculate deltas per region
    # dominance_deltas_per_region = []
    # for scores in scores_interval_norm:
    #     dominance_deltas_per_region.append(scores[g1_idx] - scores[g2_idx])
    #
    # # 3. Map to Grid
    # Z_dominance_flat = np.zeros(grid_res ** 2)
    # # We need the normalized grid to match the interval logic (0.0 to 1.0)
    # grid_coords_norm_torch = torch.tensor(scaler_X.transform(man_input_denorm),
    #                                       dtype=torch.float32, device=device)
    #
    # for i, label in enumerate(final_labels):
    #     mask_grid = torch.ones(grid_res ** 2, dtype=torch.bool, device=device)
    #     for feat_d in selected_features_data:
    #         idx = feat_d['index']
    #         x_col = grid_coords_norm_torch[:, idx]
    #         for sub_label in feat_d['labels']:
    #             if sub_label in label:
    #                 parts = sub_label.split('<')
    #                 lb, ub = float(parts[0]), float(parts[2])
    #                 mask_grid = mask_grid & (x_col > lb) & (x_col <= ub)
    #     if torch.any(mask_grid):
    #         Z_dominance_flat[mask_grid.cpu().numpy()] = dominance_deltas_per_region[i]
    #
    # Z_dominance = Z_dominance_flat.reshape(grid_res, grid_res)
    #
    # # 4. Plotting with Alignment Fix
    # fig_dom, ax_dom = plt.subplots(figsize=(12, 8))
    #
    # # [FIX] Use X1_mesh and X2_mesh explicitly to ensure pcolormesh anchors
    # # to the correct denormalized coordinates.
    # limit = np.max(np.abs(Z_dominance))
    # im = ax_dom.pcolormesh(X1_mesh, X2_mesh, Z_dominance,
    #                        cmap='RdBu_r', vmin=-limit, vmax=limit,
    #                        shading='nearest', alpha=0.8)  # 'nearest' helps alignment
    #
    # cbar = plt.colorbar(im, ax=ax_dom)
    # cbar.set_label(f'Score Difference ({g1_name} - {g2_name})', rotation=270, labelpad=20)
    #
    # # Overlay points
    # ax_dom.scatter(X_train_denorm[:, f1_idx], X_train_denorm[:, f2_idx],
    #                c='black', s=10, alpha=0.4, label='Data Points')
    #
    # # [FIX] Re-draw boundaries using the denormalized inflection points
    # # Ensure these lines use the same scale as X1_mesh/X2_mesh
    # for ip in f1_ips:
    #     ax_dom.axvline(x=ip, color='black', linestyle='--', alpha=0.6, lw=1.5)
    # for ip in f2_ips:
    #     ax_dom.axhline(y=ip, color='black', linestyle='--', alpha=0.6, lw=1.5)
    #
    # ax_dom.set_title(f"Global Top-2 Dominance Map\nRed: {g1_name} Dominates | Blue: {g2_name} Dominates")
    # ax_dom.set_xlabel(f1_name)
    # ax_dom.set_ylabel(f2_name)
    #
    # # Ensure limits match the data range exactly to prevent "drifting" visuals
    # ax_dom.set_xlim(x1_min, x1_max)
    # ax_dom.set_ylim(x2_min, x2_max)
    #
    # plt.tight_layout()
    # dom_map_path = os.path.join(savepath, f"{data_name}_global_dominance_map_aligned.png")
    # plt.savefig(dom_map_path, dpi=300)
    # plt.show()

    # ==========================================
    # 11. Global Top-2 Log-Ratio Map (log10(Rank 1 / Rank 2))
    # ==========================================
    print("\n⚖️ Generating Global Top-2 Log-Ratio Map...")

    # 1. Identify Global Ranks
    global_ranks = np.argsort(scores_tot)[::-1]
    g1_idx, g2_idx = global_ranks[0], global_ranks[1]
    g1_name, g2_name = feat_names[g1_idx], feat_names[g2_idx]

    # 2. Calculate Log-Ratio per region
    log_ratio_per_region = []
    for scores in scores_interval_norm:
        s1, s2 = scores[g1_idx], scores[g2_idx]
        # log10(s1/s2) = log10(s1) - log10(s2)
        # Adding small epsilon to avoid log(0)
        val = np.log10((s1 + 1e-9) / (s2 + 1e-9))
        log_ratio_per_region.append(val)

    # 3. Map to Grid
    Z_log_ratio_flat = np.zeros(grid_res ** 2)
    grid_coords_norm_torch = torch.tensor(scaler_X.transform(mean_input_denorm),
                                          dtype=torch.float32, device=device)

    for i, label in enumerate(final_labels):
        mask_grid = torch.ones(grid_res ** 2, dtype=torch.bool, device=device)
        for feat_d in selected_features_data:
            idx = feat_d['index']
            x_col = grid_coords_norm_torch[:, idx]
            for sub_label in feat_d['labels']:
                if sub_label in label:
                    parts = sub_label.split('<')
                    lb, ub = float(parts[0]), float(parts[2])
                    mask_grid = mask_grid & (x_col > lb) & (x_col <= ub)
        if torch.any(mask_grid):
            Z_log_ratio_flat[mask_grid.cpu().numpy()] = log_ratio_per_region[i]

    Z_log_ratio = Z_log_ratio_flat.reshape(grid_res, grid_res)

    # 4. Plotting
    fig_log, ax_log = plt.subplots(figsize=(5, 4))

    # Use a diverging colormap (RdBu_r) centered at 0
    limit = np.max(np.abs(Z_log_ratio))
    im = ax_log.pcolormesh(X1_mesh, X2_mesh, Z_log_ratio,
                           cmap='RdBu_r', vmin=-limit, vmax=limit,
                           shading='nearest', alpha=0.8)

    cbar = plt.colorbar(im, ax=ax_log)
    cbar.set_label(f'$\log_{{10}}$({g1_name} / {g2_name})', rotation=270, labelpad=20)

    # Overlay points and boundaries
    if data_on_contour:
        ax_log.scatter(X_train_denorm[:, f1_idx], X_train_denorm[:, f2_idx],
                   c='black', s=10, alpha=0.3, label='Data Points')
        ax_log.set_xlim([x1_min, x1_max])
        ax_log.set_ylim([x2_min, x2_max])

    for ip in f1_ips:
        ax_log.axvline(x=ip, color='black', linestyle='--', alpha=0.4, lw=2.5)
    for ip in f2_ips:
        ax_log.axhline(y=ip, color='black', linestyle='--', alpha=0.4, lw=2.5)

    ax_log.set_title(f"Red: {g1_name} | Blue: {g2_name}")
    ax_log.set_xlabel(f1_name, fontsize=15)
    ax_log.set_ylabel(f2_name, fontsize=15)

    # Fix boundaries
    ax_log.set_xlim(x1_min, x1_max)
    ax_log.set_ylim(x2_min, x2_max)

    plt.tight_layout()
    log_ratio_path = os.path.join(savepath, f"{data_name}_log_ratio_map.png")
    plt.savefig(log_ratio_path, dpi=300)


if __name__ == "__main__":
    main()