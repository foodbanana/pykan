import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams

fs = 10
dpi = 75
config_figure = {'figure.figsize': (3, 2.5), 'figure.titlesize': fs,
                 'font.size': fs, 'font.family': 'sans-serif', 'font.serif': ['computer modern roman'],
                 # 'font.sans-serif': ['Helvetica Neue LT Pro'],  # Avenir LT Std, Helvetica Neue LT Pro, Helvetica LT Std
                 'font.weight': '300', 'axes.titleweight': '400', 'axes.labelweight': '300',
                 'axes.xmargin': 0, 'axes.titlesize': fs, 'axes.labelsize': fs, 'axes.labelpad': 2,
                 'xtick.labelsize': fs-2, 'ytick.labelsize': fs-2, 'xtick.major.pad': 0, 'ytick.major.pad': 0,
                 'legend.fontsize': fs-2, 'legend.title_fontsize': fs, 'legend.frameon': False,
                 'legend.labelspacing': 0.5, 'legend.columnspacing': 0.5, 'legend.handletextpad': 0.2,
                 'lines.linewidth': 1, 'hatch.linewidth': 0.5, 'hatch.color': 'w',
                 'figure.subplot.left': 0.15, 'figure.subplot.right': 0.93,
                 'figure.subplot.top': 0.95, 'figure.subplot.bottom': 0.15,
                 'figure.dpi': dpi, 'savefig.dpi': dpi*10, 'savefig.transparent': False,  # change here True if you want transparent background
                 'text.usetex': False, 'mathtext.default': 'regular',
                 'text.latex.preamble': r'\usepackage{amsmath,amssymb,bm,physics,lmodern,cmbright}'}
rcParams.update(config_figure)

def remove_outliers_iqr(df_in, df_out, rr=6):

    combined_df = pd.concat([df_in, df_out],
                            axis=1)

    numeric_cols = combined_df.select_dtypes(
        include=np.number).columns

    Q1 = combined_df[numeric_cols].quantile(0.25)
    Q3 = combined_df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1  # IQR은 대략 상위 25% - 상위75% = 중간정도의 값에 해당

    lower_bound = Q1 - rr * IQR  # 보통은 1.5* IQR을 진행하지만 최대한 삭제되는 데이터가 적도록 진행
    upper_bound = Q3 + rr * IQR

    condition = ~((combined_df[numeric_cols] < lower_bound) | (combined_df[numeric_cols] > upper_bound)).any(axis=1)

    df_in_no_outliers = df_in[condition]
    df_out_no_outliers = df_out[condition]

    return df_in_no_outliers, df_out_no_outliers


def evaluate_model_performance(model, dataset, scaler_y, phase="validation", display=False):  # phase = validation 아니면 test 이다

    if phase == "validation":
        input_tensor = dataset['val_input']
        label_tensor = dataset['val_label']
    elif phase == "test":
        input_tensor = dataset['test_input']
        label_tensor = dataset['test_label']
    else:
        raise ValueError("phase는 'validation' 또는 'test'만 가능합니다")

    # 예측 수행
    with torch.no_grad():  # 굳이 기울기 계산할 필요 X because 이거는 test 이기에 학습 X --- 시간 더 빠르게 하려고 torch.no_grad()
        pred_norm = model(input_tensor)  # input_tensor 는 val_inut or test_input / pred_norm은 그에 대한 출력값

    # 역정규화
    pred_real = scaler_y.inverse_transform(
        pred_norm.cpu().detach().numpy())  # pred_real 은 0.1~0.9 사이의 입력 val_input or test_input을 받고 출력된 값은 다시 역정규화 한 실제 출력값
    label_real = scaler_y.inverse_transform(
        label_tensor.cpu().detach().numpy())  # label_real은 0.1~0.9 사이의 입력 val_label or test_label을 받고 출력한 값 역정규화
    # inverse_transform 함수는  numpy 를 입력으로 받기 떄문에 pytorch tensor를 cpu로 옮기고 numpy로 변환

    # numpy는 cpu에서만 돌아가니까 tensor를 .cpu로 옮기고 그다음 tensor의 추가정보 (numpy 정보 + 어떻게 계산되었는지 식에 대한 정보)를 detach --- 그 다음에 .numpy()를 통해 numpy로 변환

    # 성능 지표 계산 from 역정규화된 label_real, pred_real 값들 from val input or test input + numpy에서 계산
    rmse = np.sqrt(mean_squared_error(label_real, pred_real))  # 오차 제곱 평균의 루트
    r2 = r2_score(label_real, pred_real)  # 1에 가까울수록 좋다
    mae = np.mean(np.abs(label_real - pred_real))  # 오차 절댓값들의 평균 -- MAE

    if display:
        print(f"{phase} SET Performance Evaluation")  # phase = validation 또는 test
        print(f"RMSE: {rmse:.4f}")  # f"{변수:포맷코드}"
        print(f"R²: {r2:.4f}")
        print(f"MAE: {mae:.4f}")

        print(f"실제값 평균: {label_real.mean():.4f}")  # label_real의 평균값(실제값)
        print(f"예측값 평균: {pred_real.mean():.4f}")  # pred_real의 평균값(KAN 모델로 예측한 값)

    return pred_real, label_real, {'rmse': rmse, 'r2': r2, 'mae': mae}

def plot_data_per_interval(X_norm, y_norm, name_X, name_y, mask_idx, mask_interval, ymax=1.):
    nx = X_norm.shape[1]
    fig, axs = plt.subplots(nrows=1, ncols=nx, figsize=(20, 3.5), constrained_layout=True)
    axs = np.atleast_1d(axs)
    for idx_x in range(nx):
        ax = axs[idx_x]
        ax.scatter(X_norm[:, idx_x], y_norm, color='tab:gray')
        ax.set_title(name_X[idx_x], fontsize=8)

    x_mask = X_norm[:, mask_idx]
    y_vals = y_norm.ravel()  # y가 (N,1)이어도 (N,)으로 평탄화

    # mask_interval = [0, 0.3, 0.6, 1.0]
    masks = [ ((x_mask > lb) & (x_mask <= ub)) for lb, ub in zip(mask_interval, mask_interval[1:] + [ymax])]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    labels = [f'{lb} < x{mask_idx} <= {ub}' for lb, ub in zip(mask_interval, mask_interval[1:] + [ymax])]

    for mask, c, lab in zip(masks, colors, labels):
        if np.any(mask):  # 해당 구간 데이터가 있을 때만 그림
            for idx_x in range(nx):
                ax = axs[idx_x]
                ax.scatter(X_norm[mask, idx_x], y_vals[mask], s=20, color=c, alpha=0.75, edgecolor='none', label=lab)
                ax.set_title(name_X[idx_x], fontsize=8)
    axs[0].set_ylabel(name_y, fontsize=8)
    axs[-1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    # Use constrained_layout above to manage spacing automatically
    # fig.tight_layout()  # not needed when constrained_layout=True
    return fig, axs


def plot_activation_functions(model, x=None, layers=None, show=True, titles=True, save_heading=None):
    """
    Plot per-edge activation functions (input->output splines) of a trained (and optionally pruned) KAN/MultKAN model.

    Parameters
    - model: trained KAN/MultKAN model
    - x: input tensor/ndarray or dataset dict (uses ['train_input']). If None, uses model.cache_data if available.
    - layers: list of layer indices to plot. If None, plots all layers.
    - show: whether to call plt.show() for each figure
    - titles: add small titles for each subplot indicating (in_idx -> out_idx)

    Returns
    - figs: list of matplotlib Figure objects created (one per layer)
    """
    # Ensure activations are cached
    if isinstance(x, dict):
        x_use = x.get('train_input', None)
    else:
        x_use = x
    try:
        model.get_act(x_use)
    except Exception as e:
        # Try again using cached data if available
        if getattr(model, 'cache_data', None) is not None and x_use is None:
            model.get_act(model.cache_data)
        else:
            raise e

    figs = []
    depth = len(model.act_fun)
    layers_to_plot = list(range(depth)) if layers is None else layers

    for l in layers_to_plot:
        act = model.act_fun[l]
        ni, no = act.coef.shape[:2]
        # Dynamic figure size and constrained layout to avoid overlaps
        fig, axs = plt.subplots(nrows=no, ncols=ni, squeeze=False,
                                figsize=(max(2.5*ni, 6), max(2.5*no, 3.5)),
                                constrained_layout=True)
        for i in range(ni):
            for j in range(no):
                ax = axs[j, i]
                # Gather pre- and post- activations and sort by input
                inputs = model.spline_preacts[l][:, j, i].cpu().detach().numpy()
                outputs = model.spline_postacts[l][:, j, i].cpu().detach().numpy()
                rank = np.argsort(inputs)
                ax.plot(inputs[rank], outputs[rank], marker='o', ms=2, lw=1)
                if titles:
                    ax.set_title(f'in {i} → out {j}', fontsize=8)
        # Add a legend-like layer title in the last subplot
        axs[-1, -1].text(0.99, 0.01, f'Layer {l}', transform=axs[-1, -1].transAxes,
                         ha='right', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        if save_heading is not None:
            plt.savefig(f'{save_heading}_act_L{l}.png')
        if show:
            plt.show()
        figs.append(fig)
    return figs


def get_masks(X, mask_idx, mask_interval):
    x_mask = X[:, mask_idx]
    masks = [ ((x_mask > lb) & (x_mask <= ub)) for lb, ub in zip(mask_interval[:-1], mask_interval[1:])]
    return masks


def plot_spline_coefficients(model, save_heading=None, show=True):
    plots = []
    for layer, act_fun in enumerate(model.act_fun):
        ni, no = act_fun.coef.shape[:2]
        coef = act_fun.coef.tolist()
        # Dynamically size figure and use constrained layout to prevent overlaps
        fig, axs = plt.subplots(nrows=no, ncols=ni, figsize=(max(2.5*ni, 6), max(2.5*no, 3.5)),
                                squeeze=False, constrained_layout=True)
        for idx_in, coef_in in enumerate(coef):
            for idx_out, coef_node in enumerate(coef_in):
                ax = axs[idx_out, idx_in]
                ax.scatter(np.linspace(0.1, 0.9, (len(coef_node))), coef_node, label='Coefficients')
                slope = [x - y for x, y in zip(coef_node[1:], coef_node[:-1])]
                ax.bar(np.linspace(0.1, 0.9, len(slope)), slope, width=0.02, align='edge', label='Slope')
                ax.set_title(f'In {idx_in} -- Out {idx_out}', fontsize=10)
        axs[-1, -1].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, title=f'Layer {layer}')
        if save_heading is not None:
            plt.savefig(f'{save_heading}_spline_coef_L{layer}.png')
        if show:
            plt.show()
        plots.append((fig, axs))
    return plots

def plot_activation_and_spline_coefficients(
        model, x=None, layers=None, save_heading=None, show=True, titles=True
):
    # Ensure activations are cached
    if isinstance(x, dict):
        x_use = x.get('train_input', None)
    else:
        x_use = x
    try:
        model.get_act(x_use)
    except Exception as e:
        # Try again using cached data if available
        if getattr(model, 'cache_data', None) is not None and x_use is None:
            model.get_act(model.cache_data)
        else:
            raise e

    figs = []
    depth = len(model.act_fun)
    layers_to_plot = list(range(depth)) if layers is None else layers

    for l in layers_to_plot:
        act = model.act_fun[l]
        ni, no = act.coef.shape[:2]
        coef = act.coef.tolist()

        # Dynamic figure size and constrained layout to avoid overlaps
        fig, axs = plt.subplots(nrows=no, ncols=ni, squeeze=False,
                                figsize=(max(2.5*ni, 6), max(2.5*no, 3.5)),
                                constrained_layout=True)
        second_axs = np.zeros_like(axs)
        for i in range(ni):
            for j in range(no):
                ax = axs[j, i]
                # Gather pre- and post-activations and sort by input
                inputs = model.spline_preacts[l][:, j, i].cpu().detach().numpy()
                outputs = model.spline_postacts[l][:, j, i].cpu().detach().numpy()
                coef_node = coef[i][j]
                num_knot = act.grid.shape[1]
                spline_radius = int((num_knot - len(coef_node)) / 2)
                bar_width = min(act.grid[i, 1:] - act.grid[i, :-1]) / 2

                rank = np.argsort(inputs)
                ax.plot(inputs[rank], outputs[rank], marker='o', ms=2, lw=1, label='Activations')

                ax2 = ax.twinx()
                second_axs[j, i] = ax2
                ax2.scatter(act.grid[i, spline_radius:-spline_radius], coef_node,
                           s=20, color='white', edgecolor='k', label='Coefficients')
                slope = [x - y for x, y in zip(coef_node[1:], coef_node[:-1])]
                bars = ax2.bar(act.grid[i, spline_radius:-(spline_radius + 1)], slope,
                        width=bar_width, align='edge', color='r', label='Slope')
                # ax2.bar_label(bars, fmt='%.2f', fontsize=6, padding=3)
                if titles:
                    ax.set_title(f'in {i} -- out {j}', fontsize=10)
        second_axs[-1, -1].legend(loc='best', fontsize=8, title=f'---Layer {l}---', title_fontsize=8)

        if save_heading is not None:
            plt.savefig(f'{save_heading}_act_coef_L{l}.png')
        if show:
            plt.show()
        figs.append(fig)
    return figs

def get_spline_coef_and_slope(model, x=None):
    # Ensure activations are cached
    if isinstance(x, dict):
        x_use = x.get('train_input', None)
    else:
        x_use = x
    try:
        model.get_act(x_use)
    except Exception as e:
        # Try again using cached data if available
        if getattr(model, 'cache_data', None) is not None and x_use is None:
            model.get_act(model.cache_data)
        else:
            raise e

    coefs = []
    slopes = []
    depth = len(model.act_fun)
    layers_to_plot = list(range(depth))

    for l in layers_to_plot:
        act = model.act_fun[l]
        ni, no = act.coef.shape[:2]
        coef = act.coef.tolist()

        for i in range(ni):
            coef_layer = []
            slope_layer = []
            for j in range(no):
                coef_node = coef[i][j]
                slope = [x - y for x, y in zip(coef_node[1:], coef_node[:-1])]
                coef_layer.append(coef_node)
                slope_layer.append(slope)
            coefs.append(coef_layer)
            slopes.append(slope_layer)

    return coefs, slopes

def find_index_sign_revert(data_list):
    """
    Finds the first index where the sign changes and persists for
    more than one element.

    Args:
        data_list (list): A list of numbers.

    Returns:
        int or None: The first index of a persistent sign change,
                     or None if no such change is found.
    """
    if len(data_list) < 3:
        # Not possible to have a persistent change
        return None

    signs = np.sign(data_list)

    for i in range(1, len(signs) - 1):
        if signs[i] != signs[i-1]:
            if signs[i+1] == signs[i]:
                return i

    return None
