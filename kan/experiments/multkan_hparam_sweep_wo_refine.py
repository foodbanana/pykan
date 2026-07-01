import itertools
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Custom MultKAN written by Dr. DDP import path within this repository
from kan.custom_multkan_ddp import MultKAN
from sklearn.metrics import mean_squared_error, r2_score
import datetime
auto_save_path = f"multkan_hparam_sweep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

# import colorcet as cc  # pip install colorcet
from matplotlib import colors, rcParams, cm
import time

fs = 10
dpi = 200
config_figure = {'figure.figsize': (3, 2.5), 'figure.titlesize': fs,
                 'font.size': fs, 'font.family': 'sans-serif', 'font.serif': ['computer modern roman'],
                 'font.sans-serif': ['Helvetica Neue LT Pro'],  # Avenir LT Std, Helvetica Neue LT Pro, Helvetica LT Std
                 'font.weight': '300', 'axes.titleweight': '400', 'axes.labelweight': '300',
                 'axes.xmargin': 0, 'axes.titlesize': fs, 'axes.labelsize': fs, 'axes.labelpad': 2,
                 'xtick.labelsize': fs-2, 'ytick.labelsize': fs-2, 'xtick.major.pad': 0, 'ytick.major.pad': 0,
                 'legend.fontsize': fs-2, 'legend.title_fontsize': fs, 'legend.frameon': False,
                 'legend.labelspacing': 0.5, 'legend.columnspacing': 0.5, 'legend.handletextpad': 0.2,
                 'lines.linewidth': 1, 'hatch.linewidth': 0.5, 'hatch.color': 'w',
                 'figure.subplot.left': 0.15, 'figure.subplot.right': 0.93,
                 'figure.subplot.top': 0.95, 'figure.subplot.bottom': 0.15,
                 'figure.dpi': dpi, 'savefig.dpi': dpi*5, 'savefig.transparent': False,  # change here True if you want transparent background
                 'text.usetex': False, 'mathtext.default': 'regular',
                 'text.latex.preamble': r'\usepackage{amsmath,amssymb,bm,physics,lmodern,cmbright}'}
rcParams.update(config_figure)

def _seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic for reproducibility across processes
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrialResult:
    params: Dict[str, Any]
    val_loss: float
    train_loss: float
    test_loss: Optional[float]
    r2_train: Optional[float]
    r2_val: Optional[float]
    r2_test: Optional[float]
    seed: int
    device: str
    spline_train_loss: Optional[float] = None
    spline_test_loss: Optional[float] = None

class PruningError(Exception):
    """모델 Pruning 과정에서 에러가 발생했을 때 사용합니다."""
    pass

class SymbolificationError(Exception):
    """모델 Symbolic 변환 과정에서 에러가 발생했을 때 사용합니다."""
    pass

def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def _build_dataset(X_train, y_train, X_val, y_val, X_test=None, y_test=None, device: Optional[torch.device] = None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = {
        'train_input': _to_tensor(X_train, device),
        'train_label': _to_tensor(y_train, device),
        'val_input': _to_tensor(X_val, device),
        'val_label': _to_tensor(y_val, device),
    }
    if X_test is not None and y_test is not None:
        dataset['test_input'] = _to_tensor(X_test, device)
        dataset['test_label'] = _to_tensor(y_test, device)
    return dataset


def mae_and_r2(model: MultKAN, xk: torch.Tensor, yk: torch.Tensor, scaler_y: Optional[Any] = None)\
        -> Tuple[torch.Tensor, torch.Tensor, float, Optional[float]]:
    yhat = model(xk)
    y_true = yk.detach().cpu().numpy()
    y_pred = yhat.detach().cpu().numpy()
    # Optionally inverse-transform to the original scale
    if scaler_y is not None:
        try:
            y_true = scaler_y.inverse_transform(y_true)
            y_pred = scaler_y.inverse_transform(y_pred)
        except Exception:
            pass
    # mean absolute error (scalar)
    mae = np.mean(np.abs(y_true - y_pred))
    # r2_score may raise warnings/errors for constant targets in some versions; guard defensively
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = float('nan')

    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # plt.scatter(y_true, y_pred, color='k')
    # plt.scatter(y_true, y_true, color='red')
    # plt.show()

    return y_true, y_pred, float(mae), r2


def _evaluate(model: MultKAN, dataset: Dict[str, torch.Tensor], scaler_y: Optional[Any] = None) \
        -> Tuple[float, float, Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Evaluate using sklearn.metrics.mean_squared_error and r2_score on CPU numpy arrays.

    If scaler_y is provided (a fitted sklearn-like scaler with inverse_transform),
    y_true and y_pred are inverse-transformed back to the original scale before computing metrics.
    """
    with torch.no_grad():
        _, _, mae_train, r2_train = mae_and_r2(model, dataset['train_input'], dataset['train_label'], scaler_y=scaler_y)
        _, _, mae_val, r2_val = mae_and_r2(model, dataset['val_input'], dataset['val_label'], scaler_y=scaler_y)
        mae_test, r2_test = None, None
        if 'test_input' in dataset and 'test_label' in dataset:
            _, _, mae_test, r2_test = mae_and_r2(model, dataset['test_input'], dataset['test_label'], scaler_y=scaler_y)
    return mae_train, mae_val, mae_test, r2_train, r2_val, r2_test


def _run_single_trial(args, verbose=False) -> Tuple[TrialResult, MultKAN, Dict[str, Any], Dict[str, Any]]:
    X_train, y_train, X_val, y_val, X_test, y_test, params, device_str, scaler_y, seed = args
    device = torch.device(device_str)
    _seed_everything(seed)

    dataset = _build_dataset(X_train, y_train, X_val, y_val, X_test, y_test, device=device)

    # Separate model constructor kwargs from fit kwargs
    model_kwargs = {k: params[k] for k in ['width', 'grid', 'grid_eps', 'k', 'mult_arity', 'seed', 'device'] if k in params}
    # Override device/seed per trial
    model_kwargs['device'] = device
    model_kwargs['seed'] = seed
    model_kwargs['grid_range'] = params.get('grid_range', [0.1, 0.9])

    model = MultKAN(**model_kwargs)

    fit_kwargs = {
        'opt': params.get('opt', 'LBFGS'),
        'steps': params.get('steps', 50),
        'stop_grid_update_step': params.get('stop_grid_update_step', 50),
        'lamb': params.get('lamb', 0.0),
        'lamb_l1': params.get('lamb_l1', 1.0),
        'lamb_entropy': params.get('lamb_entropy', 2.0),
        'lamb_coef': params.get('lamb_coef', 0.0),
        'lamb_coefdiff': params.get('lamb_coefdiff', 0.0),
        'update_grid': params.get('update_grid', True),
        'lr': params.get('lr', 1.0),
        'batch': params.get('batch', -1),
        'log': params.get('log', 1),
    }

    res_spline = model.fit(dataset, **fit_kwargs)
    if verbose:
        model.plot()
        plt.show()

    # Optional pruning to mimic notebook behavior
    def _want_prune(p: Dict[str, Any]) -> bool:
        val = p.get('prune', None)
        if isinstance(val, str):
            v = val.strip().lower()
            return v in ('1', 'true', 'yes', 'y', 't')
        return bool(val)

    def _want_symbolic(p: Dict[str, Any]) -> bool:
        val = p.get('symbolic', False)
        if isinstance(val, str):
            v = val.strip().lower()
            return v in ('1', 'true', 'yes', 'y', 't')
        return bool(val)

    if _want_prune(params):
        # Unified pruning threshold handling: if 'pruning_th' is provided, use it for both node_th and edge_th
        pruning_th = params.get('pruning_th', 1e-2)
        node_th = params.get('pruning_node_th', pruning_th)
        edge_th = params.get('pruning_edge_th', pruning_th)
        try:
            model = model.prune(node_th=node_th, edge_th=edge_th)
        except Exception as e:
            raise PruningError(f"Failed to prune model with node_th={node_th}, edge_th={edge_th}") from e

    if _want_symbolic(params):
        lib = ['sin', 'cos', 'x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', '1/x', '1/x^2']
        sym_weight_simple = params.get('sym_weight_simple', 0.8)
        sym_r2_threshold = params.get('sym_r2_threshold', 0.)
        sym_a_range = params.get('sym_a_range', (-10, 10))
        sym_b_range = params.get('sym_b_range', (-10, 10))
        try:
            model.auto_symbolic(lib=lib, weight_simple=sym_weight_simple, r2_threshold=sym_r2_threshold,
                                verbose=verbose, a_range=sym_a_range, b_range=sym_b_range)
            model.fit(dataset, **fit_kwargs)

            if verbose:
                model.plot()
                plt.show()

        except Exception as e:
            raise SymbolificationError(f"Failed during symbolification or subsequent fitting/plotting") from e

    mae_train, mae_val, mae_test, r2_train, r2_val, r2_test = _evaluate(model, dataset, scaler_y=scaler_y)

    return TrialResult(
        params=params,
        val_loss=mae_val,
        train_loss=mae_train,
        test_loss=mae_test,
        r2_train=r2_train,
        r2_val=r2_val,
        r2_test=r2_test,
        seed=seed,
        device=str(device),
        spline_train_loss=float(res_spline['train_loss'][0]),
        spline_test_loss=float(res_spline['test_loss'][0]),
    ), model, fit_kwargs, dataset


def _expand_param_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    values = [param_grid[k] if isinstance(param_grid[k], (list, tuple)) else [param_grid[k]] for k in keys]
    combos = []
    for vs in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, vs)})
    return combos


def _trial_to_row(trial: Any) -> Dict[str, Any]:
    """Flatten a TrialResult (or dict) to a single row dict suitable for DataFrame."""
    if hasattr(trial, '__dict__') and isinstance(trial, TrialResult):
        d = asdict(trial)
    elif isinstance(trial, dict):
        d = dict(trial)
    else:
        # Fallback: try dataclasses.asdict or direct cast
        try:
            d = asdict(trial)
        except Exception:
            d = dict(trial)
    params = d.pop('params', {}) or {}
    row = {}
    # Copy top-level scalar fields
    for k, v in d.items():
        row[k] = v
    # Flatten params into columns prefixed by 'param_'
    for pk, pv in params.items():
        # Lists/dicts to JSON strings for Excel cell
        if isinstance(pv, (list, dict, tuple)):
            row[f'param_{pk}'] = json.dumps(pv)
        else:
            row[f'param_{pk}'] = pv
    return row


def _results_to_dataframe(results: List[Any]) -> pd.DataFrame:
    rows = [_trial_to_row(r) for r in results]
    df = pd.DataFrame(rows)
    # Order columns: metrics (losses and R^2), seed/device first, then params
    metric_cols = [c for c in ['r2_train', 'r2_val', 'r2_test', 'train_loss', 'val_loss', 'test_loss',
                               'spline_train_loss', 'spline_test_loss'] if c in df.columns]
    first_cols = metric_cols + [c for c in ['seed', 'device'] if c in df.columns]
    param_cols = sorted([c for c in df.columns if c.startswith('param_')])
    other_cols = [c for c in df.columns if c not in first_cols + param_cols]
    df = df[first_cols + other_cols + param_cols]
    return df


essential_best_cols = ['train_loss', 'val_loss', 'test_loss', 'r2_train', 'r2_val', 'r2_test', 'seed', 'device']


def _save_excel_single(excel_path: str, results: List[Any], best: Any, progress: Dict[str, Any], last_result: Any):
    if not excel_path.lower().endswith(('.xlsx', '.xls')):
        excel_path = excel_path + '.xls'

    df_results = _results_to_dataframe(results)

    # Prepare best/progress/last_result sheets
    if hasattr(best, '__dict__') and isinstance(best, TrialResult):
        best_dict = asdict(best)
    elif isinstance(best, dict):
        best_dict = dict(best)
    else:
        try:
            best_dict = asdict(best)
        except Exception:
            best_dict = dict(best)
    df_best = pd.DataFrame([_trial_to_row(best_dict)])

    if isinstance(progress, dict):
        df_progress = pd.DataFrame([progress])
    else:
        df_progress = pd.DataFrame([{'completed': progress[0], 'total': progress[1]}])

    if hasattr(last_result, '__dict__') and isinstance(last_result, TrialResult):
        last_dict = asdict(last_result)
    elif isinstance(last_result, dict):
        last_dict = dict(last_result)
    else:
        try:
            last_dict = asdict(last_result)
        except Exception:
            last_dict = dict(last_result)
    df_last = pd.DataFrame([_trial_to_row(last_dict)])

    if os.path.exists(excel_path):
        writer_options = dict(mode='a', engine='openpyxl', if_sheet_exists='replace')
    else:
        writer_options = {}
    with pd.ExcelWriter(excel_path, **writer_options) as writer:
        df_results.to_excel(writer, index=False, sheet_name='results')
        df_best.to_excel(writer, index=False, sheet_name='best')
        df_progress.to_excel(writer, index=False, sheet_name='progress')
        df_last.to_excel(writer, index=False, sheet_name='last_result')


def _aggregate_by_params(results: List[Any]):
    df_tmp = _results_to_dataframe(results)
    metric_cols = [c for c in ['r2_train', 'r2_val', 'r2_test', 'train_loss', 'val_loss', 'test_loss',
                               'spline_train_loss', 'spline_test_loss']
                   if c in df_tmp.columns]
    param_cols = [c for c in df_tmp.columns if c.startswith('param_')]
    if param_cols:
        gdf = df_tmp[param_cols + metric_cols]
        grouped = gdf.groupby(param_cols, dropna=False)
        mean_df = grouped.mean(numeric_only=True)
        std_df = grouped.std(numeric_only=True).fillna(0.0)
        count_series = grouped.size().rename('n_trials')
        # Build rows
        agg_rows = []
        for idx_vals, mean_row in mean_df.iterrows():
            if not isinstance(idx_vals, tuple):
                idx_vals = (idx_vals,)
            params_dict = {k[len('param_'):]: v for k, v in zip(param_cols, idx_vals)}
            row = {'params': params_dict, 'n_trials': int(count_series.loc[idx_vals])}
            for m in metric_cols:
                row[f'{m}_mean'] = float(mean_row[m]) if pd.notna(mean_row[m]) else None
                std_val = std_df.loc[idx_vals][m] if m in std_df.columns else None
                row[f'{m}_std'] = (float(std_val) if (std_val is not None and pd.notna(std_val)) else 0.0)
            agg_rows.append(row)
        # Determine best aggregated
        best_agg = None
        if agg_rows:
            valid_r2 = [r for r in agg_rows if r.get('r2_val_mean') is not None and not (
                    isinstance(r.get('r2_val_mean'), float) and np.isnan(r.get('r2_val_mean')))]
            if valid_r2:
                best_agg = max(valid_r2, key=lambda r: r.get('r2_val_mean'))
            else:
                valid_val = [r for r in agg_rows if r.get('val_loss_mean') is not None and not (
                        isinstance(r.get('val_loss_mean'), float) and np.isnan(r.get('val_loss_mean')))]
                if valid_val:
                    best_agg = min(valid_val, key=lambda r: r.get('val_loss_mean'))

    else:
        agg_rows = []
        best_agg = None

    return agg_rows, best_agg


# For Excel, expand params into columns
def _expand_params(row: Dict[str, Any]) -> Dict[str, Any]:
    base = {k: v for k, v in row.items() if k != 'params'}
    for pk, pv in (row.get('params') or {}).items():
        base[f'param_{pk}'] = pv
    return base


def _save_excel_params_group(excel_path: str, agg_rows: List[Dict[str, Any]], best_agg: Dict[str, Any]):
    if not excel_path.lower().endswith(('.xlsx', '.xls')):
        excel_path = excel_path + '.xls'

    df_agg = pd.DataFrame([_expand_params(r) for r in agg_rows]) if agg_rows else pd.DataFrame()

    # Best aggregated row (if any)
    if best_agg is not None:
        best_agg_expanded = pd.DataFrame([_expand_params(best_agg)])
    else:
        best_agg_expanded = pd.DataFrame()

    if os.path.exists(excel_path):
        writer_options = dict(mode='a', engine='openpyxl', if_sheet_exists='replace')
    else:
        writer_options = {}
    with pd.ExcelWriter(excel_path, **writer_options) as writer:
            if not df_agg.empty:
                df_agg.to_excel(writer, index=False, sheet_name='results_avg_by_params')
            if not best_agg_expanded.empty:
                best_agg_expanded.to_excel(writer, index=False, sheet_name='best_avg_by_params')


def sweep_multkan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    seeds: Optional[List[int]] = None,
    n_jobs: int = os.cpu_count() or 1,
    use_cuda: bool = True,
    scaler_y: Optional[Any] = None,
    save_heading: str = None,
) -> Dict[str, Any]:
    """
    Run a hyperparameter sweep for MultKAN (sequential execution; no parallel computing).

    Parameters:
    - X_train, y_train, X_val, y_val, (optional X_test, y_test): numpy arrays
    - param_grid: dict of parameter lists to combine. Recognized keys include
        width, grid, k, mult_arity, steps, lamb, lr, etc. Keys not used by the model constructor are
        passed to fit().
    - seeds: list of seeds per trial combination; if None, uses [0].
    - n_jobs: number of worker processes (ignored; kept for backward compatibility).
    - use_cuda: if True and CUDA available, use cuda:0; otherwise CPU.
    - save_path: if provided, progress will be saved after each trial. If the path
      ends with .xlsx or .xls, results are written to an Excel workbook with sheets
      [results, best, progress, last_result]; otherwise a JSON snapshot is written.

    Returns a dict with keys: results (list of TrialResult dicts), best (best TrialResult dict).

    Notes:
    - To mimic the notebook behavior, you can include pruning in param_grid by setting
      'prune': [True] (or 'pruning': [True]). You can provide a unified threshold with 'pruning_th' to apply the same
      value to both node and edge pruning. Alternatively, you may still specify 'prune_node_th' and/or 'prune_edge_th'.
      If you set 'prune' or 'pruning' to False, the pruning step will be skipped. By default, pruning is disabled (False).
    - If you pass a fitted scaler_y (e.g., sklearn.preprocessing.MinMaxScaler used on y),
      metrics will be computed on the inverse-transformed (original) scale to match the notebook.
    """
    if param_grid is None:
        param_grid = {
            'width': [[X_train.shape[1], 8, 1]],
            'grid': [3],
            'k': [3],
            'mult_arity': [2],
            'steps': [50],
            'opt': ['LBFGS'],
            'lr': [1.0],
            'lamb': [0.0],
            'update_grid': [True],
            'prune': [False],
        }
    seeds = seeds or [0]

    combos = _expand_param_grid(param_grid)

    # Choose a single device for all runs
    if use_cuda and torch.cuda.is_available():
        device_choice = 'cuda:0'
    else:
        device_choice = 'cpu'

    tasks = []
    for ci, combo in enumerate(combos):
        dev = device_choice
        tasks.append((X_train, y_train, X_val, y_val, X_test, y_test, combo, dev, scaler_y))

    results: List[TrialResult] = []

    # Determine a single autosave path for this run (updated after each trial)
    if save_heading is None:
        save_heading = auto_save_path
    save_data_path = f"{save_heading}.xlsx"

    def _get_r2_val(r):
        try:
            return r.r2_val if hasattr(r, 'r2_val') else r.get('r2_val')
        except Exception:
            return None

    def _is_success(r):
        rv = _get_r2_val(r)
        if rv is None:
            return False
        try:
            from math import isnan
            return not isnan(rv)
        except Exception:
            return True

    def _asdict_any(r):
        try:
            return asdict(r)
        except Exception:
            return dict(r)

    # Sequential execution (parallel computing removed)
    total = len(tasks)
    for idx, t in enumerate(tasks, start=1):
        combo_params = t[6]
        prune_msg = ''
        # respect either 'prune' or 'pruning' flags (bool or string)
        def _want_prune_log(p: Dict[str, Any]) -> bool:
            val = p.get('prune', None)
            if val is None:
                val = p.get('pruning', False)
            if isinstance(val, str):
                v = val.strip().lower()
                return v in ('1','true','yes','y','t')
            return bool(val)
        if _want_prune_log(combo_params):
            # Prefer unified 'pruning_th' for display; fallback to individual thresholds
            _p_th = combo_params.get('pruning_th', None)
            _node_th = combo_params.get('prune_node_th', _p_th if _p_th is not None else 1e-2)
            _edge_th = combo_params.get('prune_edge_th', _p_th if _p_th is not None else 3e-2)
            if _p_th is not None:
                prune_msg = f", prune=True(th={_p_th})"
            else:
                prune_msg = f", prune=True(node_th={_node_th}, edge_th={_edge_th})"
        for seed_idx, seed_val in enumerate(seeds, start=1):
            print(f"[MultKAN Sweep] Training model {idx}/{total} -- seed # {seed_idx} (params={ {k: combo_params[k] for k in combo_params if k in ['lamb','lr',]} }{prune_msg})")
            try:
                res, model, _, _ = _run_single_trial((*t, seed_val))
                results.append(res)
                last_result_for_save: Any = res
            except Exception as e:
                import traceback, datetime
                err_info = {
                    'params': combo_params,
                    'val_loss': 999,
                    'train_loss': 999,
                    'test_loss': 999,
                    'r2_train': -99,
                    'r2_val': -99,
                    'r2_test': -99,
                    'seed': seed_val,
                    'device': t[7],
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'failed_at_index': idx,
                    'failed_at_total': total,
                    'timestamp': datetime.datetime.now().isoformat(timespec='seconds')
                }
                print(f"[MultKAN Sweep] Error on model {idx}/{total} -- seed #{seed_idx}: {e}")
                # Record the failure as a result row and continue
                results.append(err_info)
                last_result_for_save = err_info
            # After each trial (success or failure), attempt to save progress mandatorily
            try:
                successful = [r for r in results if _is_success(r)]
                if successful:
                    best_so_far = max(successful, key=lambda r: (getattr(r, 'r2_val', None) if hasattr(r, 'r2_val') else r.get('r2_val')))
                else:
                    best_so_far = None
                progress = {'completed': idx, 'total': total}
                _save_excel_single(
                    save_data_path, results, best_so_far[0] if best_so_far == [] else (best_so_far if best_so_far else {}),
                    progress, last_result_for_save)
            except Exception as e2:
                print(f"[MultKAN Sweep] Warning: failed to save progress: {e2}")
        agg_rows, best_agg = _aggregate_by_params(results)
        _save_excel_params_group(save_data_path, agg_rows, best_agg)


    out = {
        'results': [_asdict_any(r) for r in results],
        'results_avg_by_params': agg_rows,
        'results_table': _results_to_dataframe(results),
        'results_avg_table': pd.DataFrame([_expand_params(r) for r in agg_rows]) if agg_rows else pd.DataFrame(),
        'best': best_agg,
    }
    return out


def evaluate_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any],
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    seed: int = None,
    scaler_y: Optional[Any] = None,
    device_str: Optional[str] = 'cpu',
    save_heading: str = None,
) -> Tuple[TrialResult, MultKAN, Dict[str, Any], Dict[str, Any]]:

    if seed is None:
        seed = 0

    if type(params['width']) is str:
        params['width'] = eval(params['width'])
    if type(params.get('grid_range')) is str:
        params['grid_range'] = eval(params['grid_range'])
    if type(params.get('sym_a_range')) is str:
        params['sym_a_range'] = eval(params['sym_a_range'])
    if type(params.get('sym_b_range')) is str:
        params['sym_b_range'] = eval(params['sym_b_range'])
    params['width'] = [item[0] if type(item) is list else item for item in params['width']]

    if save_heading is None:
        save_heading = auto_save_path
    fig_name = f"{save_heading}_eval.png"

    res, model, fit_kwargs, dataset = _run_single_trial(
        (X_train, y_train, X_val, y_val, X_test, y_test, params, device_str, scaler_y, seed), verbose=True)
    device = torch.device(device_str)
    y_true, y_pred, mae, r2 = mae_and_r2(model, _to_tensor(X_val, device), _to_tensor(y_val, device), scaler_y=scaler_y)

    fig, ax = plt.subplots()
    plt.scatter(y_true, y_pred, color='k')
    plt.scatter(y_true, y_true, color='red')
    plt.savefig(fig_name)
    plt.show()

    return res, model, fit_kwargs, dataset


def _make_toy_dataset(n=200, noise=0.0, seed=0):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    _seed_everything(seed)
    x1 = np.linspace(-np.pi, np.pi, int(np.cbrt(n)))
    x2 = np.linspace(-1, 1, int(np.cbrt(n)))
    x3 = np.linspace(-1, 1, int(np.cbrt(n)))
    x1, x2, x3 = np.meshgrid(x1, x2, x3)
    y = 5 * np.exp(-x1) + 3 * x2 - x3 + noise * np.random.randn(*x1.shape)
    X = np.stack((x1.flatten(), x2.flatten(), x3.flatten()), axis=1)
    y = y.flatten().reshape(-1, 1)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

    sx, sy = MinMaxScaler((0.1, 0.9)), MinMaxScaler((0.1, 0.9))
    X_train = sx.fit_transform(X_train); X_val = sx.transform(X_val); X_test = sx.transform(X_test)
    y_train = sy.fit_transform(y_train); y_val = sy.transform(y_val); y_test = sy.transform(y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Hyperparameter sweep for MultKAN (sequential)')
    # Keep --n_jobs for backward compatibility but ignore it
    parser.add_argument('--n_jobs', type=int, default=1, help='Ignored (sequential execution)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available')
    parser.add_argument('--out', type=str, default='./multkan_sweep_results.json')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = _make_toy_dataset(seed=args.seed)

    param_grid = {
        'width': [[X_train.shape[1], 6, 1], [X_train.shape[1], 10, 1]],
        'grid': [3, 5],
        'k': [3],
        'mult_arity': [2, 3],
        'steps': [40, 60],
        'opt': ['LBFGS'],
        'lr': [1.0],
        'lamb': [0.0, 0.01],
        'update_grid': [True],
    }

    out = sweep_multkan(
        X_train, y_train, X_val, y_val, X_test, y_test,
        param_grid=param_grid,
        seeds=[0, 1],
        n_jobs=args.n_jobs,
        use_cuda=not args.no_cuda,
    )

    # Final save (overwrites with the complete results)
    try:
        if args.out.lower().endswith(('.xlsx', '.xls')):
            all_results = out['results']
            progress = {'completed': len(all_results), 'total': len(all_results)}
            results_objs = [TrialResult(**r) if isinstance(r, dict) else r for r in all_results]
            _save_excel_single(args.out, results_objs, out['best'], progress, all_results[-1])

            agg_rows_objs, best_agg_objs = _aggregate_by_params(results_objs)
            _save_excel_params_group(args.out, agg_rows_objs, best_agg_objs)
        else:
            with open(args.out, 'w') as f:
                json.dump(out, f, indent=2)
    except Exception as e:
        print(f"[MultKAN Sweep] Warning: failed to write final output to {args.out}: {e}")

    # Pretty print best
    best = out['best']
    print('Best configuration:')
    print(json.dumps(best, indent=2))


if __name__ == '__main__':
    main()
