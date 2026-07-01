import os
import platform
import warnings
import itertools
import random
import dill
from copy import deepcopy
from collections import Counter

import numpy as np
import pandas as pd
import sympy as sp
import torch
from torch.distributions.normal import Normal
from torch.optim import Adam
from gpytorch.models import ExactGP, exact_prediction_strategies
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.priors import GammaPrior
from gpytorch.means import Mean, ZeroMean, ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from kan import ex_round
from kan.custom import MultKAN
from kan.utils import create_dataset_from_data, truncate_numbers

from datamanager import get_X, get_Z, get_Y, get_Ymax, load_dataset, DATASET
from plotter import plot_data, plot_2D_contour, plot_regressor_accuracy, plot_XY, \
    plot_trajectory, plot_descriptor_transition, plot_descriptor_distribution
from utils import plt, mpl, gridspec, imgdir, datadir, cprint, flatten_list, isarray, \
    cleanup_directory, zoomout, checkexists, flatten_list, load_data
from putils import ray, parallel_eval


warnings.filterwarnings('error', category=RuntimeWarning)


def calc_distance(x1, x2, mode='L2-norm'):
    if 'L2-norm':
        return np.sqrt(np.sum((x1 - x2)**2))
    else:
        raise('#TODO')


def detach(value):
    return value.detach().cpu().numpy()


def tensorize(dataset):
    for name, data in dataset.items():
        if isinstance(data, torch.Tensor):
            pass
        elif isinstance(data, np.ndarray):
            dataset[name] = torch.from_numpy(data).float()
        else:
            raise('Unexpected behavior')
    return dataset


def inverse_transformation(formula):  # x = exp(z) or equivalently z = log(x) transformed case
    new_expr = deepcopy(formula)
    subs = {}
    for sym in new_expr.free_symbols:
        subs[sym] = sp.log(sym)  # replace z -> log(x)
    new_expr = new_expr.xreplace(subs)
    return new_expr


def replace_half_powers(expr):
    half = sp.Rational(1, 2)
    new_expr = deepcopy(expr)
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, sp.Pow):
            exp = sub.exp
            if (exp == half 
                or (exp.is_Float and abs(float(exp) - 0.5) < 1e-14) 
                or exp.equals(half)):
                new_expr = new_expr.xreplace({sub: sp.sqrt(sub.base)})
    return new_expr
    

def sqrt_clip(z, threshold=1e-6):
    z = np.asarray(z, dtype=float)
    return np.sqrt(np.clip(z, threshold, None))


def pow_clip(base, exp):
    base = np.asarray(base, dtype=float)
    try:
        if np.isscalar(exp) and float(exp) == 0.5:
            return sqrt_safe(base)
    except Exception:
        pass
    
    if isinstance(exp, (np.ndarray, list, tuple)) and np.allclose(np.asarray(exp, dtype=float), 0.5):
        return sqrt_safe(base)    
    return np.power(base, exp)
    

def log_clip(z, threshold=-10.0):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    mask = z > 0.0
    out[~mask] = float(threshold)
    out[mask] = np.log(z[mask])
    return out


clip_functions = {'sqrt': sqrt_clip, '**': pow_clip, 'pow': pow_clip, 'log': log_clip}
    
    
def eval_func(func, x):
    x = np.asarray(x, dtype=float)
    args = [x[:, i] for i in range(x.shape[1])]
    value = func(*args)
    
    if type(value) == list:  # maybe input x is grid
        value = np.vstack(value).T.mean(axis=0)
    else:
        value = value.squeeze()
    return value


def apply_epsilon(x, xrange, eps):
    xmin, xmax = xrange

    if isinstance(x, np.ndarray):
        if np.any(x < xmin) or np.any(x > xmax):
            raise ValueError('Unexpected behavior: x out of bounds')
        x_ = x.copy()
        x_[x_ == xmin] = xmin + eps
        x_[x_ == xmax] = xmax - eps
        return x_
    elif isinstance(x, torch.Tensor):
        if torch.any(x < xmin) or torch.any(x > xmax):
            raise ValueError('Unexpected behavior: x out of bounds')
        x_ = x.clone()
        x_[x_ == xmin] = xmin + eps
        x_[x_ == xmax] = xmax - eps
        return x_
    else:
        raise TypeError('x must be a numpy.ndarray or torch.Tensor')


def plot_equation(f, feature_range=(0.1, 0.9), eps=1e-6, n_points=100, n_discretization=5, figsize=(10, 3)):
    xvars = sorted(list(f.free_symbols), key=lambda s: s.name)
    input_dim = len(xvars)
    
    _f = sp.lambdify(xvars, f, modules=[clip_functions, 'numpy'])

    fig, axes = plt.subplots(1, input_dim, figsize=(10, 2.5))
    x_grid = np.linspace(*feature_range, n_discretization)
    x_grid[0] += eps
    x_grid[-1] -= eps
    points = list(itertools.product(x_grid, repeat=input_dim-1))
    
    for i, xi in enumerate(xvars):
        ax = axes[i]
        xline = np.linspace(*feature_range, n_points)
        
        for p in points:
            xothers = np.stack(p)
            Xothers = np.tile(xothers, (n_points, 1))
            X = np.insert(Xothers, i, xline, axis=1)
            yline = eval_func(_f, X)
            ax.plot(xline, yline, lw=1.0, alpha=0.9, color='k')

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(xi)
        if i == 0:
            ax.set_ylabel('y(X)')
    
    fig.tight_layout()
    return fig, axes
    

class KANMean(Mean):
    def __init__(self):
        super(KANMean, self).__init__()
    
    
    def plot_training_process(self, result):
        fig, ax = plt.subplots(1, 1)
        tr = np.array(result['train_loss']).ravel()
        te = np.array(result['test_loss']).ravel()
        ax.plot(tr, label='Train')
        ax.plot(te, label='Test')
        
        # Decoration
        ax.legend(frameon=False, loc='upper right')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss (RMSE)')
        return
    
    
    def get_dimensions(self, X, y):
        input_dim = X.shape[-1]
        
        if y.ndim == 1:
            output_dim = 1
        else:
            output_dim = y.shape[-1]
        
        return input_dim, output_dim
    
    
    def set_seed(self, seed):  # randomness control
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        eval("setattr(torch.backends.cudnn, 'deterministic', True)")  # for parallelization
        eval("setattr(torch.backends.cudnn, 'benchmark', False)")
        torch.cuda.manual_seed_all(seed)
        return
    
    
    def get_feature_range(self, dataset):
        X, y = self.concat_data(dataset)
        
        X_min, X_max = X.min(axis=0), X.max(axis=0)
        assert len(torch.unique(X_min.values)) == 1
        assert len(torch.unique(X_max.values)) == 1
        
        X_min = X_min.values[0]
        X_max = X_max.values[0]
        return torch.stack([X_min, X_max])
    
    
    def transform_data(self, dataset):  # transform x by z = log(x)
        X_train, X_test, _, _ = self.unpack_data(dataset)
        
        X_train = torch.log(X_train)
        X_test = torch.log(X_test)
        
        dataset['train_input'] = X_train
        dataset['test_input'] = X_test
        return dataset
    
    
    def unpack_data(self, dataset):
        X_train = dataset['train_input']
        X_test = dataset['test_input']
        y_train = dataset['train_label']
        y_test = dataset['test_label']
        return X_train, X_test, y_train, y_test
    
    
    def concat_data(self, dataset):
        X_train, X_test, y_train, y_test = self.unpack_data(dataset)
        X = torch.concat([X_train, X_test], axis=0)
        y = torch.concat([y_train, y_test], axis=0)
        return X, y
    
    
    def check_data(self, dataset):
        # Variable dimension should be the same for train and test data
        assert dataset['train_input'].shape[1] == dataset['test_input'].shape[1]
        
        # Prediction target (y) should be in (n,1) matrix
        for key in ['train_label', 'test_label']:
            if dataset[key].ndim == 1:
                dataset[key] = dataset[key].reshape(-1,1)
        return dataset
    
    
    def fit(self, dataset, feature_range=None, mult_arity=2, grid=5, seed=0, 
            k=3, hidden_dim=[1, 1], sparse_init=False, update_grid=False, 
            device='cpu', opt='LBFGS', lr=1e-3, check_initialization=True, 
            log_transformation=True, pre_regularization=True, 
            lamb=0.01, lamb_entropy=0.2, lamb_coef=0.0, lamb_coefdiff=0.1, pruning_th=0.1,
            draw=True, verbose=True):
        self.dataset = dataset
        self.mult_arity = mult_arity
        self.grid = grid
        self.seed = seed
        self.k = k
        self.sparse_init = sparse_init
        self.update_grid = update_grid
        self.device = device
        self.opt = opt
        self.lr = lr
        self.check_initialization = check_initialization
        self.log_transformation = log_transformation
        self.pre_regularization = pre_regularization
        self.lamb = lamb
        self.lamb_entropy = lamb_entropy
        self.lamb_coef = lamb_coef
        self.lamb_coefdiff = lamb_coefdiff
        self.pruning_th = pruning_th
        self.draw = draw
        self.verbose = verbose
        
        # Check dataformat
        dataset = deepcopy(dataset)  # unlink
        dataset = self.check_data(dataset)
        
        # Randomness control
        self.set_seed(seed)
        
        # Data transformation
        dataset = tensorize(dataset)
        if log_transformation:
            dataset = self.transform_data(dataset)
        self.log_transformation = log_transformation
        
        # Get feature range
        if feature_range is None:
            feature_range = self.get_feature_range(dataset)
        self.feature_range = feature_range
        
        # Get X and y for future use
        if (len(dataset['train_label']) == len(dataset['test_label'])) and \
            all(dataset['train_label'] == dataset['test_label']):  # in BO
            X = dataset['train_input']
            y = dataset['train_label']
        else:
            X, y = self.concat_data(dataset)
            
        self.X = X
        self.y = y
        
        # Set architecture dimension
        input_dim, output_dim = self.get_dimensions(X, y)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Set xvars
        xvars_string = ['x' + str(i) for i in range(input_dim)]
        xvars = sp.symbols(xvars_string)
        self.xvars = xvars
        
        # Set optimizer configuration
        assert opt in ['Adam', 'AdamW', 'LBFGS']
        if opt in ['Adam', 'AdamW']:
            steps_train = 500
            steps_retrain = 200
        elif opt == 'LBFGS':
            steps_train = 100
            steps_retrain = 50
        else:
            raise('#TODO')
        
        # Generate network
        model = MultKAN(width=[input_dim, hidden_dim, output_dim], k=k, 
                        mult_arity=mult_arity, grid=grid, grid_range=feature_range, 
                        seed=seed, sparse_init=sparse_init, device=device)
        self.model = model

        # Check initialized network
        if check_initialization and draw:
            model(X)
            model.plot()
        
        # Train
        cprint('Training KAN model...')
        if pre_regularization:
            result = model.fit(dataset, opt=opt, lr=lr, steps=steps_train, 
                               update_grid=update_grid, monitor=verbose,
                               lamb=lamb, lamb_entropy=lamb_entropy, 
                               lamb_coef=lamb_coef, lamb_coefdiff=lamb_coefdiff)
        else:
            result = model.fit(dataset, opt=opt, lr=lr, steps=steps_train, 
                               update_grid=update_grid, monitor=verbose)
            
        # Pruning
        cprint('Pruning node and edges of KAN model...')
        model_sparse = model.prune(node_th=pruning_th, edge_th=pruning_th)
        
        # Re-training with pruned model
        cprint('Re-training with pruned KAN model...')
        result_sparse = model_sparse.fit(dataset, opt=opt, lr=lr, steps=steps_retrain, 
                                         update_grid=update_grid, monitor=verbose,
                                         lamb=lamb, lamb_entropy=lamb_entropy, 
                                         lamb_coef=lamb_coef, lamb_coefdiff=lamb_coefdiff)
        self.model_sparse = model_sparse
        
        # Symbolification
        self.formula = self.get_formula(verbose=verbose)
        
        if len(self.formula.free_symbols) == 0:
            raise('Unexpected behavior! self.get_formula returned may have returned constant value')
        
        if draw:
            self.plot_training_process(result)
            self.plot_training_process(result_sparse)
            
            self.model.plot()
            self.model_sparse.plot()
            
            self.plot_parity(dataset, use_formula=True)
            plt.show()
        
        return {'result': result, 'result_sparse': result_sparse}
    
    
    def get_formula(self, weight_simple=0.0, verbose=True, pretty=True):
        if self.log_transformation:
            lib = ['1/x', 'x', 'exp', 'gaussian']
        else:
            lib = ['x']
        
        # Get symbolic functions
        self.model_sparse.auto_symbolic(lib=lib, weight_simple=weight_simple, verbose=verbose)
        equations = self.model_sparse.symbolic_formula(var=self.xvars)[0]
        assert len(equations) == 1
        formula = equations[0]
        
        # Get pretty symbolic function
        if pretty:
            formula = truncate_numbers(formula)
        
        # Replace x**0.5 to sqrt(x)
        formula = replace_half_powers(formula)
        
        # Get the inverse-transformed equation
        if self.log_transformation:
            formula = inverse_transformation(formula)
        
        if verbose:
            print(formula)
        return formula
    
    
    def get_complexity(self, verbose=True):
        complexity = self.model_sparse.eval_complexity()
        
        if verbose:
            print(complexity)
        return complexity
    
    
    def plot_parity(self, dataset=None, use_formula=True, use_sparse=True):
        if dataset is None:  # use archived
            dataset = self.dataset
        
        # Parse data
        X_train, X_test, y_train, y_test = self.unpack_data(dataset)
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
        
        # Get predictions
        self.model.eval()
        self.model_sparse.eval()
        with torch.no_grad():
            y_train_pred = detach(self.forward(X_train, use_formula=use_formula, use_sparse=use_sparse))
            y_test_pred = detach(self.forward(X_test, use_formula=use_formula, use_sparse=use_sparse))
        
        # Plot
        fig, ax = plt.subplots(1, 1)
        ax.scatter(y_train, y_train_pred, label='Train', facecolor='k', edgecolor='k', zorder=1)
        ax.scatter(y_test, y_test_pred, label='Test', facecolor='w', edgecolor='k', zorder=0)
        
        # Draw y=x line
        ax.plot([0, 1], [0, 1], color='k', linestyle='-', label='y(pred) = y(exp)', zorder=-2)
        
        # Decoration
        ax.set_xlabel('y(exp)')
        ax.set_ylabel('y(pred)')
        ax.legend(frameon=False, loc='upper left')
        return
    
    
    def forward(self, x, use_formula=True, use_sparse=True):
        if isinstance(x, torch.Tensor):
            x = x.float()
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        else:
            raise('Unexpected behavior')
        
        if use_formula:  # formula is already inverse-transformed
            f_KAN = sp.lambdify(self.xvars, self.formula, [clip_functions, 'numpy'])
            y = eval_func(f_KAN, x)
            y = torch.from_numpy(y).float()
            
        elif use_sparse:
            if self.log_transformation:
                x = torch.log(x)
            y = self.model_sparse(x).squeeze()
            
        else:
            if self.log_transformation:
                x = torch.log(x)
            y = self.model(x).squeeze()
        
        if torch.isnan(y).any():
            raise('Unexpected behavior!')
        return y
    
    
class ExactGPModel(ExactGP):
    def __init__(self, x, y, likelihood, nu, prior_length, prior_alpha, mean=None):
        super(ExactGPModel, self).__init__(x, y, likelihood)
        
        # Mean
        self.mean = mean
        
        # Base kerel
        kernel = MaternKernel(nu=nu, ard_num_dims=x.shape[-1], lengthscale_prior=prior_length['param'])
        
        # Covraiance
        self.cov = ScaleKernel(kernel, outputscale_prior=prior_alpha['param'])
        self.cov.outputscale = prior_alpha['mode']  # initial value
        self.cov.base_kernel.lengthscale = prior_length['mode']  # initial value

    
    def forward(self, x):
        x_mu = self.mean(x)
        x_cov = self.cov(x)
        mu_and_sigma = MultivariateNormal(x_mu, x_cov)
        
        # Checker
        if torch.isnan(x_mu).any():
            raise('Unexpected behavior!')
        if torch.isnan(mu_and_sigma.mean).any():
            raise('Unexpected behavior!')
        if torch.isnan(mu_and_sigma.stddev).any():
            raise('Unexpected behavior!')
        return mu_and_sigma


class DiscreteBO:  #! convert df to solve maximization problem if necessary
    def __init__(self, dataset_name, df, indices_initial, feature_range, m=0, kernel='GP',
                 acquisition='EI', y_bar=0.8, nu=5/2, sigma_min=1e-5, device='cpu', 
                 draw=True, monitor=True, savefig=True, imgoutdir=None):
        
        # Importing dataset
        self.dataset_name = dataset_name
        df.reset_index(drop=True, inplace=True)
        self.df = df
        self._X = get_X(df)
        self._Y = get_Y(df)
        
        self.indices = df.index.values.tolist()
        self.indices_initial = indices_initial
        
        self.input_dim = self._X.shape[-1]
        self.xvars = ['x' + str(i) for i in range(self.input_dim)]
        
        # Get y_bar
        self.feature_range = feature_range
        feature_span = feature_range[1] - feature_range[0]
        self.y_bar = feature_range[0] + y_bar * feature_span
        self.indices_y_bar = list(np.array(list(df.index))[self._Y >= self.y_bar])
        
        # Set y_opt
        self.idx_opt = self.get_optimal_idx(df)
        self.y_opt = self._Y[self.idx_opt]
        
        # Configurations
        self.device = device
        self.zlabel = df.attrs['embedder']
        self.draw = draw
        self.monitor = monitor
        self.savefig = savefig
        self.imgoutdir = imgoutdir
        
        # Set model hyperparameters
        self.kernel = kernel
        self.acquisition = acquisition
        self.nu = nu  # smoothness parameter
        self.sigma_min = sigma_min  # minimum noise
        self.prior_sigma = {'param': GammaPrior(0.5, 5.0), 'mode': 0.1}  # noise parameter / GammaPrior(alpha, beta)
        
        # Set m and kernel models
        if isinstance(m, Mean):
            self.m = m
        elif m == 0 or m.lower() == 'zero':
            self.m = ZeroMean()
        elif m == 'constant':
            self.m = ConstantMean()  # default
        elif m == 'KAN':
            self.m = KANMean()
            self.params = load_data(os.path.join(datadir, self.dataset_name, 'best_param_KAN.pkl'))
        else:
            raise('#TODO')
        self.descriptors = None  # initialization
        
        if kernel == 'GP':
            self.prior_alpha = {'param': GammaPrior(2.0, 0.5), 'mode': 2.0}  # kernal output scale parameter / GammaPrior(alpha, beta)
            self.prior_length = {'param': GammaPrior(2.0, 1.0), 'mode': 2.0}  # kernal length parameter
        else:
            raise('#TODO')
        
        # Set logger
        self.log = self.initialize_log()
        return
    
    
    @classmethod
    def set_figure_panel(self, acquisition='DDI'):
        if acquisition in ['EI', 'UCB', 'EI']:
            fig = plt.figure(figsize=(7, 5))

            columns = fig.add_gridspec(
                1, 2,
                width_ratios=[1.0, 2.0],
                wspace=0.3,
                top=0.92, bottom=0.07, left=0.07, right=0.95
            )

            left  = columns[0, 0].subgridspec(2, 1, hspace=0.4)
            right = columns[0, 1].subgridspec(2, 2, hspace=0.4, wspace=0.3)

            axes = np.empty((2, 3), dtype=object)

            axes[0, 0] = fig.add_subplot(left[0, 0])
            axes[1, 0] = fig.add_subplot(left[1, 0])
            axes[0, 1] = fig.add_subplot(right[0, 0])
            axes[0, 2] = fig.add_subplot(right[0, 1])
            axes[1, 1] = fig.add_subplot(right[1, 0])
            axes[1, 2] = fig.add_subplot(right[1, 1])
        
        elif acquisition == 'DDI':
            fig = plt.figure(figsize=(7, 6))
            
            main = fig.add_gridspec(
                2, 1,
                height_ratios=[2, 0.9],
                hspace=0.2, top=0.92, bottom=0.07, left=0.07, right=0.95
            )

            top = main[0].subgridspec(1, 2, width_ratios=[1.0, 2.0], wspace=0.3)
            
            bottom = main[1].subgridspec(1, 2, width_ratios=[2.6, 1.0], wspace=0.2)
            bottom_left = bottom[0,0].subgridspec(1, 2, width_ratios=[2, 0.5], wspace=0.1)

            upper_left = top[0, 0].subgridspec(2, 1, hspace=0.4)
            upper_right = top[0, 1].subgridspec(2, 2, hspace=0.4, wspace=0.3)

            axes = np.empty((3, 3), dtype=object)
            
            axes[0, 0] = fig.add_subplot(upper_left[0, 0])
            axes[1, 0] = fig.add_subplot(upper_left[1, 0])
            
            axes[0, 1] = fig.add_subplot(upper_right[0, 0])
            axes[0, 2] = fig.add_subplot(upper_right[0, 1])
            axes[1, 1] = fig.add_subplot(upper_right[1, 0])
            axes[1, 2] = fig.add_subplot(upper_right[1, 1])
            
            axes[2, 0] = fig.add_subplot(bottom_left[0,0])
            axes[2, 1] = fig.add_subplot(bottom_left[0,1])
            axes[2, 2] = fig.add_subplot(bottom[0,1])
            
        else:
            raise('#TODO')
        
        system = platform.system().lower()
        
        if system == 'windows':
            manager = plt.get_current_fig_manager()
            manager.window.wm_geometry("+0+0")  # set display location based on the monitor
        
        return fig, axes
    
    
    def run(self, verbose=True):
        df = self.df
        indices_initial = self.indices_initial
        indices_sampled = deepcopy(indices_initial)
        self.indices_sampled = indices_sampled
        
        N = len(df) - len(indices_initial)
        idx_prev = indices_sampled[-1]
        
        # Setup figures
        if self.draw:
            assert self.imgoutdir
            cleanup_directory(self.imgoutdir)
            fig, axes = self.set_figure_panel(self.acquisition)
        
        # Run BO
        for n in range(N):
            cprint('Running BO | Iter = (', n, '/', N, ')', color='c', inspect=False)
            
            # Construct the model and train it
            self.construct_model(indices_sampled, verbose=verbose)
            
            # Train the kernel
            self.fit(verbose=verbose)
                
            # Propose next location
            idx_next = self.propose_location(self.acquisition, verbose=verbose)
            self.idx_next = idx_next
            
            # Conduct experiment
            y_next = self.virtual_experiment(idx_next)
            self.y_next = y_next
            indices_sampled.append(idx_next)
            self.indices_sampled = indices_sampled
            
            # Find current max
            y_max = self.find_current_y_max(indices_sampled)
            
            # Record progress
            self.record(idx_prev, idx_next, n, verbose=verbose)
            
            # Monitor progress
            if self.draw:
                self.monitor_progress(fig, axes, monitor=self.monitor, savefig=self.savefig)
            
            # Break or update
            if n + 1 == N:  # last iteration
                cprint('Reached last iteration point.')
                break
            else:
                self.update_priors()
                idx_prev = idx_next
            
            # Save image only up to y reaches y_bar
            if self.y_max > self.y_bar:
                self.savefig = False
        
        # Generate animation
        # render_animation(self.imgoutdir, extension='.png')  # postprocessing only the key results in __main__ of plotter.py
        return self.log
    
    
    def find_current_y_max(self, indices_sampled):
        y = get_Y(self.df)[indices_sampled]
        idx_max = indices_sampled[y.argmax()]
        y_max = y[y.argmax()]
        self.y_max = y_max
        self.idx_max = idx_max
        return y_max
    
            
    def initialize_log(self):
        return pd.DataFrame(columns=['iter', 'event', 'x_prev', 'x_next', 'y_next', 'y_max', 'a', 'I', 'loss', 'descriptors', 'noise', 'outputscale', 'lengthscale', 'd_xnext_xprev', 'd_xnext_xopt'])
    
    
    def get_optimal_idx(self, df):
        return int(np.argmax(df['Y'].values))
    
    
    def construct_model(self, indices_sampled, verbose=True):
        indices_unsampled = list(set(self.indices) - set(indices_sampled))
        
        # Parse data
        X = get_X(self.df, indices_sampled)
        y = get_Y(self.df, indices_sampled)
        _X = get_X(self.df, indices_unsampled)
        _y = get_Y(self.df, indices_unsampled)
        
        self.X = X
        self.y = y
        
        # Construct dataset
        dataset = {}
        dataset['train_input'] = X
        dataset['test_input'] = _X
        dataset['train_label'] = y.reshape(-1, 1)
        dataset['test_label'] = _y.reshape(-1, 1)
        
        # Update mu
        if isinstance(self.m, KANMean):
            self.m.train()
            for param in self.m.parameters():
                param.requires_grad = True
            
            self.m.fit(dataset, feature_range=self.feature_range, device=self.device, 
                       draw=False, verbose=verbose, **self.params)
            
            self.m.eval()
            for param in self.m.parameters():
                param.requires_grad = False
        elif isinstance(self.m, (ZeroMean, ConstantMean)):
            pass
        else:
            raise('#TODO')
        
        # Likelihood
        self.likelihood = GaussianLikelihood(noise_prior=self.prior_sigma['param'],
                                             noise_constraint=GreaterThan(self.sigma_min))
        self.likelihood.noise = torch.tensor([float(self.prior_sigma['mode'])])  # initial value
        
        # Posterior
        model = ExactGPModel(self.X, self.y, likelihood=self.likelihood, nu=self.nu, 
                             prior_length=self.prior_length, prior_alpha=self.prior_alpha, 
                             mean=self.m)
        
        # Set computation mode
        if torch.cuda.is_available() and self.device in ['gpu', 'cuda']:
            model = model.cuda()
        
        # For future use
        self.model = model
        return
        

    def fit(self, lr=0.1, n_iters=100, verbose=True):  # train the model
        self.model.train()
        self.likelihood.train()
        
        # Load data archived during model construction
        X = self.X
        y = self.y
        
        # Set optimizer
        optimizer = Adam(self.model.parameters(), lr=lr)
        mll = ExactMarginalLogLikelihood(likelihood=self.likelihood, model=self.model)
        
        # Fit hyperparameters
        cprint('Updating parameters...')
        for i in range(n_iters):
            optimizer.zero_grad()
            yhat = self.forward(X)
            loss = -mll(yhat, y)
            loss.backward()
            
            loss = np.round(loss.item(), 3)
            noise = np.round(self.model.likelihood.noise.item(), 3)
            outputscale = np.round(self.model.cov.outputscale.item(), 2)
            lengthscale = np.round(self.model.cov.base_kernel.lengthscale.squeeze().tolist(), 2)
            
            if verbose and (i+1) % (n_iters/10) == 0:  # check convergence
                cprint('    Iter', i+1, '/', n_iters, '| loss:', loss, '| noise:', noise, '| outputscale:', outputscale, '| lengthscale:', lengthscale, color='w', inspect=False)
            
            optimizer.step()
        
        # Record states
        self.loss = loss
        self.noise = noise
        self.outputscale = outputscale
        self.lengthscale = lengthscale
        
        self.model.eval()
        self.likelihood.eval()
        return
    
    
    def forward(self, x):  # sampling
        mu_and_sigma = self.model(x)
        yhat = self.likelihood(mu_and_sigma)
        return yhat


    def predict(self, x,  tensor=True):
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad():
            yhat = self.forward(x)
            
            mean = yhat.mean
            cov = yhat.covariance_matrix
            var  = cov.diag()
            var = var.clamp_min(self.prior_sigma['mode'])  # for numerical stability
            
        if tensor:
            return mean, var
        else:
            return detach(mean), detach(var)


    def sample_posterior(self, x, n_samples):
        self.model.eval()
        self.likelihood.eval()
        
        posterior = self.forward(x)
        
        samples = posterior.sample(n_samples)
        return samples
    
    
    def propose_location(self, acquisition=None, verbose=True):
        indices_unsampled = list(set(self.indices) - set(self.indices_sampled))
        X = get_X(self.df, indices_unsampled)
        
        if not acquisition:
            acquisition = self.acquisition
            
        a = self.eval_acquisition(X, acquisition=acquisition, verbose=verbose)
        idx = int(torch.argmax(a))
        
        return indices_unsampled[idx]
    
    
    def record(self, idx_prev, idx_next, n, verbose=True):
        loss = self.loss
        noise = self.noise
        outputscale = self.outputscale
        lengthscale = self.lengthscale
        a = self.a.tolist()
        I = self.I.tolist()
        y_next = self.y_next.tolist()
        y_max = self.y_max.tolist()
        y_bar = self.y_bar
        descriptors = self.descriptors
        
        X = get_X(self.df, tensor=False)
        x_prev = X[idx_prev,:]
        x_next = X[idx_next,:]
        x_opt = X[self.idx_opt,:]
        d_xnext_xprev = calc_distance(x_next, x_prev)
        d_xnext_xopt = calc_distance(x_next, x_opt)

        if hasattr(self, 'gate_disc') and self.gate_disc != 1.0:
            event = 'H_bar < H_bar_thres'
        elif hasattr(self, 'gate_exp') and self.gate_exp != 1.0:
            event = 'y_max > y_bar'
        else:
            event = None
        row = [n, event, x_prev, x_next, y_next, y_max, a, I, loss, descriptors, noise, outputscale, lengthscale, d_xnext_xprev, d_xnext_xopt]
        self.log.loc[n] = row
        
        if verbose:
            cprint('Descriptors in the last five rows')
            print(self.log['descriptors'].tail(5))
        return
    
    
    def evar_H_bar_gate(self, threshold=0.9):
        if len(self.log['descriptors']) < self.input_dim:  # not enough counts
            g = torch.tensor(1.0)
        else:
            counts = np.vstack(self.log['descriptors']).flatten()
                
            hist = Counter(counts)
            total_count = len(counts)
            probs = {x: hist.get(x, 0) / total_count for x in self.xvars}
                        
            p = np.array(list(probs.values()))
            p = p[p > 0]
            H = -np.sum(p * np.log(p))
            Hbar = H / torch.log(torch.tensor(self.input_dim, dtype=torch.float32))
            
            g = (Hbar >= threshold).float()
        return g
    
    
    def eval_DDI(self, x, model=None, gate_exp_threshold=1.0, gate_disc_threshold=0.5, eps=1e-6):
        # Unpack
        X = self.X
        y = self.y
        y_best = torch.max(y)
        y_bar = self.y_bar
        feature_range = self.feature_range
        
        # Exploitative improvement function
        with torch.no_grad():
            if model == None:
                mu = self.model(x).mean
            else:
                mu = model(x).mean
        
        r = y_bar / y_best
        gate_exp = (1/r <= gate_exp_threshold).float()
        I_exp = (mu - feature_range[0]) / (feature_range[1] - feature_range[0]) * r * gate_exp
        self.gate_exp = gate_exp
        
        # Discriminative improvement function
        f_KAN = self.m.formula
        zvars = sorted(f_KAN.free_symbols, key=lambda s: s.name)

        df = [sp.diff(f_KAN, z)**2 for z in zvars]  # measure for curvature
        df = [replace_half_powers(_df) for _df in df]
        _df = sp.lambdify(zvars, df, modules=[clip_functions, 'numpy'])

        zgrid = np.linspace(feature_range[0] + eps, feature_range[1] - eps, 10)
        meshes = np.meshgrid(*[zgrid]*len(zvars), indexing='ij')
        Z_grid = np.column_stack([m.ravel() for m in meshes])

        C = eval_func(_df, Z_grid)
        indices = np.argsort(C)
        
        if len(indices) > 1:
            j1 = indices[-1]  # best descriptor
            j2 = indices[-2]  # runner-up
            z1 = zvars[j1]
            z2 = zvars[j2]
            self.descriptors = [z1.name, z2.name]

            f_disc = 1/2 * (1 + (sp.diff(f_KAN, z1)**2 - sp.diff(f_KAN, z2)**2)/(sp.diff(f_KAN, z1)**2 + sp.diff(f_KAN, z2)**2))
            f_disc = replace_half_powers(f_disc)
            _f_disc = sp.lambdify(zvars, f_disc, modules=[clip_functions, 'numpy'])
            
            indices = [int(z.name.split('x')[1]) for z in zvars]
            z = x[:, indices]
            z = apply_epsilon(z, feature_range, eps=eps)
            
            gate_disc = self.evar_H_bar_gate(threshold=gate_disc_threshold)
            self.gate_disc = gate_disc
            
            I_disc = torch.from_numpy(eval_func(_f_disc, z)).float() * gate_disc
            
        elif len(indices) == 1:
            j1 = indices[0]
            z1 = zvars[j1]
            self.descriptors = [z1.name, z1.name]
            
            I_disc = I_exp * 0  # no need to discriminate
        else:
            raise('Unexpected behavior!')
            
        # Uniform improvement function
        Z = X[:, j1]
        z = x[:, j1]
        d_min = torch.min(np.abs(z[:,None] - Z[None,:]), axis=1).values
        I_uni = d_min / (feature_range[1] - feature_range[0])
        
        # Evaluate aquisition function
        a = I_exp + I_disc + I_uni
        if torch.isnan(a).any():
            raise('Unexpected behavior!')
        
        return a, (I_exp, I_disc, I_uni)
    
    
    def eval_EI(self, x):
        y_max = get_Ymax(self.df, self.indices_sampled)  # current Ymax
        mu, var = self.predict(x)
        sigma = torch.sqrt(var)
        
        diff = mu - y_max
        z = diff/sigma
        normal = Normal(0, 1)
        I_exploitation = diff * normal.cdf(z)
        I_exploration = sigma * normal.log_prob(z).exp()
        a = I_exploitation + I_exploration
        a[sigma <= 0.] = 0.
        
        return a, (I_exploitation, I_exploration)
    
    
    def eval_acquisition(self, x, acquisition=None, tensor=True, verbose=True):
        if acquisition is None:
            acquisition = self.acquisition
        
        if acquisition == 'EI':
            a, I = self.eval_EI(x)
            I_exploitation, I_exploration = I
            
            # Archive for monitoring
            idx = torch.argmax(a)
            self.a = a[idx]
            self.I = torch.tensor([I_exploitation[idx], I_exploration[idx]])
            self.I_labels = ['$I_{exploit}$', '$I_{explore}$']
            
        elif acquisition == 'UCB':
            mu, var = self.predict(x)
            sigma = torch.sqrt(var)
            kappa = 2.576
            a = mu + kappa * sigma
            I_exploitation = mu
            I_exploration = kappa * sigma

            idx = torch.argmax(a)
            self.a = a[idx]
            self.I = torch.tensor([I_exploitation[idx], I_exploration[idx]])
            self.I_labels = ['$I_{exploit}$', '$I_{explore}$']    
        
        elif acquisition == 'TS':
            samples = self.sample_posterior(x, n_samples=1).squeeze()
            a = samples
            I_exploitation = samples

            idx = torch.argmax(a)
            self.a = a[idx]
            self.I = torch.tensor([I_exploitation[idx]])
            self.I_labels = ['$I_{exploit}$']
    
        elif acquisition == 'DDI':  # descriptor discovery
            a, I = self.eval_DDI(x)
            I_exp, I_disc, I_uni = I
            
            # Archive for monitoring
            idx = torch.argmax(a)
            self.a = a[idx]
            self.I = self._I = torch.tensor([I_exp[idx], I_disc[idx], I_uni[idx]])
            self.I_labels = self._I_labels = ['$I_{exp}$', '$I_{disc}$', '$I_{uni}$']
            
        else:
            raise('#TODO')
        
        if tensor:
            return a
        else:
            return detach(a)
    
    
    def update_priors(self):
        self.prior_alpha['mode'] = self.model.cov.outputscale.detach()
        self.prior_length['mode'] = self.model.cov.base_kernel.lengthscale.detach()[0]
        self.prior_sigma['mode'] = self.model.likelihood.noise.detach()[0]
        return
    
    
    def virtual_experiment(self, idx):
        df = self.df
        return get_Y(df, tensor=False)[idx]
    
    
    def monitor_progress(self, fig, axes, monitor=True, savefig=True):
        # Parse data
        X = get_X(self.df, tensor=True)
        Y = get_Y(self.df, tensor=False)
        Z = get_Z(self.df, tensor=False)
        feature_range = self.feature_range
        
        # Evaluate model prediction
        Y_mean, Y_cov = self.predict(X, tensor=False)
        
        # Evaluate acquisition function values
        A = self.eval_acquisition(X, tensor=False, verbose=False)
        X = detach(X)  # now it's ok to unload
        
        # Get indices
        indices = self.indices
        indices_initial = self.indices_initial
        indices_sampled = self.indices_sampled
        indices_unsampled = list(set(indices) - set(indices_sampled))
        indices_y_bar = self.indices_y_bar
        idx_next = self.idx_next
        idx_opt = self.idx_opt
        idx_max = self.idx_max
        
        # Plot acquisition function
        plot_2D_contour(fig, axes[0,0], Z, Y_mean, indices_initial, 
                        indices_sampled, indices_unsampled, indices_y_bar,
                        idx_next, idx_max, idx_opt, xlabel=self.zlabel, ylabel='$\hat{Y}$')
        
        # Plot objective surface
        plot_2D_contour(fig, axes[1,0], Z, A, indices_initial, 
                        indices_sampled, indices_unsampled, indices_y_bar,
                        idx_next, idx_max, idx_opt, xlabel=self.zlabel, ylabel='a(x)', legend=False)
        
        # Plot regressor accuracy
        plot_regressor_accuracy(axes[0,1], Y, Y_mean, indices_initial, 
                                indices_sampled, indices_unsampled, indices_y_bar,
                                idx_next, idx_max, idx_opt, feature_range=feature_range,
                                xlabel='$Y(x)$', ylabel='$\mu(x)$', legend=False)
        
        # Plot improvement function
        n = self.log.index.values
        I = np.vstack(self.log['I'].values)
        plot_trajectory(axes[1,1], n, I, xlabel='Iterations', ylabel='$I(x_{next})$', labels=self.I_labels)
        
        # Plot distances
        d_xnext_xopt = np.vstack(self.log['d_xnext_xopt'].values)
        d_xnext_xprev = np.vstack(self.log['d_xnext_xprev'].values)
        
        plot_trajectory(axes[0,2], n, np.hstack((d_xnext_xopt, d_xnext_xprev)), 
                        xlabel='Iterations', ylabel='Distance', 
                        labels=['d($x_{next}$, $x^*$)', 'd($x_{next}$, $x_{prev}$)'])
        
        if self.acquisition in ['EI', 'UCB', 'TS']:
            # Plot kernel parameters
            noise = np.vstack(self.log['noise'].values)
            outputscale = np.vstack(self.log['outputscale'].values)
            lengthscale = np.vstack(self.log['lengthscale'].values)
            length_scale_labels = [f'$\ell_{i}$' for i in range(lengthscale.shape[-1])]
            
            plot_trajectory(axes[1,2], n, np.hstack((lengthscale, outputscale, noise)), 
                            xlabel='Iterations', ylabel='Measures', 
                            labels=[*length_scale_labels, '$\sigma^{2}_{f}$', '$\sigma^{2}_{n}$'])
            
        elif self.acquisition == 'DDI':
            # Plot XY (x = descriptors)
            descriptors = self.descriptors
            xvar1, xvar2 = descriptors[0], descriptors[1]
            idx1, idx2 = int(xvar1.split('x')[1]), int(xvar2.split('x')[1])
            X1 = X[:,idx1].squeeze()
            X2 = X[:,idx2].squeeze()
            xlabel1 = 'Best descriptor (' + xvar1 + ')'
            xlabel2 = 'Runner up (' + xvar2 + ')'
            plot_XY(axes[1,2], X1, Y, indices_initial, 
                    indices_sampled, indices_unsampled, indices_y_bar,
                    idx_next, idx_max, idx_opt, feature_range=feature_range, 
                    xlabel=xlabel1, ylabel='Y(x)', legend=False)
            plot_XY(axes[2,2], X2, Y, indices_initial, 
                    indices_sampled, indices_unsampled, indices_y_bar,
                    idx_next, idx_max, idx_opt, feature_range=feature_range, 
                    xlabel=xlabel2, ylabel='Y(x)', legend=False)
            
            # Plot selected descriptors
            n = self.log.index.values
            D = np.vstack(self.log['descriptors'])
            events = self.log['event'].values
            plot_descriptor_transition(axes[2,0], n, D, events, xlabel='Iterations', ylabel='Variables', yticks=self.xvars)
            plot_descriptor_distribution(axes[2,1], n, D, yticks=self.xvars)
        
        if monitor:
            plt.pause(0.1)
        
        if savefig:
            filedir =  os.path.join(self.imgoutdir, str(n[-1]) + '.png')
            fig.savefig(filedir)
        return


if __name__ == '__main__':
    # # Test panel generation
    # DiscreteBO.set_figure_panel()
    # plt.show()
    
    a = 1