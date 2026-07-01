import os
import re
import sys
import signal
import errno
import itertools
import time
import dill
import pickle
import math
import subprocess
import warnings
from datetime import datetime
from inspect import stack

import numpy as np
import pandas as pd
import colorcet as cc  # pip install colorcet
import matplotlib as mpl
from matplotlib import colors, rcParams, cm
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm, to_rgb
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, AutoMinorLocator, MaxNLocator
import matplotlib.pyplot as plt  # pip install matplotlib==3.8.4
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec


# Turn-off negligible warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=SyntaxWarning)


# Figure settings
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

Figure.adjust = Figure.subplots_adjust  # define alias


_orig_axes_init = Axes.__init__
def _axes_init(self, *args, **kwargs):
    _orig_axes_init(self, *args, **kwargs)
    self.xaxis.set_minor_locator(AutoMinorLocator(5))
    self.yaxis.set_minor_locator(AutoMinorLocator(5))
Axes.__init__ = _axes_init


def list_fonts():
    fpaths = fm.findSystemFonts()

    for fpath in fpaths:
        try:
            f = fm.get_font(fpath)
        except:
            print('Error occurred in', fpath)
        print(f.family_name)
    return


def remove_ticks(ax, axis, label=False, spines=True):
    if 'x' in axis:
        ax.tick_params(axis='x', which='both', length=0, labelbottom=label)
        if not spines:
            ax.spines['bottom'].set_visible(False)
    
    if 'y' in axis:
        ax.tick_params(axis='x', which='both', length=0, labelleft=label)
        if not spines:
            ax.spines['left'].set_visible(False)
    return


def cleanup_directory(path):
    # Ensure the parent directories exist
    os.makedirs(path, exist_ok=True)  

    # Remove all files and subdirectories if the directory already exists
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # remove file or symbolic link
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # remove directory

    cprint('Directory', path, 'is now ready to use.', color='w')
    return


def temp_save(data, filename=None, extension='.dill'):
    temp_folder = os.path.join(datadir, 'temp')
    
    if not checkexists(temp_folder, size_threshold=0):
        os.makedirs(temp_folder, exist_ok=True)
        cprint('Created', temp_folder, color='c')
    
    if filename:
        filedir = os.path.join(temp_folder, filename + '_' + timestamp() + extension)
    else:
        filedir = os.path.join(temp_folder, timestamp() + extension)
        
    data = save(filedir)
    return


def temp_load(filename=None, extension='.dill'):
    temp_folder = os.path.join(datadir, 'temp')
    
    timestamp = get_latest_timeindex(None, 'temp', filename, extension=extension)
    
    if filename:
        filename = filename + '_' + timestamp + extension
        filedir = os.path.join(temp_folder, filename)
    else:
        filenames = os.listdir(temp_folder)
        for filename in filenames:
            if timestamp in filename:
                filedir = os.path.join(temp_folder, filename)
                break
    
    data = load(filedir)
    return data


def save(filedir, data, verbose=True):
    if '.dill' in filedir:
        with open(filedir, 'wb') as f:
            dill.dump(data, f, protocol=dill.HIGHEST_PROTOCOL)
    elif '.pkl' in filedir:
        with open(filedir, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise('#TODO')

    if verbose:
        cprint('Data has been saved to:', filedir)
    return


def load(filedir, verbose=True):
    if verbose:
        cprint('Loading data from', filedir, '...')
        
    if '.dill' in filedir:
        with open(filedir, 'rb') as f:
            data = dill.load(f)
    elif '.pkl' in filedir:
        with open(filedir, 'rb') as f:
            data = pickle.load(f)
    else:
        raise('#TODO')
    return data


def get_last_file_segment(basedir):
    folder, basename = basedir.rsplit('/', 1)
    saved_indices = [int(f.path.split('/')[-1].split('.')[0].split('_')[-1]) \
                    for f in os.scandir(folder) if basename in f.path]
    
    if saved_indices:
        processed_upto = np.max(saved_indices)
        filename = basename + '_' + str(processed_upto) + '.pkl'
        filedir = os.path.join(folder, filename)
        return filedir, processed_upto
    else:
        return None, None


def load_last_record(filename):
    basedir = os.path.join(datadir, 'temp', filename)
    filedir, processed_upto = get_last_file_segment(basedir)
    
    if filedir:
        results = load(filedir)
    else:
        results = list()
        processed_upto = 0
        
    return results, processed_upto


def get_latest_timeindex(project, folder, filename, extension='.dill'):
    if project == None:
        resultdir = os.path.join(datadir, folder)
    else:
        resultdir = os.path.join(datadir, project, folder)
    
    # Get the latest time stamp
    files = os.listdir(resultdir)
    time_stamps = []
    for file in files:
        strings = file.split('_')
        namepart = '_'.join(strings[:-2])  # except time stamp
        
        if filename == namepart and extension in file:
            pattern = r'\d{6}_\d{6}'
            match = re.search(pattern, file)
            time_stamps.append(match.group())
    
    l = [datetime.strptime(time_stamp, '%y%m%d_%H%M%S') for time_stamp in time_stamps]
    
    try: 
        latest = max(l)
    except ValueError:
        cprint(f'file containing {filename} cannot be found in te datadir/{folder}', color='r')
        raise()
    
    timeindex = latest.strftime('%y%m%d_%H%M%S')
    return timeindex


def isstring(value):
    return isinstance(value, str)


def isnumeric(value):
    return isinstance(value, (float, int, np.number)) and not np.isnan(value)

def isscalar(value):
    return isnumeric(value)


def isarray(value):
    return isinstance(value, (list, tuple, np.ndarray))


def istuple(value):
    return isinstance(value, tuple)


def isstringdata(data):
    return any(isstring(x) for x in data)


def isnumericdata(data):
    return any(isnumeric(x) for x in data)

def isscalardata(data):
    return isnumericdata(data)


def isarraydata(data):
    return any(isarray(x) for x in data)


def istupledata(data):
    return any(istuple(x) for x in data)


def apply_specialc(carray, specialc, specialc_loc, X):
    scolor = np.array(list(colors.to_rgba(specialc))).reshape(1, -1)
    if specialc_loc == 'first':
        loc = (X == np.argmin(X))
    else:
        loc = (X == np.argmax(X))
    carray[loc] = scolor
    return carray


def assign_groups(X, N):
    sorted_indices = np.argsort(X)
    groups = np.empty_like(X, dtype=int)
    splits = np.array_split(sorted_indices, N)
    for group_label, indices in enumerate(splits):
        groups[indices] = group_label
    return groups


def get_carray(X, mapping='linear', palette=None, alpha=1, specialc=False, 
               specialc_loc='first', N=None, shift_to_median=False):
    # Parse input
    if isinstance(X, int):  # number is given
        N = X
        X = list(range(N))
        mapping = 'discrete'  # override
        if palette is None:
            palette = 'cet_glasbey_category10'  # override
    else:
        if palette is None:
            palette = 'winter'
        
    # Assign color
    cmap = plt.get_cmap(palette)
    
    if mapping == 'discrete':
        carray = cmap(np.linspace(0, 1, N))
        if specialc:
            carray = apply_specialc(carray, specialc, specialc_loc, X)
        indices = np.argsort(X)
        carray = carray[indices,:]
    elif mapping == 'discrete_range':
        assert N
        carray = cmap(np.linspace(0, 1, N))
        indices = assign_groups(X, N)
        carray = np.vstack([carray[idx, :] for idx in indices])
    elif mapping == 'linear':  # linearly mapped
        if shift_to_median:
            norm = TwoSlopeNorm(vmin=np.min(X), vcenter=np.median(X), vmax=np.max(X))
        else:
            norm = plt.Normalize()
        carray = cmap(norm(X))
        if specialc:
            carray = apply_specialc(carray, specialc, specialc_loc, X)
    else:
        raise('#TODO')
    
    # Transparency
    carray[:, -1] = alpha
    return carray


def set_colorbar_deprecat(obj, pos, values, palette='winter', label=None, 
                 orientation='vertical', location='right', remove_ticks=False, 
                 labelpad=2, fontsize=10, shrink=1, reinitialize=False):
    
    # Set colormap
    cmap = plt.get_cmap(palette)
    norm = mpl.colors.Normalize(vmin=np.min(values), vmax=np.max(values))
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # Draw
    fig = obj[0]
    ax = obj[1]
    lid = 'cbar_' + label
    if hasattr(set_colorbar, lid) and not reinitialize:
        cbar = getattr(set_colorbar, lid)  # use existing cbar and update the colormap only
        cbar.update_normal(mappable)
    else:
        cbar = fig.colorbar(mappable=mappable, ax=ax, location=location, 
                            orientation=orientation, shrink=shrink, label=label)
    
    # Make cache
    setattr(set_colorbar, lid, cbar)
    
    # Decoration
    if orientation == 'horizontal':
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.ax.xaxis.set_label_position('bottom')
    
    if remove_ticks:
        cbar.set_ticks([])
        
    if label:
        if orientation == 'horizontal':
            cbar.ax.set_xlabel(label, labelpad=labelpad, fontsize=fontsize)
        else:
            cbar.ax.set_ylabel(label, labelpad=labelpad, fontsize=fontsize)
    return cbar


def set_colorbar(fig, pos, values, palette='winter', label=None, label_location='bottom',
                 orientation='vertical', remove_ticks=False, labelpad=2, 
                 labelfontsize=10, tickerfontsize=8, shrink=1, n_ticks=3,
                 shift_to_median=False):
    # Set colormap
    cmap = plt.get_cmap(palette)
    if shift_to_median:
        norm = TwoSlopeNorm(vmin=np.min(values), vcenter=np.median(values), vmax=np.max(values))
    else:
        norm = colors.Normalize(vmin=np.min(values), vmax=np.max(values))
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # Draw colorbar
    lid = 'cbar_' + label
    if hasattr(set_colorbar, lid):  # use existing cbar and update the colormap only
        cbar = getattr(set_colorbar, lid)
        cbar.update_normal(mappable)
    else:
        cax = fig.add_axes(pos)
        cbar = fig.colorbar(mappable=mappable, cax=cax, orientation=orientation, shrink=shrink)
        cbar.ax.tick_params(labelsize=tickerfontsize)
        if orientation == 'horizontal':
            cbar.ax.xaxis.set_ticks_position(label_location)
            cbar.ax.xaxis.set_label_position(label_location)
    
    # Make cache
    setattr(set_colorbar, lid, cbar)
    
    # Decoration
    if remove_ticks:
        cbar.set_ticks([])
    else:
        ticks = np.linspace(np.min(values), np.max(values), n_ticks)
        span = np.max(values) - np.min(values)
        if span > 10:
            labelformat = FormatStrFormatter('%.0f')
        elif span > 0.1:
            labelformat = FormatStrFormatter('%.1f')
        elif span > 0.01:
            labelformat = FormatStrFormatter('%.2f')
        else:
            labelformat = FuncFormatter(one_digit_exponent)
        
        if orientation == 'horizontal':
            cbar.ax.set_xticks(ticks)
            cbar.ax.xaxis.set_major_formatter(labelformat)
        else:
            cbar.ax.set_yticks(ticks)
            cbar.ax.yaxis.set_major_formatter(labelformat)
            
    if label:
        if orientation == 'horizontal':
            cbar.ax.set_xlabel(label, labelpad=labelpad, fontsize=labelfontsize)
        else:
            cbar.ax.set_ylabel(label, labelpad=labelpad, fontsize=labelfontsize)
            
    return cbar


def one_digit_exponent(x, pos):
    s = f"{x:.1e}"                     # e.g. "1.23e+03"
    s = re.sub(r"e([+-])0+(\d+)",      # regex to find e+0? or e-0?
               r"e\1\2",               # and collapse to e+? or e-?
               s)
    return s
    

def inspect_missing_values(df):
    num_inf = np.isinf(df).values.sum()
    num_nan = np.isnan(df).values.sum()
    num_null = df.isnull().values.sum()
    if num_inf + num_nan + num_null > 0:
        raise('Some non-numeric values were detected')
    else:
        return
    
    
def prettynum(x):
    if isarray(x):
        return [prettynum(_x) for _x in x]
    else:
        if abs(x) > 1:
            return round(x, 1)  
        elif abs(x) > 0.1:
            return round(x, 2)
        elif abs(x) > 0.01:
            return round(x, 3)
        elif abs(x) > 0.001:
            return round(x, 4)
        else:
            return sci_round(x, sig=2)


def sci_round(x, sig=2):
    if x == 0:
        return x
    
    exp = math.floor(math.log10(abs(x)))
    dec_places = -exp + (sig - 1)
    return round(x, dec_places)

    
def kill_processes(process, n_threshold=0):
    try:
        # Count the number of process
        command = 'tasklist | find /I /C "' + process + '"'
        out = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
        n_proc = int(out.decode().splitlines()[0])
        
        # Kill Aspen processes if they are too many
        if n_proc > n_threshold:
            command = 'taskkill /F /IM ' + '"' + process + '" /T'
            out = subprocess.run(command, capture_output=True)
            cprint('All', process, 'processes are terminated.', color='c')
    except:
        pass
    return
    
    
def timer(i, N=None):
    while 1:
        if timestamp(only_hour=True) >= 18 or \
            timestamp(only_hour=True) < 9 or \
            datetime.today().strftime('%A') in ['Saturday', 'Sunday']:
            return
        else:
            if N:
                string = '(' + str(i) + '/' + str(N) + ')'
            else:
                string = i
            cprint('Finished up to', string, '. Sleep 1 hour...')
            time.sleep(60*60)  # 1 hour sleep
    return


def timestamp(formal=False, only_hour=False):
    if only_hour:
        return int(time.strftime('%H'))
    elif formal:
        return time.strftime('%Y-%m-%d %H:%M:%S')
    else:
        return time.strftime('%y%m%d_%H%M%S')
    return


def remove_axes(ax):
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return


def find_factors(number):
    factors = list()
    for i in range(1, number + 1):
        if number % i == 0:
            factors.append(i)
    return factors


def make_rectangular(array):  # working for list and array
    if len(array) <= 4:
        array = np.atleast_2d(array)
        m, n = array.shape
        return array, m, n
    
    N = len(array)
    if N % 2 != 0:
        if type(array) is list:
            array.append(None) 
        elif type(array) is np.ndarray:
            array = np.append(array, None)
        else:
            raise('#TODO')
        N += 1
    array = np.atleast_2d(array)
    factors = find_factors(N)
    for m in np.flipud(factors):
        n = int(N/m)
        if m/n < 1:
            array = array.reshape(m, n)
            return array, m, n


def overrite_folder(source_folder, destination_folder, ignore=None):
    for file_name in os.listdir(source_folder):
        if file_name in ignore:
            continue
        source = os.path.join(source_folder, file_name)
        destination = os.path.join(destination_folder, file_name)
        shutil.copy(source, destination)
        

def detect_root(path, debug=False):
    try:
        # folderlist = [d for d in os.listdir(path) if os.path.isdir(d)]
        folderlist = [d for d in os.listdir(path)]
        if debug:
            print('Files inside ', path)
            for d in os.listdir(path):
                print(d + ' is ' + str(os.path.isdir(d)))
            print('')
        
        if 'script' in folderlist:
            return True
        else:
            return False
    except:
        return False
        
        
def find_root(path):
    max_depth = 5
    depth = 0
    while 1:
        path = os.path.abspath(os.path.join(path, os.pardir))   
        if detect_root(path):
            break
        elif depth > max_depth:
            raise('Cannot find root.')
        depth += 1
    return path


def convert_numbers_to_subscript(string, latex=True):
    string_ = re.sub(r'(\d+)', r'_{\1}', string)
    if latex:
        return f"${string_}$"
    else:
        return string_


def zoomout(ax, margin=0.1, axis='both'):
    x0, x1 = ax.get_xlim()
    xspan = x1 - x0
    
    y0, y1 = ax.get_ylim()
    yspan = y1 - y0

    if axis == 'both' or axis == 'x':
        ax.set_xlim(x0 - margin*xspan, x1 + margin*xspan)
    
    if axis == 'both' or axis == 'y':
        ax.set_ylim(y0 - margin*yspan, y1 + margin*yspan)
    return


def drop_keys(dictionary, mode):
    del_keys = []
    for key, value in dictionary.items():
        if mode == 'zero' and value == 0:
            del_keys.append(key)
        elif mode == 'negative' and value < 0:
            del_keys.append(key)
        elif mode == 'trace' and abs(value) < 1e-3:
            del_keys.append(key)
            
    for key in del_keys:
        dictionary.pop(key)
    return dictionary


def filter_non_trivial_value_keys(dictionary):
    keys = []
    for key, value in dictionary.items():
        if isinstance(value, pd.DataFrame):
            keys.append(key)
        elif isnumeric(value) and value != 0:
            keys.append(key)
        elif isnumeric(value) and value == 0:
            continue
        else:
            raise('#TODO')
    return keys


def set_path(workingdir):
    rootdir = find_root(workingdir)
    bindir = os.path.join(rootdir, 'bin')
    datadir = os.path.join(rootdir, 'dat')
    scriptdir = os.path.join(rootdir, 'script')
    imgdir = os.path.join(rootdir, 'img')
    srcdir = os.path.join(rootdir, 'src')
    sys.path.append(scriptdir)
    sys.path.append(srcdir)
    return workingdir, rootdir, bindir, datadir, scriptdir, imgdir, srcdir
workingdir, rootdir, bindir, datadir, scriptdir, imgdir, srcdir = set_path(os.path.abspath(__file__))


def flatten_list(l):
    if not type(l[0]) is list:
        cprint('Given list is not a nested list. Skip.')
        return l
    try:
        return list(itertools.chain(*l))
    except:
        out = []
        _l = l
        while _l:
            x = _l.pop(0)
            if isinstance(x, list):
                _l[0:0] = x
            else:
                out.append(x)
        return out


def find_null_indices(l):
    if type(l) is pd.Series:
        return l.index[l.isnull()]
    elif type(l) is list:
        return [idx for idx, x in enumerate(l) if (x is None) or (len(x) == 0)]
    else:
        raise('#TODO')


def remove_none(data, targets='all', axis=0, condition='any'):
    if type(data) in [pd.Series, pd.DataFrame]:
        if targets == 'all':
            targets = list(data.columns)
        if type(targets) != list:
            targets = [targets]
        indices = list()
        for target in targets:
            idx = data[target].isnull()
            indices.append(idx)
        indices = np.array(indices).T
        if axis == 0:
            if condition == 'any':
                idx_match = indices.any(axis=1)
            elif condition == 'all':
                idx_match = indices.all(axis=1)
            return data.loc[~idx_match,:]
        else:
            if condition == 'any':
                idx_match = indices.any(axis=0)
            elif condition == 'all':
                idx_match = indices.all(axis=0)
            return data.loc[:,~idx_match]
    else:
        return list(filter(None, data))


def remove_nan(data):
    return list(filter(np.nan, data))


def remove_inf(data):
    return list(filter(np.inf, data))


def remove_infeasible_values(data):
    return remove_nan(remove_inf(remove_none(data)))


def markrow(i):
    return '(Row ' + str(i) + ')'


def get_keys(dictionary, value):
    return [k for k, v in dictionary.items() if v == value]


def reverse_dict(dictionary):
    return dict((v,k) for k,v in dictionary.items())


class bcolors:
    magenta = '\033[95m'; m = '\033[95m'
    blue = '\033[94m'; b = '\033[94m'
    cyan = '\033[96m'; c = '\033[96m'
    green = '\033[92m'; g = '\033[92m'
    yellow = '\033[33m'; y = '\033[33m'
    red = '\033[91m'; r = '\033[91m'
    white = '\033[37m'; w = '\033[37m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    end = '\033[0m'


def cprint(*args, color='cyan', inspect=True, end='\n', time=False):
    args = [str(arg) for arg in args]
    string = ' '.join(args)
    
    if inspect:
        parent_func_name = stack()[1][3]
        string = parent_func_name + ':: ' + string
    
    # Time
    if time:
        string = '[' + timestamp(formal=True) + ']  ' + string
    
    # Color
    cstring = getattr(bcolors, color) + string + getattr(bcolors, 'end')
    
    print(cstring, flush=True, end=end)
    return


def checkexists(filedir, size_threshold=1):  # in byte
    if os.path.exists(filedir) and (os.path.getsize(filedir) >= size_threshold):
        return True
    else:
        return False


def check_df_size(df):
    df.info(memory_usage='deep')
    return


def optimize_df(df):
    cprint('Checking original data size...', color='c')
    check_df_size(df)
    for col in df.columns:
        data = df[col]
        dtype = data.dtype
        if any(data.isna() | data.isnull()):
            pass
        elif str(dtype) == 'int':
            c_min = data.min()
            c_max = data.max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                data = data.astype(np.int8)
            elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                data = data.astype(np.uint8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                data = data.astype(np.int16)
            elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                data = data.astype(np.uint16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                data = data.astype(np.int32)
            elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                data = data.astype(np.uint32)                    
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                data = data.astype(np.int64)
            elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                data = data.astype(np.uint64)
        elif str(dtype) in ['float16', 'float32', 'float64']:
            c_min = data.min()
            c_max = data.max()
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                data = data.astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                data = data.astype(np.float32)
            elif c_min > np.finfo(np.float64).min and c_max < np.finfo(np.float64).max:
                data = data.astype(np.float64)
        elif str(dtype) == 'object':
            if isstringdata(data):
                if len(data) == len(data.unique()):
                    pass
                else:
                    data = data.astype('category')
                pass  # no room for optimization
            elif isnumericdata(data):
                _data = np.vstack(data)
                c_min = _data.min()
                c_max = _data.max()
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    _data = _data.astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    _data = _data.astype(np.float32)
                elif c_min > np.finfo(np.float64).min and c_max < np.finfo(np.float64).max:
                    _data = _data.astype(np.float64)
                data = np.split(_data, _data.shape[0])
            else:
                pass  # possibly 3rd party class instances
        else:
            raise('#TODO')
        df[col] = data
    cprint('Checking data size after optimization...', color='c')
    check_df_size(df)
    return df


def replace_slash(string, reverse=False):
    if reverse:
        return string.replace('!%!', '/')
        return string.replace('!$!', '\\')
    else:
        return string.replace('/', '!%!')
        return string.replace('\\', '!$!')


class TimeoutError(Exception):
    pass


def wtimeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    from functools import wraps
    
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)
        
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                print(e)
            finally:
                signal.alarm(0)
            return result
        return wraps(func)(wrapper)
    
    return decorator