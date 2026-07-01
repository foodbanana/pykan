import os
import sys
import pickle
import time
import platform
import subprocess
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor

import ray
import numpy as np
import pandas as pd

from utils import cprint, timestamp, remove_none, flatten_list, datadir, \
    scriptdir, checkexists, save, load, load_last_record

os.makedirs(os.path.join(datadir, 'temp'), exist_ok=True)

def get_connected_node_ip(domain='192.*'):
    command = 'cat /etc/hosts |grep -Po ' + domain
    lines = subprocess.check_output(command, shell=True).decode('utf-8').replace('\t', ' ').split('\n')
    info = {}
    for line in lines:
        if not line:
            continue
        string = line.split(' ')
        ip = string[0]
        node = string[1]
        info[node] = ip
    return info


def get_current_conda_env():
    """
    Detect the name of the currently active conda environment.

    Returns:
        str: The active conda environment name (e.g., 'doe', 'autoid').
    """
    env_name = os.environ.get('CONDA_DEFAULT_ENV')
    if env_name:
        return env_name

    try:
        output = subprocess.check_output(['conda', 'info', '--json'], stderr=subprocess.DEVNULL)
        import json
        info = json.loads(output)
        env_path = info.get('active_prefix_name', None)
        if env_path:
            return env_path
    except Exception:
        pass

    exe_path = sys.executable
    if '/envs/' in exe_path:
        return exe_path.split('/envs/')[-1].split('/')[0]
    elif '\\envs\\' in exe_path:
        return exe_path.split('\\envs\\')[-1].split('\\')[0]
    else:
        raise('Unexpected behavior')
    return 'base'


def get_parallel_settings(target, head='nodemaster01', domain='192.168.211.*', port='40001', 
                          env_name=None, n_cores_rest=2, redis_password='5241590000000000', 
                          reset_config=False):
    filedir = os.path.join(datadir, 'parallel_config.pkl')
    
    # Get environment name
    if env_name is None:
        env_name = get_current_conda_env()
    
    # Load configuration file if exists
    if checkexists(filedir, size_threshold=100) and not reset_config:
        settings = load(filedir, verbose=False)
        
        # Check if the loaded configuration matches to the current configuration
        if settings['target'] == target and settings['head'] == head and \
            settings['port'] == port and settings['domain'] == domain and \
            settings['runtime_env']['conda'] == env_name and \
            settings['_redis_password'] == redis_password:
               return settings
    
    # System-specific resource allocation
    settings = {}
    system = platform.system().lower()
    if target == 'local' or system == 'windows':
        settings['address'] = 'local'
        settings['_node_ip_address'] = 'localhost'
        settings['_redis_password'] = ''
        settings['num_cpus'] = max(os.cpu_count() - n_cores_rest, 1)
    elif system == 'linux':
        iplist = get_connected_node_ip(domain=domain)
        settings['address'] = 'ray://' + iplist[head] + ':' + port
        settings['_node_ip_address'] = iplist[head]
        settings['_redis_password'] = redis_password
        settings['num_cpus'] = None  # use all
    else:
        raise('#TODO')

    settings['target'] = target
    settings['system'] = system
    settings['head'] = head
    settings['port'] = port
    settings['domain'] = domain
    settings['runtime_env'] = {'env_vars': {'PYTHONPATH': scriptdir}, 'conda': env_name}

    # Save configuration for future use
    save(filedir, settings, verbose=False)
    return settings
    

def parallel_eval(func, inputs, params=None, parallel=True, resume=False, target='dist',
                  head='nodemaster01', port='40001', domain='192.168.211.*', redis_password='5241590000000000',
                  env_name=None, namespace='default', n_cores_rest=2, _save=False, basename=None,
                  saving_interval=10, init_params=True, debug=False, timeout=None,
                  reset_config=False, batch=False, batch_size=1000, **kwargs):
    '''
    - Use code format below:
    def function_name(input, params=None, i=None):
        if type(params) is ray._raylet.ObjectRef:
            params = ray.get(params)
    '''
    # Saving
    if not parallel and not _save:
        cprint('parallel_eval:: Running in the serial mode but saving is not activated.')
        
    if _save or resume:
        assert basename
        assert '/' not in basename
    
    # Parallel mode
    if parallel:
        settings = get_parallel_settings(target=target, head=head, port=port, domain=domain, 
                                         env_name=env_name, n_cores_rest=n_cores_rest, 
                                         redis_password=redis_password, reset_config=reset_config)
        cprint(timestamp(formal=True), '| Running', func.__name__, 'in parallel mode.', color='cyan')
        cprint(timestamp(formal=True), '| Target:', target, color='cyan')
        cprint(timestamp(formal=True), '| Parallel configuration:'); pprint(settings)
        
        if hasattr(parallel_eval, 'initiated'):
            if init_params:  # send parameters again
                params = ray.put(params)
                setattr(parallel_eval, 'params', params)
            else:
                params = getattr(parallel_eval, 'params')
            pass
        else:
            ray.init(namespace=namespace, address=settings['address'],  runtime_env=settings['runtime_env'], 
                     _node_ip_address=settings['_node_ip_address'], _redis_password=settings['_redis_password'],
                     num_cpus=settings['num_cpus'])
            parallel_eval.initiated = True
            params = ray.put(params)

            setattr(parallel_eval, 'params', params)
        
        # @ray.remote(scheduling_strategy="SPREAD")
        @ray.remote
        def pfunc(*args, **kwargs):
            return func(*args, **kwargs)
        
        cprint('Distributing jobs...', color='cyan')
        if isinstance(inputs, pd.DataFrame):
            works = [pfunc.remote(row, params=params, i=i, **kwargs) for i, (_, row) in enumerate(inputs.iterrows())]
        else:
            works = [pfunc.remote(_input, params=params, i=i, **kwargs) for i, _input in enumerate(inputs)]
        n_works = len(works)

        cprint('Job distribution completed. Start receiving...', color='cyan')
        if batch:
            results = []
            remaining = works
            while remaining:
                n = min(batch_size, len(remaining))
                finished, remaining = ray.wait(remaining, num_returns=n)
                results.extend(ray.get(finished))
                if _save:
                    filedir = os.path.join(datadir, 'temp', basename + '_' + str(n_works-n) + '.dill')
                    save(filedir, results, verbose=False)
        elif timeout:
            cprint('Timeout is activated. This can slow down parallel processing.', color='y')
            N = len(inputs) if not isinstance(inputs, pd.DataFrame) else len(inputs.index)
            results = [None] * N
            
            for i, work in enumerate(works):
                try:
                    results[i] = ray.get(work, timeout=timeout)
                    if _save:
                        filedir = os.path.join(datadir, 'temp', basename + '_' + str(n_works) + '.dill')
                        save(filedir, results, verbose=False)
                except Exception:
                    cprint('Timed out work:', i, color='y')
        elif debug:  # use this for debugging purpose
            cprint('Debugging is activated. This can slow down parallel processing.', color='y')
            results = [None]*len(inputs)
            for i, work in enumerate(works):
                try:
                    results[i] = ray.get(work)
                    if _save:
                        filedir = os.path.join(datadir, 'temp', basename + '_' + str(n_works) + '.dill')
                        save(filedir, results, verbose=False)
                except Exception as e:
                    cprint(f"[Need debugging] Error at i={i}: {e}", color='r')
                    raise
        else:
            results = ray.get(works)
            if _save:
                filedir = os.path.join(datadir, 'temp', basename + '_' + str(n_works) + '.dill')
                save(filedir, results, verbose=False)

    # Serial mode
    else:
        cprint(timestamp(formal=True), '| Running', func.__name__, 'in serial mode.', color='cyan')
        
        if resume:
            results, processed_upto = load_last_record(basename)
        else:
            results, processed_upto = [], -1
            
        if isinstance(inputs, pd.DataFrame):
            for i, (_, row) in enumerate(inputs.iterrows()):
                if resume and i < processed_upto:
                    continue
                results.append(func(row, params=params, i=i, **kwargs))
                if _save and (i != 0) and (i % saving_interval == 0):
                    filedir = os.path.join(datadir, 'temp', basename + '_' + str(i) + '.pkl')
                    save(filedir, results, verbose=False)
        else:
            for i, _input in enumerate(inputs):
                if resume and i < processed_upto:
                    continue
                results.append(func(_input, params=params, i=i, **kwargs))
                if _save and (i != 0) and (i % saving_interval == 0):
                    filedir = os.path.join(datadir, 'temp', basename + '_' + str(i) + '.pkl')
                    save(filedir, results, verbose=False)
    return results
    

def check_serializability(func):
    from ray.util import inspect_serializability
    inspect_serializability(func)
    return


if __name__ == '__main__':
    # Test parallel_eval
    def square(x, params=None, i=None):
        if type(params) is ray._raylet.ObjectRef:
            params = ray.get(params)
        return x*x
    
    func = square
    inputs = np.arange(10000)
    results = parallel_eval(func, inputs, params=None, parallel=True, target='dist', reset_config=True)
    a = 1