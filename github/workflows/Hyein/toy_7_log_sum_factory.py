import torch
import numpy as np


# ==========================================
# 1. Core Logic (Matches your KAN code)
# ==========================================
def create_log_sum_function(n_inputs, m_min=1, m_max=5, _device='cpu', seed=None):
    """
    Function factory from your KAN code.
    """
    generator = torch.Generator(device=_device)
    if seed is not None:
        generator.manual_seed(seed)

    multipliers = m_min + (m_max - m_min) * torch.rand(n_inputs, generator=generator, device=_device)
    # print(multipliers)

    def target_function(x):
        # x: [batch_size, n_inputs]
        safe_x = x + torch.ones_like(x) * (1 + 1e-3)
        multiplied_x = safe_x * (5 ** multipliers)
        log_x = torch.log(multiplied_x + 100.)
        final_sum = torch.sum(log_x, dim=1)
        return final_sum

    return target_function, multipliers


# ==========================================
# 2. Numpy Wrapper (For SHAP/SALib compatibility)
# ==========================================
def make_numpy_wrapper(torch_func, device='cpu'):
    """
    Wraps PyTorch function to accept Numpy arrays (for SALib/SHAP).
    """

    def wrapper(x_np):
        # x_np comes in as (n_features,) from apply_along_axis
        # We convert to (1, n_features) tensor
        x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            y_tensor = torch_func(x_tensor)
        return y_tensor.item()

    return wrapper


# ==========================================
# 3. Pre-generate Fixed Instances
# ==========================================
# We generate the exact functions used in your KAN paper/code here.
# Seed=0 ensures consistency across all your scripts.

LOG_SUM_ZOO = {}
TARGET_DIMS = [2, 5, 10, 30]

for nx in TARGET_DIMS:
    # Use seed=0 to match your "old code" exactly
    f_torch, multipliers = create_log_sum_function(nx, _device='cpu', seed=0)

    # Create the Numpy-friendly version
    f_numpy = make_numpy_wrapper(f_torch)

    # Store in the dictionary
    name = f"log_sum_{nx}d"
    LOG_SUM_ZOO[name] = {
        "func": f_numpy,  # The wrapper function
        "torch_func": f_torch,  # The original torch function (if needed)
        "multipliers": multipliers,  # The ground truth multipliers
        "bounds": [[-1, 1]] * nx,  # Bounds as used in your dummy_data
        "names": [f"x{i}" for i in range(nx)]
    }
