import subprocess
import os

def main():
    target_scripts = [
        # "github/workflows/Hyein/material_NN_tuning.py",
        # "github/workflows/Hyein/material_NN_SHAP.py",
        "github/workflows/Hyein/material_KAN_sweep.py",
        "github/workflows/Hyein/material_KAN_analyze.py",
        # "github/workflows/Hyein/material_NN_for_KANrange.py",
    ]
    target_data = [
        # 'AgNP',
        # 'CO2RRLCA',
        # 'CO2RRNPV',
        # 'CO2RRMSP',
        'CO2HPx10v2',
        # 'CO2HEx10v2', # v2: 3 variables are excluded
        # 'CO2RRCC', 'CO2RRRA', 'CO2RRC', 'CO2RRA', 'CO2RREE', 'CO2RRCA',
        # 'CO2RRE', 'CO2RRRC', 'CO2RRUR',
        # 'AutoAM', 'Perovskite',
        # 'P3HT', 'CrossedBarrel',
        # 'MofEtotal'
    ]
    rand_seed = []
    # rand_seed = [i for i in range(10)]


    # target_scripts = [
    #     "github/workflows/Hyein/toy_KAN_sweep.py",
    #     "github/workflows/Hyein/toy_KAN_analyze.py",
    #     # "github/workflows/Hyein/toy_KAN_analyze_multi.py",
    #     "github/workflows/Hyein/toy_analytic_SHAP_Sobol.py",
    #     # DEPRECATED
    #     # "github/workflows/Hyein/toy_NN_tuning.py",
    #     # "github/workflows/Hyein/toy_NN_SHAP_Sobol.py",
    # ]
    # target_data = [
    #     # 'convolution', 'original', 'mult_periodic',
    #     'exponential', 'logarithm',
    #     # 'multiplication', 'conditional'
    #     # 'log_sum_2d', 'log_sum_5d', 'log_sum_10d', 'log_sum_30d',
    # ] # + [f'convex_seed_{i}' for i in range(30)]

    for s in target_scripts:
        for data in target_data:
            subprocess.run(['python', s, data])

if __name__ == '__main__':
    main()
