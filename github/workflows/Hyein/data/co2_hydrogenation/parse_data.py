""" Only With Conda Environment np-dill (numpy>=2.0)"""

import pandas as pd
import os

d = pd.read_pickle(open('D:\pykan\github\workflows\Hyein\data\co2_hydrogenation\capture_mea_lower_duty_v1_direct_hydrogenation_v8_heat_integration_corrected_251115_235534_3.pkl', 'rb'))
save_heading = os.path.join(os.getcwd(), "github", "workflows", "Hyein", "data", "co2_hydrogenation", "data")

list_of_arrays = d['x'].tolist()
x_cols = pd.DataFrame(list_of_arrays)
x_cols.columns = [f'x{i+1}' for i in range(x_cols.shape[1])]

d_final = pd.concat([d, x_cols], axis=1)
d_final.to_csv(save_heading + "_parsed.csv", index=False)
