import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

data_pressure = [1, 5, 10, 50, 100, 200, 300, 400, 500, 700, 10000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000] # kPa

data_txt = pd.read_excel(r'.ignore/isothermnet_full_dataset/texturalProperties_vol.xlsx')
data_txt.head()

# Load the data into a variable
# data_x = torch.load('../../../.ignore/isothermnet_full_dataset/X_dataset_electro_xyz_bond_struc.pth')
data_y = torch.load('.ignore/isothermnet_full_dataset/y_dataset19.pth')
# data_H = torch.load('../../../.ignore/isothermnet_full_dataset/H_dataset.pth')


n_data = 5

x_vals = data_y['isotherm'][:n_data,:2].numpy()  # (5394, 19)
n_cat = x_vals.shape[1]

txt_vals = data_txt.iloc[:n_data,:].values

plt.scatter(x_vals[:,0], txt_vals[:,1], s=1, alpha=0.3)
plt.show()