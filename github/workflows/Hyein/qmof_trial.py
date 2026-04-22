#%%
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Now import your other libraries
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
file_thermo = "D:\pykan\github\workflows\Hyein\data\qmof\qmof_thermo_database\qmof_thermo_database\qmof_thermo.json"

# Load the JSON directly into a table
df = pd.read_json(file_thermo)

# 2. Define the columns you want to visualize
# Add or remove names based on your specific JSON keys
target_columns = [
    'energy_above_hull',
    'formation_energy',
    'energy_total'
]
target_pretty_names = [
    'Ehull', 'Eform', 'Etotal'
]
# 3. Setup the Grid (e.g., 2 rows, 2 columns)
num_cols = len(target_columns)
cols_per_row = 4
rows = (num_cols + 1) // 4

fig, axes = plt.subplots(rows, cols_per_row, figsize=(16, 5 * rows))
axes = axes.flatten()  # Flattens to 1D so we can loop easily

# 4. The Loop
for i, col_name in enumerate(target_columns):
    ax = axes[i]

    # Ensure data is numeric/float (handling those Decimal objects)
    data_series = pd.to_numeric(df[col_name], errors='coerce')

    # Draw the histogram
    ax.hist(data_series.dropna(), bins=40, color='teal', edgecolor='white', alpha=0.8)

    # Styling
    ax.set_title(f'Distribution of {col_name.replace("_", " ").title()}', fontsize=12)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

# 5. Clean up extra subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

#%%
import ijson

# Path to your downloaded QMOF JSON file
file_structure = "D:\pykan\github\workflows\Hyein\data\qmof\qmof_database\qmof_database\qmof_structure_data.json"
# file_path = "D:\pykan\github\workflows\Hyein\data\qmof\qmof_thermo_database\qmof_thermo_database\qmof_thermo.json"

with open(file_structure, 'rb') as f:
    # 'item' assumes your JSON is a list: [{...}, {...}]
    objects = ijson.items(f, 'item')

    # Get just the first object and stop
    first_entry = next(objects)

    print("--- First Entry Keys ---")
    print(first_entry.keys())

#%%
import pandas as pd
import numpy as np
from decimal import Decimal
import json

with open(file_structure, 'r') as f:
    data_structure = json.load(f)

num_sites = [len(m['structure']['sites']) for m in data_structure]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hist(num_sites, bins=40, color='teal', edgecolor='white', alpha=0.8)

#%%
def flatten_mof(data):
    # 1. Convert all Decimals to float first
    def to_f(obj):
        return float(obj) if isinstance(obj, Decimal) else obj

    # 2. Extract Lattice Features
    lat = data['lattice']
    row = {
        'lattice_a': to_f(lat['a']),
        'lattice_b': to_f(lat['b']),
        'lattice_c': to_f(lat['c']),
        'lattice_alpha': to_f(lat['alpha']),
        'lattice_beta': to_f(lat['beta']),
        'lattice_gamma': to_f(lat['gamma']),
        'lattice_volume': to_f(lat['volume']),
        'num_sites': len(data['sites'])
    }
    return row

# Assuming data_structure is your list of MOF dictionaries
# Note: I'm using m.get('structure', m) in case 'structure' is the top-level key
df_structure = pd.DataFrame([flatten_mof(m['structure'] if 'structure' in m else m) for m in data_structure])

# 3. Save to CSV
for o, name in zip(target_columns, target_pretty_names):
    mof_out = df[o]
    df_out = pd.concat([df_structure, mof_out], axis=1)
    output_filename = f"github/workflows/Hyein/data/Mof{name}.csv"
    df_out.to_csv(output_filename, index=False)

    print(f"Success! Saved {len(df_out)} MOFs to {output_filename}")
    print(df_out.head())

#%%
import pandas as pd
import matplotlib.pyplot as plt

df_out = pd.read_csv("github/workflows/Hyein/data/MofEtotal.csv")
y_name = df_out.columns[-1]

num_x = len(df_out.columns[:-1])
cols_per_row = 4
rows = (num_x + 1) // cols_per_row
fig, axes = plt.subplots(rows, cols_per_row, figsize=(16, 5 * rows))
axes = axes.flatten()  # Flattens to 1D so we can loop easily

for x, ax in zip(df_out.columns[:-1], axes):
    ax.scatter(df_out[x], df_out[y_name])
plt.show()
