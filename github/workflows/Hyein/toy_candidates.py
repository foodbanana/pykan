from kan.experiments.multkan_hparam_sweep_materials import sweep_multkan, evaluate_params
import numpy as np
import matplotlib.pyplot as plt
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"This script is running on {device}.")

x1_grid = np.linspace(-1, 1, 100)
x2_grid = np.linspace(-1, 1, 100)
# x3_grid = np.linspace(-1, 1, 10)
# x1, x2, x3 = np.meshgrid(x1_grid, x2_grid, x3_grid)
# X = np.stack((x1.flatten(), x2.flatten(), x3.flatten()), axis=1)
# y = np.exp(-x1) + x2 - x3**2
# y = 5 * np.exp(np.sin(x1)) + 3 * x2 - x3

x1, x2= np.meshgrid(x1_grid, x2_grid)
# X = np.stack((x1.flatten(), x2.flatten()), axis=1)
#%%
import os
save_dir = os.path.join(os.getcwd(), "github\workflows\Hyein\example_toys")

y_fun = lambda _x0, _x1: (1 - 2*_x0) ** 2 + 100 * (1 + 2*_x1 - 4 * _x0**2) ** 2
y = np.array([[y_fun(xxx1, xxx2) for xxx1, xxx2 in zip(xx1, xx2)] for xx1, xx2 in zip(x1, x2)])
eqn = "rosenbrock_log"

fig = plt.figure(figsize=(10, 8))

# Single variable
# ax = fig.add_subplot(111)
# ax.plot(x1_grid, y)
# ax.set_xlabel('x0')
# ax.set_ylabel('y')
# plt.savefig(os.path.join(save_dir, f"{eqn}.png"))
# plt.show()

# Double variable
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(x2, x1, np.log(y), cmap='viridis', edgecolor='none')
ax.set_xlabel('x1')
ax.set_ylabel('x0')
ax.set_zlabel('log y')
fig.colorbar(surface, shrink=0.5, aspect=5)
plt.savefig(os.path.join(save_dir, f"{eqn}.png"))
plt.show()

#%%
# Create a new figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)  # standard 2D plot (removed projection='3d')

# Create filled contour plot
# levels=20 determines the number of color bands (higher = smoother)
contour_filled = ax.contourf(x1, x2, np.log(y), cmap='viridis', levels=20)

# Optional: Add contour lines on top for better definition
# ax.contour(x2, x1, y, colors='white', alpha=0.3, linewidths=0.5)

ax.set_ylabel('x0')
ax.set_xlabel('x1')
ax.set_title(f'Contour Plot of {eqn}')

# Add colorbar
cbar = fig.colorbar(contour_filled)
cbar.set_label('log y')

# Save and Show
plt.savefig(os.path.join(save_dir, f"{eqn}_contour.png"))
plt.show()