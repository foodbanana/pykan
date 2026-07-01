import torch
from kan import create_dataset
from kan.custom_multkan_ddp import MultKAN
import matplotlib.pyplot as plt
from kan.utils import ex_round

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

f = lambda x: torch.sin(2 * x[:, [0]]) * x[:, [1]]
dataset = create_dataset(f,
                         ranges=[-torch.pi, torch.pi],
                         n_var=2,
                         train_num=3000,
                         test_num=300,
                         device=device,
                         normalize_label=True,
                         normalize_input=True,
                         )

model = MultKAN(width=[2,4,1], grid=3, k=3, seed=0, device=device)
fit_kwargs = {'opt': 'LBFGS', 'lr': 1, 'lamb': 0.01,
              'lamb_entropy': 0.1, 'lamb_coef': 0.1, 'lamb_coefdiff': 0.5,}
model.fit(dataset, steps=20, **fit_kwargs)
model.plot()
plt.show()

model = model.prune(edge_th=0.05, node_th=0.05)
model.plot()
plt.show()

grids = [3, 5, 10, 20, 30]

train_rmse = []
test_rmse = []

for i in range(len(grids)):
    model = model.refine(grids[i])
    results = model.fit(dataset, steps=50, stop_grid_update_step=20, **fit_kwargs)
    model = model.prune(edge_th=3e-2, node_th=1e-2)
    train_rmse.append(results['train_loss'][-1].item())
    test_rmse.append(results['test_loss'][-1].item())

y_pred = model(dataset['test_input']).detach().cpu().numpy()
plt.scatter(dataset['test_label'], y_pred, color='k')
plt.scatter(dataset['test_label'], dataset['test_label'], color='r')
plt.show()

model.auto_symbolic(weight_simple=0.5)
fit_kwargs_sym = {'opt': 'LBFGS', 'lr': 1, 'lamb': 0.01,
                  'lamb_entropy': 0.1, 'lamb_coef': 0.1, 'lamb_coefdiff': 0.5,}
fit_kwargs_sym['update_grid'] = False
res = model.fit(dataset, steps=50, **fit_kwargs_sym)
print(res['test_loss'][-1])
model.plot()
plt.show()

exp = ex_round(model.symbolic_formula()[0][0], 4)
print(exp)

y_pred = model(dataset['test_input']).detach().cpu().numpy()
plt.scatter(dataset['test_label'], y_pred, color='k')
plt.scatter(dataset['test_label'], dataset['test_label'], color='r')
plt.show()
