from .LBFGS import LBFGS as LBFGS_raw
import torch

class LBFGS(LBFGS_raw):
    def __init__(self,
                 params,
                 lr=1,
                 max_iter=20,
                 max_eval=None,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-9,
                 tolerance_ys=1e-32,
                 history_size=100,
                 line_search_fn=None):
        super().__init__(
            params, lr, max_iter, max_eval, tolerance_grad, tolerance_change, tolerance_ys, history_size, line_search_fn
        )

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if torch.isnan(p).any():
                raise
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        device = views[0].device
        return torch.cat(views, dim=0)

