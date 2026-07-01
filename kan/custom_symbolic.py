import torch
import torch.nn as nn
import numpy as np
import sympy
from .utils import *
from .Symbolic_KANLayer import Symbolic_KANLayer as Symbolic_KANLayer_raw
"""
Customization by Hyein Jung
"""

class Symbolic_KANLayer(Symbolic_KANLayer_raw):
    '''
    KANLayer class

    Attributes:
    -----------
        in_dim : int
            input dimension
        out_dim : int
            output dimension
        funs : 2D array of torch functions (or lambda functions)
            symbolic functions (torch)
        funs_avoid_singularity : 2D array of torch functions (or lambda functions) with singularity avoiding
        funs_name : 2D arry of str
            names of symbolic functions
        funs_sympy : 2D array of sympy functions (or lambda functions)
            symbolic functions (sympy)
        affine : 3D array of floats
            affine transformations of inputs and outputs
    '''

    def __init__(self, in_dim=3, out_dim=2, device='cpu'):
        super().__init__(in_dim=in_dim, out_dim=out_dim, device=device)
        self.funs_avoid_singularity = [[lambda x, y_th: (x * 0., (), x * 0.) for i in range(self.in_dim)] for j in
                                       range(self.out_dim)]
        # name
        self.funs_name = [['0' for i in range(self.in_dim)] for j in range(self.out_dim)]
        # sympy
        self.funs_sympy = [[lambda x: x * 0. for i in range(self.in_dim)] for j in range(self.out_dim)]
        ### make funs_name the only parameter, and make others as the properties of funs_name?

        self.affine = torch.nn.Parameter(torch.zeros(out_dim, in_dim, 4, device=device))
        # c*f(a*x+b)+d

        self.device = device
        self.to(device)

    def to(self, device):
        '''
        move to device
        '''
        super(Symbolic_KANLayer, self).to(device)
        self.device = device
        return self

    def forward(self, x, singularity_avoiding=False, y_th=10.):
        '''
        forward

        Args:
        -----
            x : 2D array
                inputs, shape (batch, input dimension)
            singularity_avoiding : bool
                if True, funs_avoid_singularity is used; if False, funs is used.
            y_th : float
                the singularity threshold

        Returns:
        --------
            y : 2D array
                outputs, shape (batch, output dimension)
            postacts : 3D array
                activations after activation functions but before being summed on nodes

        Example
        -------
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, postacts = sb(x)
        >>> y.shape, postacts.shape
        (torch.Size([100, 5]), torch.Size([100, 5, 3]))
        '''

        batch = x.shape[0]
        postacts = []

        for i in range(self.in_dim):
            postacts_ = []
            for j in range(self.out_dim):
                if singularity_avoiding:
                    xij = self.affine[j,i,2]*self.funs_avoid_singularity[j][i](self.affine[j,i,0]*x[:,[i]]+self.affine[j,i,1], torch.tensor(y_th))[-1]+self.affine[j,i,3]
                else:
                    xij = self.affine[j, i, 2] * self.funs[j][i](
                        self.affine[j, i, 0] * x[:, [i]] + self.affine[j, i, 1]) + self.affine[j, i, 3]
                postacts_.append(self.mask[j][i] * xij)
            postacts.append(torch.stack(postacts_))

        postacts = torch.stack(postacts)
        postacts = postacts.permute(2, 1, 0, 3)[:, :, :, 0]
        y = torch.sum(postacts, dim=2)

        return y, postacts

    def get_subset(self, in_id, out_id):
        '''
        get a smaller Symbolic_KANLayer from a larger Symbolic_KANLayer (used for pruning)

        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons

        Returns:
        --------
            spb : Symbolic_KANLayer

        Example
        -------
        >>> sb_large = Symbolic_KANLayer(in_dim=10, out_dim=10)
        >>> sb_small = sb_large.get_subset([0,9],[1,2,3])
        >>> sb_small.in_dim, sb_small.out_dim
        '''
        sbb = Symbolic_KANLayer(self.in_dim, self.out_dim, device=self.device)
        sbb.in_dim = len(in_id)
        sbb.out_dim = len(out_id)
        sbb.mask.data = self.mask.data[out_id][:, in_id]
        sbb.funs = [[self.funs[j][i] for i in in_id] for j in out_id]
        sbb.funs_avoid_singularity = [[self.funs_avoid_singularity[j][i] for i in in_id] for j in out_id]
        sbb.funs_sympy = [[self.funs_sympy[j][i] for i in in_id] for j in out_id]
        sbb.funs_name = [[self.funs_name[j][i] for i in in_id] for j in out_id]
        sbb.affine.data = self.affine.data[out_id][:, in_id]
        return sbb
