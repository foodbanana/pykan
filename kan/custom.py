from .MultKAN import *
from .MultKAN import MultKAN as MultKAN_raw
"""
Customization by Damdae Park
"""


def detach(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    elif isinstance(value, (tuple, list)):
        return type(value)(detach(v) for v in value)
    else:
        return value



class MultKAN(MultKAN_raw):
    def __init__(self,
                 width=None, grid=3, k=3, mult_arity=2, noise_scale=0.3,
                 scale_base_mu=0.0, scale_base_sigma=1.0, base_fun='silu',
                 symbolic_enabled=True, affine_trainable=False, grid_eps=0.02,
                 grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, seed=1,
                 save_act=True, sparse_init=False, auto_save=False,  # changed auto_save from True to False
                 first_init=True, ckpt_path='./model', state_id=0, round=0, device='cpu'):
        super().__init__(width=width, grid=grid, k=k, mult_arity=mult_arity,
                         noise_scale=noise_scale, scale_base_mu=scale_base_mu,
                         scale_base_sigma=scale_base_sigma, base_fun=base_fun,
                         symbolic_enabled=symbolic_enabled, affine_trainable=affine_trainable,
                         grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable,
                         sb_trainable=sb_trainable, seed=seed, save_act=save_act,
                         sparse_init=sparse_init, auto_save=auto_save,
                         first_init=first_init, ckpt_path=ckpt_path, state_id=state_id,
                         round=round, device=device)
        
    def log_history(self, method_name, verbose=True):  # added verbose=True
        if self.auto_save:
            # save to log file
            #print(func.__name__)
            with open(self.ckpt_path+'/history.txt', 'a') as file:
                file.write(str(self.round) + '.' + str(self.state_id)+' => '+ method_name + ' => ' 
                           + str(self.round) + '.' + str(self.state_id+1) + '\n')

            # update state_id
            self.state_id += 1

            # save to ckpt
            self.saveckpt(path=self.ckpt_path + '/' + str(self.round) + '.' + str(self.state_id))
            
            if verbose:
                print('saving model version ' + str(self.round) + '.' + str(self.state_id))


    def refine(self, new_grid, log_history=False):  # added log_history=False
        model_new = self.__class__(width=self.width,
                                   grid=new_grid,
                                   k=self.k,
                                   mult_arity=self.mult_arity,
                                   base_fun=self.base_fun_name,
                                   symbolic_enabled=self.symbolic_enabled,
                                   affine_trainable=self.affine_trainable,
                                   grid_eps=self.grid_eps,
                                   grid_range=self.grid_range,
                                   sp_trainable=self.sp_trainable,
                                   sb_trainable=self.sb_trainable,
                                   ckpt_path=self.ckpt_path,
                                   auto_save=True,
                                   first_init=False,
                                   state_id=self.state_id,
                                   round=self.round,
                                   device=self.device)

        model_new.initialize_from_another_model(self, self.cache_data)
        model_new.cache_data = self.cache_data
        model_new.grid = new_grid
        
        if log_history:
            self.log_history('refine')
        model_new.state_id += 1
        
        return model_new.to(self.device)
    
    
    def forward(self, x, singularity_avoiding=True, y_th=10.0):  # changed singularity_avoiding from False to True
        '''
        forward pass
        
        Args:
        -----
            x : 2D torch.tensor
                inputs
            singularity_avoiding : bool
                whether to avoid singularity for the symbolic branch
            y_th : float
                the threshold for singularity

        Returns:
        --------
            None
            
        Example1
        --------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        >>> x = torch.rand(100,2)
        >>> model(x).shape
        
        Example2
        --------
        >>> from kan import *
        >>> model = KAN(width=[1,1], grid=5, k=3, seed=0)
        >>> x = torch.tensor([[1],[-0.01]])
        >>> model.fix_symbolic(0,0,0,'log',fit_params_bool=False)
        >>> print(model(x))
        >>> print(model(x, singularity_avoiding=True))
        >>> print(model(x, singularity_avoiding=True, y_th=1.))
        '''
        x = x[:,self.input_id.long()]
        assert x.shape[1] == self.width_in[0]
        
        # cache data
        self.cache_data = x
        
        self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
        self.acts_premult = []
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_spline = []
        self.subnode_actscale = []
        self.edge_actscale = []
        # self.neurons_scale = []

        self.acts.append(x)  # acts shape: (batch, width[l])

        for l in range(self.depth):
            
            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)
            #print(preacts, postacts_numerical, postspline)
            
            if torch.isnan(x_numerical).any():  # added checker
                raise
            elif torch.isnan(preacts).any():
                raise
            elif torch.isnan(postacts_numerical).any():
                raise
            elif torch.isnan(postspline).any():
                raise
            else:
                pass
            
            if self.symbolic_enabled == True:
                x_symbolic, postacts_symbolic = self.symbolic_fun[l](x, singularity_avoiding=singularity_avoiding, y_th=y_th)
            else:
                x_symbolic = 0.
                postacts_symbolic = 0.

            x = x_numerical + x_symbolic
            
            if self.save_act:
                # save subnode_scale
                self.subnode_actscale.append(torch.std(x, dim=0).detach())
            
            # subnode affine transform
            x = self.subnode_scale[l][None,:] * x + self.subnode_bias[l][None,:]
            
            if self.save_act:
                postacts = postacts_numerical + postacts_symbolic

                # self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
                #grid_reshape = self.act_fun[l].grid.reshape(self.width_out[l + 1], self.width_in[l], -1)
                input_range = torch.std(preacts, dim=0) + 0.1
                output_range_spline = torch.std(postacts_numerical, dim=0) # for training, only penalize the spline part
                output_range = torch.std(postacts, dim=0) # for visualization, include the contribution from both spline + symbolic
                # save edge_scale
                self.edge_actscale.append(output_range)
                
                self.acts_scale.append((output_range / input_range).detach())
                self.acts_scale_spline.append(output_range_spline / input_range)
                self.spline_preacts.append(preacts.detach())
                self.spline_postacts.append(postacts.detach())
                self.spline_postsplines.append(postspline.detach())

                self.acts_premult.append(x.detach())
            
            # multiplication
            dim_sum = self.width[l+1][0]
            dim_mult = self.width[l+1][1]
            
            if self.mult_homo == True:
                for i in range(self.mult_arity-1):
                    if i == 0:
                        x_mult = x[:,dim_sum::self.mult_arity] * x[:,dim_sum+1::self.mult_arity]
                    else:
                        x_mult = x_mult * x[:,dim_sum+i+1::self.mult_arity]
                        
            else:
                for j in range(dim_mult):
                    acml_id = dim_sum + np.sum(self.mult_arity[l+1][:j])
                    for i in range(self.mult_arity[l+1][j]-1):
                        if i == 0:
                            x_mult_j = x[:,[acml_id]] * x[:,[acml_id+1]]
                        else:
                            x_mult_j = x_mult_j * x[:,[acml_id+i+1]]
                            
                    if j == 0:
                        x_mult = x_mult_j
                    else:
                        x_mult = torch.cat([x_mult, x_mult_j], dim=1)
                
            if self.width[l+1][1] > 0:
                x = torch.cat([x[:,:dim_sum], x_mult], dim=1)
            
            # x = x + self.biases[l].weight
            # node affine transform
            x = self.node_scale[l][None,:] * x + self.node_bias[l][None,:]
            
            self.acts.append(x.detach())
            
        
        return x
    
    
    def fix_symbolic(self, l, i, j, fun_name, fit_params_bool=True,
                     a_range=(-10, 10), b_range=(-10, 10),
                     verbose=True, random=False, log_history=False):  # changed log_history from True to False
        r2 = super().fix_symbolic(l, i, j, fun_name, fit_params_bool=fit_params_bool, 
                                  a_range=a_range, b_range=b_range, verbose=verbose,
                                  random=random, log_history=log_history)
        if log_history:
            self.log_history('fix_symbolic', verbose=verbose)
        
        try:
            return r2
        except NameError:
            return None


    def unfix_symbolic(self, l, i, j, log_history=False):  # changed log_history from True to False
        return super().unfix_symbolic(l, i, j, log_history=log_history)


    def unfix_symbolic_all(self, log_history=False):  # changed log_history from True to False
        return super().unfix_symbolic_all(log_history=log_history)
    
    
    def loadckpt(path='model'):
        instance = super().load_ckpt(path=path)
        return self.wrapping_methods(instance)
    
    def rewind(self, model_id):
        instance = super().rewind(model_id)
        return self.wrapping_methods(instance)
    
    def checkout(self, model_id):
        instance = super().checkout(model_id)
        return self.wrapping_methods(instance)
    
    
    def fit(self, dataset, opt="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1.,
            lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0., update_grid=True,
            grid_update_num=10, loss_fn=None, lr=1., start_grid_update_step=-1,
            stop_grid_update_step=50, batch=-1, metrics=None, save_fig=False,
            in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video',
            singularity_avoiding=True, y_th=1000., reg_metric='edge_forward_spline_n',
            display_metrics=None, monitor=True, log_history=False, shuffle=False,
            eval_test_loss=True):  # added monitor and eval_test_loss parameters
        
        '''
        training

        Args:
        -----
            dataset : dic
                contains dataset['train_input'], dataset['train_label'], dataset['test_input'], dataset['test_label']
            opt : str
                "LBFGS" or "Adam"
            steps : int
                training steps
            log : int
                logging frequency
            lamb : float
                overall penalty strength
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient magnitude penalty strength
            lamb_coefdiff : float
                difference of nearby coefficits (smoothness) penalty strength
            update_grid : bool
                If True, update grid regularly before stop_grid_update_step
            grid_update_num : int
                the number of grid updates before stop_grid_update_step
            start_grid_update_step : int
                no grid updates before this training step
            stop_grid_update_step : int
                no grid updates after this training step
            loss_fn : function
                loss function
            lr : float
                learning rate
            batch : int
                batch size, if -1 then full.
            save_fig_freq : int
                save figure every (save_fig_freq) steps
            singularity_avoiding : bool
                indicate whether to avoid singularity for the symbolic part
            y_th : float
                singularity threshold (anything above the threshold is considered singular and is softened in some ways)
            reg_metric : str
                regularization metric. Choose from {'edge_forward_spline_n', 'edge_forward_spline_u', 'edge_forward_sum', 'edge_backward', 'node_backward'}
            metrics : a list of metrics (as functions)
                the metrics to be computed in training
            display_metrics : a list of functions
                the metric to be displayed in tqdm bar
            monitor : bool
                whether to display training process in tqdm bar
            eval_test_loss : bool
                whether to evaluate test loss
                
        Returns:
        --------
            results : dic
                results['train_loss'], 1D array of training losses (RMSE)
                results['test_loss'], 1D array of test losses (RMSE)
                results['reg'], 1D array of regularization
                other metrics specified in metrics

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.3, seed=2)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=2)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.plot()
        # Most examples in toturals involve the fit() method. Please check them for useness.
        '''

        if lamb > 0. and not self.save_act:
            print('setting lamb=0. If you want to set lamb > 0, set self.save_act=True')
            
        old_save_act, old_symbolic_enabled = self.disable_symbolic_in_fit(lamb)

        if monitor:
            pbar = tqdm(range(steps), desc='description', ncols=100)
        else:
            pbar = range(steps)

        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        if opt == "AdamW":  # scheduling applied for better convergence; added
            bounds = (-10, 10)
            for name, weights in self.named_parameters():
                if weights.requires_grad:
                    weights.register_hook(lambda grad, bounds=bounds: torch.clamp(grad, bounds[0], bounds[1]))
                    
            optimizer = torch.optim.Adam(self.get_params(), lr=lr)
            scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.8, min_lr=1e-6)
        elif opt == "Adam":
            optimizer = torch.optim.Adam(self.get_params(), lr=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(self.get_params(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        results['reg'] = []
        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []

        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
            batch_size_test = dataset['test_input'].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch

        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding, y_th=y_th)
            train_loss = loss_fn(pred, dataset['train_label'][train_id])
            if self.save_act:
                if reg_metric == 'edge_backward':
                    self.attribute()
                if reg_metric == 'node_backward':
                    self.node_attribute()
                reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
            else:
                reg_ = torch.tensor(0.)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        for i in pbar:
            if i == steps-1 and old_save_act:
                self.save_act = True
                
            if save_fig and i % save_fig_freq == 0:
                save_act = self.save_act
                self.save_act = True
            
            if shuffle:
                train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
                test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)
            else:
                train_id = list(range(batch_size))
                test_id = list(range(batch_size_test))

            if i % grid_update_freq == 0 and i < stop_grid_update_step and update_grid and i >= start_grid_update_step:
                self.update_grid(dataset['train_input'][train_id])

            if opt in ["AdamW", "Adam"]:  # edited
                pred = self.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding, y_th=y_th)
                train_loss = loss_fn(pred, dataset['train_label'][train_id])
                if self.save_act:
                    if reg_metric == 'edge_backward':
                        self.attribute()
                    if reg_metric == 'node_backward':
                        self.node_attribute()
                    reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
                else:
                    reg_ = torch.tensor(0.)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if opt == "AdamW":  # added
                    scheduler.step(loss)
            elif opt == "LBFGS":
                optimizer.step(closure)

            if eval_test_loss:  # added
                test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id]), dataset['test_label'][test_id])
            else:
                test_loss = torch.Tensor(np.array(0.0))  # dummy
            
            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
            results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())

            if monitor and i % log == 0:  # added monitor parameter
                if display_metrics == None:
                    pbar.set_description("| train_loss: %.2e | test_loss: %.2e | reg: %.2e | " % (torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(), reg_.cpu().detach().numpy()))
                else:
                    string = ''
                    data = ()
                    for metric in display_metrics:
                        string += f' {metric}: %.2e |'
                        try:
                            results[metric]
                        except:
                            raise Exception(f'{metric} not recognized')
                        data += (results[metric][-1],)
                    pbar.set_description(string % data)
                    
            if save_fig and i % save_fig_freq == 0:
                self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(i), beta=beta)
                plt.savefig(img_folder + '/' + str(i) + '.jpg', bbox_inches='tight', dpi=200)
                plt.close()
                self.save_act = save_act
            
            
        if log_history:
            self.log_history('fit')
            
        # revert back to original state
        self.symbolic_enabled = old_symbolic_enabled
        return results
    
    
    def wrapping_methods(self, instance):
        instance.__class__ = self.__class__  # transform parent instance to child instance
        return instance
    
    
    def prune_node(self, threshold=1e-2, mode="auto", active_neurons_id=None, log_history=False):
        instance = super().prune_node(threshold=threshold, mode=mode, active_neurons_id=active_neurons_id, log_history=log_history)
        return self.wrapping_methods(instance)
    
    
    def prune_edge(self, threshold=3e-2, log_history=False):
        return super().prune_edge(threshold=threshold, log_history=log_history)
    
    
    def prune(self, node_th=1e-2, edge_th=3e-2, log_history=False):
        if self.acts == None:
            self.get_act()
        
        self = self.prune_node(node_th, log_history=False)
        #self.prune_node(node_th, log_history=False)
        self.forward(self.cache_data)
        self.attribute()
        self.prune_edge(edge_th, log_history=False)
        
        if log_history:
            self.log_history('prune')
        return self
    
    
    def prune_input(self, threshold=1e-2, active_inputs=None, log_history=False):
        '''
        prune inputs

        Args:
        -----
            threshold : float
                if the attribution score of the input feature is below threshold, it is considered irrelevant.
            active_inputs : None or list
                if a list is passed, the manual mode will disregard attribution score and prune as instructed.
            
        Returns:
        --------
            pruned network : MultKAN

        Example1
        --------
        >>> # automatic
        >>> from kan import *
        >>> model = KAN(width=[3,5,1], grid=5, k=3, noise_scale=0.3, seed=2)
        >>> f = lambda x: 1 * x[:,[0]]**2 + 0.3 * x[:,[1]]**2 + 0.0 * x[:,[2]]**2
        >>> dataset = create_dataset(f, n_var=3)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.plot()
        >>> model = model.prune_input()
        >>> model.plot()
        
        Example2
        --------
        >>> # automatic
        >>> from kan import *
        >>> model = KAN(width=[3,5,1], grid=5, k=3, noise_scale=0.3, seed=2)
        >>> f = lambda x: 1 * x[:,[0]]**2 + 0.3 * x[:,[1]]**2 + 0.0 * x[:,[2]]**2
        >>> dataset = create_dataset(f, n_var=3)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.plot()
        >>> model = model.prune_input(active_inputs=[0,1])
        >>> model.plot()
        '''
        if active_inputs == None:
            self.attribute()
            input_score = self.node_scores[0]
            input_mask = input_score > threshold
            print('keep:', input_mask.tolist())
            input_id = torch.where(input_mask==True)[0]
            
        else:
            input_id = torch.tensor(active_inputs, dtype=torch.long).to(self.device)
        
        model2 = MultKAN(copy.deepcopy(self.width), grid=self.grid, k=self.k, base_fun=self.base_fun, mult_arity=self.mult_arity, ckpt_path=self.ckpt_path, auto_save=True, first_init=False, state_id=self.state_id, round=self.round).to(self.device)
        model2.load_state_dict(self.state_dict())

        model2.act_fun[0] = model2.act_fun[0].get_subset(input_id, torch.arange(self.width_out[1]))
        model2.symbolic_fun[0] = self.symbolic_fun[0].get_subset(input_id, torch.arange(self.width_out[1]))

        model2.cache_data = self.cache_data
        model2.acts = None

        model2.width[0] = [len(input_id), 0]
        model2.input_id = input_id
        
        if log_history:
            self.log_history('prune_input')
        
        model2.state_id += 1  # unindented
        return model2
    

    def remove_edge(self, l, i, j, log_history=False):
        return super().remove_edge(l, i, j, log_history=log_history)
    
    
    def remove_node(self, l, i, mode='all', log_history=False):
        return super().remove_node(l, i, mode=mode, log_history=log_history)
    
    
    def auto_symbolic(self, a_range=(-10, 10), b_range=(-10, 10), lib=None,
                      verbose=1, weight_simple=0.8, r2_threshold=0.0, log_history=False):
        '''
        automatic symbolic regression for all edges

        Args:
        -----
            a_range : tuple
                search range of a
            b_range : tuple
                search range of b
            lib : list of str
                library of candidate symbolic functions
            verbose : int
                larger verbosity => more verbosity
            weight_simple : float
                a weight that prioritizies simplicity (low complexity) over performance (high r2) - set to 0.0 to ignore complexity
            r2_threshold : float
                If r2 is below this threshold, the edge will not be fixed with any symbolic function - set to 0.0 to ignore this threshold
        Returns:
        --------
            None

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,1,1], grid=5, k=3, noise_scale=0.0, seed=0)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]])+x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=3)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.auto_symbolic()
        '''
        
        for l in range(len(self.width_in) - 1):
            for i in range(self.width_in[l]):
                for j in range(self.width_out[l + 1]):
                    if self.symbolic_fun[l].mask[j, i] > 0. and self.act_fun[l].mask[i][j] == 0.:
                        if verbose >= 1:
                            print(f'skipping ({l},{i},{j}) since already symbolic')
                    elif self.symbolic_fun[l].mask[j, i] == 0. and self.act_fun[l].mask[i][j] == 0.:
                        self.fix_symbolic(l, i, j, '0', verbose=verbose, log_history=False)
                        if verbose >= 1:
                            print(f'fixing ({l},{i},{j}) with 0')
                    else:
                        name, fun, r2, c = self.suggest_symbolic(l, i, j, a_range=a_range, b_range=b_range, lib=lib, verbose=False, weight_simple=weight_simple)
                        
                        if r2 >= r2_threshold:
                            self.fix_symbolic(l, i, j, name, verbose=verbose, log_history=False)
                            if verbose >= 1:
                                print(f'fixing ({l},{i},{j}) with {name}, r2={round(r2, 2)}, c={c}')
                        else:
                            if verbose >= 1:
                                print(f'For ({l},{i},{j}) the best fit was {name}, but r^2 = {r2} and this is lower than {r2_threshold}. This edge was omitted, keep training or try a different threshold.')
        
        if log_history:
            self.log_history('auto_symbolic', verbose=verbose)
    

    def symbolic_formula(self, var=None, normalizer=None, output_normalizer=None, simplify=False):
        '''
        get symbolic formula

        Args:
        -----
            var : None or a list of sp expression
                input variables
            normalizer : [mean, std]
            output_normalizer : [mean, std]
            
        Returns:
        --------
            None

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,1,1], grid=5, k=3, noise_scale=0.0, seed=0)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]])+x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=3)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.auto_symbolic()
        >>> model.symbolic_formula()[0][0]
        '''
        
        symbolic_acts = []
        symbolic_acts_premult = []
        x = []

        def ex_round(ex1, n_digit):
            ex2 = ex1
            for a in sympy.preorder_traversal(ex1):
                if isinstance(a, sympy.Float):
                    ex2 = ex2.subs(a, round(a, n_digit))
            return ex2

        # define variables
        if var == None:
            for ii in range(1, self.width[0][0] + 1):
                exec(f"x{ii} = sympy.Symbol('x_{ii}')")
                exec(f"x.append(x{ii})")
        elif isinstance(var[0], sympy.Expr):
            x = var
        else:
            x = [sympy.symbols(var_) for var_ in var]

        x0 = x

        if normalizer != None:
            mean = normalizer[0]
            std = normalizer[1]
            x = [(x[i] - mean[i]) / std[i] for i in range(len(x))]

        symbolic_acts.append(x)

        for l in range(len(self.width_in) - 1):
            num_sum = self.width[l + 1][0]
            num_mult = self.width[l + 1][1]
            y = []
            for j in range(self.width_out[l + 1]):
                yj = 0.
                for i in range(self.width_in[l]):
                    a, b, c, d = self.symbolic_fun[l].affine[j, i]
                    sympy_fun = self.symbolic_fun[l].funs_sympy[j][i]
                    
                    (a,b,c,d) = detach((a, b, c, d))
                    
                    try:
                        yj += c * sympy_fun(a * x[i] + b) + d  # here
                    except:
                        print('make sure all activations need to be converted to symbolic formulas first!')
                        return
                yj = self.subnode_scale[l][j] * yj + self.subnode_bias[l][j]
                
                if simplify == True:
                    y.append(sympy.simplify(yj))  #! takes too long
                else:
                    y.append(yj)
                    
            symbolic_acts_premult.append(y)
                  
            mult = []
            for k in range(num_mult):
                if isinstance(self.mult_arity, int):
                    mult_arity = self.mult_arity
                else:
                    mult_arity = self.mult_arity[l+1][k]
                for i in range(mult_arity-1):
                    if i == 0:
                        mult_k = y[num_sum+2*k] * y[num_sum+2*k+1]
                    else:
                        mult_k = mult_k * y[num_sum+2*k+i+1]
                mult.append(mult_k)
                
            y = y[:num_sum] + mult
            
            for j in range(self.width_in[l+1]):
                y[j] = self.node_scale[l][j] * y[j] + self.node_bias[l][j]
            
            x = y
            symbolic_acts.append(x)

        if output_normalizer != None:
            output_layer = symbolic_acts[-1]
            means = output_normalizer[0]
            stds = output_normalizer[1]

            assert len(output_layer) == len(means), 'output_normalizer does not match the output layer'
            assert len(output_layer) == len(stds), 'output_normalizer does not match the output layer'
            
            output_layer = [(output_layer[i] * stds[i] + means[i]) for i in range(len(output_layer))]
            symbolic_acts[-1] = output_layer


        self.symbolic_acts = [[symbolic_acts[l][i] for i in range(len(symbolic_acts[l]))] for l in range(len(symbolic_acts))]
        self.symbolic_acts_premult = [[symbolic_acts_premult[l][i] for i in range(len(symbolic_acts_premult[l]))] for l in range(len(symbolic_acts_premult))]

        out_dim = len(symbolic_acts[-1])
        #return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0
        
        return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0
    
    
    def swap(self, l, i1, i2, log_history=False):
        return super().swap(l, i1, i2, log_history=log_history)


    def eval_complexity(self, threshold=0.0):  # added
        complexity = 0
        for l in range(len(self.width_in) - 1):
            for i in range(self.width_in[l]):
                for j in range(self.width_out[l + 1]):
                    num_active = (self.act_fun[l].mask[i][j] > threshold)
                    sym_active = (
                        (self.symbolic_fun[l].mask[j, i] > threshold) and
                        (self.symbolic_fun[l].funs_name[j][i] != "0")
                    )
                    if num_active or sym_active:
                        complexity += 1
        return complexity
