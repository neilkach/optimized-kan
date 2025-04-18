from kan import MultKAN
import torch
import numpy as np
from .LBFGS import *
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

class MixedPrecisionKAN(MultKAN):

    def fit(self, dataset, opt="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1., start_grid_update_step=-1, stop_grid_update_step=50, batch=-1,
              metrics=None, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video', singularity_avoiding=False, y_th=1000., reg_metric='edge_forward_spline_n', display_metrics=None, profile=False):
        '''
        training with mixed precision support
        '''

        if lamb > 0. and not self.save_act:
            print('setting lamb=0. If you want to set lamb > 0, set self.save_act=True')
            
        old_save_act, old_symbolic_enabled = self.disable_symbolic_in_fit(lamb)

        pbar = tqdm(range(steps), desc='description', ncols=100)

        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        if profile:
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=3,
                    repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_logs'),
                record_shapes=True,
                with_stack=True,
                profile_memory=True
            )
            prof.start()

        scaler = torch.cuda.amp.GradScaler(enabled=(opt == "Adam"))
        
        if opt == "Adam":
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
            
            if opt == "Adam":
                with torch.cuda.amp.autocast():
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
                
                scaler.scale(objective).backward()
            else:
                # For LBFGS, use regular precision
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

        for _ in pbar:
            
            if _ == steps-1 and old_save_act:
                self.save_act = True
                
            if save_fig and _ % save_fig_freq == 0:
                save_act = self.save_act
                self.save_act = True
            
            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid and _ >= start_grid_update_step:
                self.update_grid(dataset['train_input'][train_id])

            if opt == "LBFGS":
                optimizer.step(closure)
                
            if opt == "Adam":
                with torch.cuda.amp.autocast():
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
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if profile:
                prof.step()

            with torch.no_grad():
                test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id]), dataset['test_label'][test_id])
            
            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
            results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())

            if _ % log == 0:
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
                    
            if save_fig and _ % save_fig_freq == 0:
                self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_), beta=beta)
                plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                plt.close()
                self.save_act = save_act

        if profile:
            prof.stop()
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            
        self.log_history('fit')
        # revert back to original state
        self.symbolic_enabled = old_symbolic_enabled
        return results
    
MPKAN = MixedPrecisionKAN