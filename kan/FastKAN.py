import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *
import wandb
import time
from tqdm import tqdm
from torch.autograd import DeviceType
import os

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        num_extrapolate_bins: int = 2
    ):
        '''this function returns the learned curves in a FastKANLayer.
        input_index: the selected index of the input, in [0, input_dim) .
        output_index: the selected index of the output, in [0, output_dim) .
        num_pts: num of points sampled for the curve.
        num_extrapolate_bins (N_e): num of bins extrapolating from the given grids. The curve
            will be calculate in the range of [grid_min - h * N_e, grid_max + h * N_e].
        '''
        ng = self.rbf.num_grids
        h = self.rbf.denominator
        assert input_index < self.input_dim
        assert output_index < self.output_dim
        w = self.spline_linear.weight[
            output_index, input_index * ng : (input_index + 1) * ng
        ]   # num_grids,
        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts
        )   # num_pts, num_grids
        with torch.no_grad():
            y = (w * self.rbf(x.to(w.dtype))).sum(-1)
        return x, y


class FastKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
        use_layernorm = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FastKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
                use_layernorm=use_layernorm,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



    def fit(self, dataset, opt="LBFGS", steps=100, log=1, lamb=0.1,
        loss_fn=None, lr=1.0, batch=-1, metrics=None, profile=False):
        '''
        Training function for FastKAN

        Args:
            dataset (dict): Contains 'train_input', 'train_label', 'test_input', 'test_label'
            opt (str): Optimizer ('Adam' or 'LBFGS')
            steps (int): Number of training steps
            log (int): Logging frequency
            lamb (float): L2 regularization strength
            loss_fn (callable): Custom loss function (default: MSELoss)
            lr (float): Learning rate
            batch (int): Batch size (-1 for full batch)
            metrics (list): List of metric functions
            profile (bool): Enable profiling
        '''

        # Initialize loss and optimizer
        device = next(self.parameters()).device
        loss_fn = loss_fn or nn.MSELoss()

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        elif opt == "LBFGS":
            optimizer = torch.optim.LBFGS(self.parameters(), lr=lr, history_size=10,
                                        line_search_fn="strong_wolfe")

        pbar = tqdm(range(steps), desc='description', ncols=100)

        # Profiling setup
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
                # on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_logs'),
                record_shapes=True,
                with_stack=True,
                profile_memory=True
            )
            prof.start()

        # Batch handling
        batch_size = batch if batch > 0 else dataset['train_input'].shape[0]
        results = {'train_loss': [], 'test_loss': []}
        if metrics:
            for metric in metrics:
                results[metric.__name__] = []

        def closure():
            optimizer.zero_grad()
            pred = self(dataset['train_input'][train_idx].to(device))
            loss = loss_fn(pred, dataset['train_label'][train_idx].to(device))
            if lamb > 0:
                loss += lamb * sum(p.pow(2).sum() for p in self.parameters())
            loss.backward()
            return loss

        for step in pbar:
            # Batch selection
            start_time = time.perf_counter()
            train_idx = torch.randperm(dataset['train_input'].shape[0])[:batch_size]

            # Optimization step
            if opt == "LBFGS":
                optimizer.step(closure)
            else:  # Adam
                loss = closure()
                optimizer.step()

            # Profiling
            if profile:
                prof.step()

            # Logging
            if step % log == 0:
                with torch.no_grad():
                    test_pred = self(dataset['test_input'].to(device))
                    test_loss = loss_fn(test_pred, dataset['test_label'].to(device))

                results['train_loss'].append(torch.sqrt(loss).cpu().detach().numpy())
                results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())

                exec_time = time.perf_counter() - start_time
                wandb.log({
                    "step": step,
                    "train_loss": float(torch.sqrt(loss)),
                    "val_loss": float(torch.sqrt(test_loss)),
                    # "reg": float(reg_),
                    "exec_time": exec_time
                })

                if metrics:
                    for metric in metrics:
                        results[metric.__name__].append(
                            metric(test_pred, dataset['test_label'].to(device)).item()
                        )

                pbar.set_description("| train_loss: %.2e | test_loss: %.2e | " % (torch.sqrt(loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy()))

        if profile:
            prof.stop()
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            sum_self_cpu_time_total = 0
            sum_cpu_time_total = 0
            sum_self_device_time_total = 0
            sum_device_time_total = 0
            for evt in prof.key_averages():
                sum_self_cpu_time_total += evt.self_cpu_time_total
                sum_cpu_time_total += evt.cpu_time_total
                if evt.device_type == DeviceType.CPU and evt.is_legacy:
                    # in legacy profiler, kernel info is stored in cpu events
                    sum_self_device_time_total += evt.self_device_time_total
                    sum_device_time_total += evt.device_time_total
                elif (
                    evt.device_type
                    in [
                        DeviceType.CUDA,
                        DeviceType.PrivateUse1,
                        DeviceType.MTIA,
                    ]
                    and not evt.is_user_annotation
                ):
                    # in kineto profiler, there're events with the correct device type (e.g. CUDA)
                    sum_self_device_time_total += evt.self_device_time_total
                    sum_device_time_total += evt.device_time_total

            # Log to wandb
            wandb.log({
                "cpu_time_total (ms)": sum_cpu_time_total / 1000,
                "cuda_time_total (ms)": sum_device_time_total / 1000,
                "self_cpu_time_total (ms)": sum_self_cpu_time_total / 1000,
                "self_cuda_time_total (ms)": sum_self_device_time_total / 1000
            })

            trace_path = os.path.join(".", "trace.json")
            prof.export_chrome_trace(trace_path)
            print('Saved to', trace_path)
            wandb.save(trace_path)
            print('Saved', trace_path, 'to W&B')

        return results


