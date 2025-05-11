from kan import FastKAN
import torch
import torch.nn as nn
from typing import *
import wandb
import time
from tqdm import tqdm
from torch.autograd import DeviceType
import os

class FastMixedKAN(FastKAN):
    
    def fit(self, dataset, opt="LBFGS", steps=100, log=1, lamb=0.1,
        loss_fn=None, lr=1.0, batch=-1, metrics=None, profile=False):
        '''
        Training function for FastMixedKAN

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
            scaler = torch.cuda.amp.GradScaler()
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
            train_input = dataset['train_input'][train_idx].to(device)
            train_label = dataset['train_label'][train_idx].to(device)
            if opt == "Adam":
                with torch.cuda.amp.autocast():
                    pred = self(train_input)
                    loss = loss_fn(pred, train_label)
                    if lamb > 0:
                        loss += lamb * sum(p.pow(2).sum() for p in self.parameters())
                scaler.scale(loss).backward()
            else:
                pred = self(train_input)
                loss = loss_fn(pred, train_label)
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
                scaler.step(optimizer)
                scaler.update()

            # Profiling
            if profile:
                prof.step()

            if step % log == 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=(opt == "Adam")):
                        test_pred = self(dataset['test_input'].to(device))
                        test_loss = loss_fn(test_pred, dataset['test_label'].to(device))

                results['train_loss'].append(torch.sqrt(loss.detach()).cpu().numpy())
                results['test_loss'].append(torch.sqrt(test_loss.detach()).cpu().numpy())

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

FastMPKAN = FastMixedKAN