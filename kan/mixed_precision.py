import torch
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def train_step(model, data, optimizer):
    inputs, targets = data
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(inputs)
        loss = torch.mean((outputs - targets)**2)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()
