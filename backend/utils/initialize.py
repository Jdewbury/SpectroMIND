import torch
from sam import SAM

def get_optimizer(model, optimizer_name, learning_rate, base_optimizer_name=None, rho=None, weight_decay=None):
    base_optimizers = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam
    }
    if base_optimizer_name:
        if base_optimizer_name not in base_optimizers:
            raise ValueError(f'Unsupported base optimizer: {base_optimizer_name}')
        
        base_optimizer = base_optimizers[base_optimizer_name]
        print(f'Base Optimizer: {base_optimizer_name}')
    else: 
        base_optimizer = None

    optimizers = {
        'Adam': lambda: torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=weight_decay),
        'SGD': lambda: torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        'SAM': lambda: SAM(model.parameters(), base_optimizer, rho=rho, adaptive=False, lr=learning_rate, weight_decay=weight_decay),
        'ASAM': lambda: SAM(model.parameters(), base_optimizer, rho=rho, adaptive=True, lr=learning_rate, weight_decay=weight_decay)
    }

    if optimizer_name not in optimizers:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')

    optimizer = optimizers[optimizer_name]()
    print(f'Optimizer: {optimizer_name}')

    return optimizer, base_optimizer

def get_scheduler(scheduler_name, optimizer, total_epochs, gamma=None, step_size=None, T_max=None):
    schedulers = {
        'step': lambda: torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
        'cosine': lambda: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)
    }
    if scheduler_name not in schedulers:
        raise ValueError(f'Unsupported scheduler: {scheduler_name}')
    
    scheduler = schedulers[scheduler_name]()
    print(f'Scheduler: {scheduler_name}')

    return scheduler