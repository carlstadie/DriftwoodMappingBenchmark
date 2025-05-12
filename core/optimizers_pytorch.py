import torch

# Pre‐configured optimizer defaults
adaDelta = torch.optim.Adadelta  # call with (params, lr=1.0, rho=0.95, eps=1e-7)
adam     = torch.optim.Adam      # call with (params, lr=3e-4, betas=(0.9,0.999), eps=1e-8, amsgrad=False)
nadam    = torch.optim.NAdam     # call with (params, lr=2e-3, betas=(0.9,0.999), eps=1e-8, momentum_decay=4e-3)
adagrad  = torch.optim.Adagrad   # call with (params, lr=0.01)

def get_optimizer(optimizer_fn, params, num_epochs=None, steps_per_epoch=None):
    """
    Build optimizer (and optional LR scheduler) mirroring the TF/Keras setup.
    
    Args:
      optimizer_fn: one of "adam", "adaDelta", "nadam", "adagrad",
                    or a callable optimizer class.
      params:       iterable of model parameters (e.g. model.parameters()).
      num_epochs:   total number of epochs (for cosine schedule).
      steps_per_epoch: number of batches per epoch (for cosine schedule).
    
    Returns:
      If using cosine‐decay:  (optimizer, scheduler)
      Otherwise:              optimizer
    """
    if optimizer_fn == "adam":
        opt = torch.optim.Adam(
            params,
            lr=3e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=False
        )
        # optional cosine‐decay schedule to 10% of initial LR
        if num_epochs is not None and steps_per_epoch is not None:
            total_steps = num_epochs * steps_per_epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=total_steps,
                eta_min=3e-4 * 0.1
            )
            return opt, scheduler
        return opt

    elif optimizer_fn == "adaDelta":
        return torch.optim.Adadelta(
            params,
            lr=1.0,
            rho=0.95,
            eps=1e-7
        )

    elif optimizer_fn == "nadam":
        # PyTorch’s NAdam momentum_decay defaults to 4e-3, matching TF’s schedule_decay
        return torch.optim.NAdam(
            params,
            lr=0.002,
            betas=(0.9, 0.999),
            eps=1e-8,
            momentum_decay=0.004
        )

    elif optimizer_fn == "adagrad":
        return torch.optim.Adagrad(
            params,
            lr=0.01,
            eps=1e-10,
            lr_decay=0.0
        )

    else:
        # assume optimizer_fn is already a torch.optim.Optimizer subclass or instance
        return optimizer_fn(params)
