# core/optimizers.py (PyTorch)
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer


class WarmUpCosineDecay:
    """
    Warmup to base_lr, then cosine decay to base_lr * final_lr_scale.
    Matches TF's CosineDecay with alpha=final_lr_scale and a linear warmup.

    LR(step) =
      if step < warmup_steps:
          base_lr * step / warmup_steps
      else:
          base_lr * [ alpha + (1 - alpha) * 0.5 * (1 + cos(pi * (step - warmup_steps) / decay_steps)) ]
    """

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        warmup_steps: int,
        final_lr_scale: float = 0.1,
    ) -> None:
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.alpha = float(final_lr_scale)
        self.decay_steps = max(1, self.total_steps - self.warmup_steps)

    def __call__(self, step: int) -> float:
        step = int(step)
        if step < self.warmup_steps:
            return self.base_lr * (step / max(1, self.warmup_steps))

        t = step - self.warmup_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * t / self.decay_steps))
        lr = self.base_lr * (self.alpha + (1.0 - self.alpha) * cosine)
        return float(lr)


class _AdamWithHooks(optim.Adam):
    """
    Adam that supports:
      - global-norm gradient clipping (clipnorm)
      - per-step LR scheduling via a callable schedule(step)->lr

    No changes are required in the training loop; the optimizer updates LR internally.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        *,
        clipnorm: Optional[float] = None,
        schedule: Optional[WarmUpCosineDecay] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        self._global_step = 0
        self._clipnorm = float(clipnorm) if clipnorm else None
        self._schedule = schedule

    @torch.no_grad()
    def step(self, closure=None):
        # Update learning rate from schedule (before applying the step)
        if self._schedule is not None:
            new_lr = self._schedule(self._global_step)
            for group in self.param_groups:
                group["lr"] = new_lr

        # Global-norm gradient clipping to match Keras clipnorm behavior
        if self._clipnorm is not None:
            for group in self.param_groups:
                params = [p for p in group["params"] if p.grad is not None]
                if params:
                    clip_grad_norm_(params, max_norm=self._clipnorm)

        out = super().step(closure)
        self._global_step += 1
        return out


class _NAdamWithClip(optim.NAdam):
    """NAdam with optional global-norm clipping."""

    def __init__(
        self,
        params,
        lr: float = 2e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        clipnorm: Optional[float] = None,
    ):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self._clipnorm = float(clipnorm) if clipnorm else None

    @torch.no_grad()
    def step(self, closure=None):
        if self._clipnorm is not None:
            for group in self.param_groups:
                params = [p for p in group["params"] if p.grad is not None]
                if params:
                    clip_grad_norm_(params, max_norm=self._clipnorm)
        return super().step(closure)


class _AdadeltaWithClip(optim.Adadelta):
    """Adadelta with optional global-norm clipping."""

    def __init__(
        self,
        params,
        lr: float = 1.0,
        rho: float = 0.95,
        eps: float = 1e-7,
        weight_decay: float = 0.0,
        *,
        clipnorm: Optional[float] = None,
    ):
        super().__init__(params, lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        self._clipnorm = float(clipnorm) if clipnorm else None

    @torch.no_grad()
    def step(self, closure=None):
        if self._clipnorm is not None:
            for group in self.param_groups:
                params = [p for p in group["params"] if p.grad is not None]
                if params:
                    clip_grad_norm_(params, max_norm=self._clipnorm)
        return super().step(closure)


class _AdagradWithClip(optim.Adagrad):
    """Adagrad with optional global-norm clipping."""

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
        *,
        clipnorm: Optional[float] = None,
    ):
        super().__init__(
            params,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
        )
        self._clipnorm = float(clipnorm) if clipnorm else None

    @torch.no_grad()
    def step(self, closure=None):
        if self._clipnorm is not None:
            for group in self.param_groups:
                params = [p for p in group["params"] if p.grad is not None]
                if params:
                    clip_grad_norm_(params, max_norm=self._clipnorm)
        return super().step(closure)


def _infer_weight_decay(model) -> float:
    """Use model.l2_weight if available (UNetAttention sets this)."""
    if model is None:
        return 0.0
    wd = getattr(model, "l2_weight", 0.0)
    try:
        return float(wd)
    except Exception:
        return 0.0


def get_optimizer(
    optimizer_fn,
    num_epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    model=None,
) -> Optimizer:
    """
    Wrapper to allow dynamic optimizer setup with optional learning rate schedule.
    Mirrors the original Keras behavior and defaults.

    Args:
        optimizer_fn: one of {"adam", "adam1", "adaDelta", "nadam", "adagrad"}
                      or an Optimizer instance.
        num_epochs: total epochs for scheduling (optional).
        steps_per_epoch: steps per epoch for scheduling (optional).
        model: torch.nn.Module; if it defines 'l2_weight', it's used as weight_decay.

    Returns:
        torch.optim.Optimizer
    """
    # If user passes a pre-built optimizer, return it
    if isinstance(optimizer_fn, Optimizer):
        return optimizer_fn

    weight_decay = _infer_weight_decay(model)

    # Adam with optional warmup+cosine and clipnorm=1.0
    if optimizer_fn == "adam":
        base_lr = 3e-4
        schedule = None
        if num_epochs and steps_per_epoch:
            total_steps = int(num_epochs) * int(steps_per_epoch)
            schedule = WarmUpCosineDecay(
                base_lr=base_lr,
                total_steps=total_steps,
                warmup_steps=1000,
                final_lr_scale=0.1,
            )
        return _AdamWithHooks(
            model.parameters() if model is not None else [],
            lr=base_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
            clipnorm=1.0,
            schedule=schedule,
        )

    elif optimizer_fn == "adam1":
        # same hyperparams as your predefined 'adam'
        return _AdamWithHooks(
            model.parameters() if model is not None else [],
            lr=3e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
            clipnorm=1.0,
            schedule=None,
        )

    elif optimizer_fn == "adaDelta":
        return _AdadeltaWithClip(
            model.parameters() if model is not None else [],
            lr=1.0,
            rho=0.95,
            eps=1e-7,
            weight_decay=0.0,
            clipnorm=None,
        )

    elif optimizer_fn == "nadam":
        # PyTorch NAdam doesn't have schedule_decay; default behavior is fine.
        return _NAdamWithClip(
            model.parameters() if model is not None else [],
            lr=2e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
            clipnorm=None,
        )

    elif optimizer_fn == "adagrad":
        return _AdagradWithClip(
            model.parameters() if model is not None else [],
            lr=1e-2,
            lr_decay=0.0,
            weight_decay=0.0,
            initial_accumulator_value=0.0,
            eps=1e-10,
            clipnorm=None,
        )

    # Fallback: if a string we don't know, raise, or if it's an object, return it
    if isinstance(optimizer_fn, str):
        raise ValueError(f"Unknown optimizer name: {optimizer_fn}")
    return optimizer_fn
