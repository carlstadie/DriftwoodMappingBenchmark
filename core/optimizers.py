# core/optimizers.py (PyTorch)
from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer


class WarmUpCosineDecay:
    """
    Warmup to base_lr, then cosine decay to base_lr * final_lr_scale.
    Linear warmup + cosine decay similar to TF CosineDecay(alpha=final_lr_scale).
    """

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        warmup_steps: int = 1000,
        final_lr_scale: float = 0.1,
    ):
        self.base_lr = float(base_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.final_lr_scale = float(final_lr_scale)

        self.decay_steps = max(1, self.total_steps - self.warmup_steps)
        self.alpha = self.final_lr_scale

    def __call__(self, step: int) -> float:
        step = int(step)

        # Linear warmup
        if step < self.warmup_steps:
            return float(self.base_lr * (step + 1) / max(1, self.warmup_steps))

        # Cosine decay to alpha * base_lr
        t = step - self.warmup_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * t / self.decay_steps))
        lr = self.base_lr * (self.alpha + (1.0 - self.alpha) * cosine)
        return float(lr)


class _AdamWithHooks(optim.Adam):
    """
    Adam with optional global-norm clipping and optional internal per-step LR schedule.
    IMPORTANT: This does NOT change LR unless schedule is provided.
    """

    def __init__(
        self,
        params,
        lr: float,
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
            lr=float(lr),
            betas=betas,
            eps=float(eps),
            weight_decay=float(weight_decay),
            amsgrad=amsgrad,
        )
        self._global_step = 0
        self._clipnorm = float(clipnorm) if (clipnorm is not None and float(clipnorm) > 0) else None
        self._schedule = schedule

    @torch.no_grad()
    def step(self, closure=None):
        # Only touch LR if schedule is explicitly provided
        if self._schedule is not None:
            new_lr = self._schedule(self._global_step)
            for group in self.param_groups:
                group["lr"] = new_lr

        # Optional clipping (if you enable it here; don’t double-clip in training loop)
        if self._clipnorm is not None:
            for group in self.param_groups:
                params = [p for p in group["params"] if p.grad is not None]
                if params:
                    clip_grad_norm_(params, max_norm=self._clipnorm)

        out = super().step(closure)
        self._global_step += 1
        return out


class _NAdamWithClip(optim.NAdam):
    """NAdam with optional global-norm clipping (no internal schedule)."""

    def __init__(
        self,
        params,
        lr: float,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        clipnorm: Optional[float] = None,
    ):
        super().__init__(
            params,
            lr=float(lr),
            betas=betas,
            eps=float(eps),
            weight_decay=float(weight_decay),
        )
        self._clipnorm = float(clipnorm) if (clipnorm is not None and float(clipnorm) > 0) else None

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
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        *,
        clipnorm: Optional[float] = None,
    ):
        super().__init__(
            params,
            lr=float(lr),
            rho=float(rho),
            eps=float(eps),
            weight_decay=float(weight_decay),
        )
        self._clipnorm = float(clipnorm) if (clipnorm is not None and float(clipnorm) > 0) else None

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
            lr=float(lr),
            lr_decay=float(lr_decay),
            weight_decay=float(weight_decay),
            initial_accumulator_value=float(initial_accumulator_value),
            eps=float(eps),
        )
        self._clipnorm = float(clipnorm) if (clipnorm is not None and float(clipnorm) > 0) else None

    @torch.no_grad()
    def step(self, closure=None):
        if self._clipnorm is not None:
            for group in self.param_groups:
                params = [p for p in group["params"] if p.grad is not None]
                if params:
                    clip_grad_norm_(params, max_norm=self._clipnorm)
        return super().step(closure)


def _infer_weight_decay(model) -> float:
    """Fallback if caller doesn't pass weight_decay explicitly."""
    if model is None:
        return 0.0
    wd = getattr(model, "l2_weight", 0.0)
    try:
        return float(wd)
    except Exception:
        return 0.0


def get_optimizer(
    optimizer_fn: Union[str, Optimizer],
    num_epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    model=None,
    *,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    clipnorm: Optional[float] = None,
    # Off by default. Enable ONLY if you want the optimizer to own LR scheduling.
    internal_schedule: Optional[str] = None,  # "warmup_cosine"
    warmup_steps: int = 1000,
    final_lr_scale: float = 0.1,
) -> Optimizer:
    """
    Optimizer factory that respects config-provided lr / weight_decay / clipnorm.

    Default behavior:
      - No internal LR schedule (so external schedulers like OneCycleLR work correctly)
      - No implicit clipnorm (so training.py can control it via config.clip_norm)

    To enable internal warmup+cosine (not recommended if you already use OneCycleLR):
      internal_schedule="warmup_cosine"
    """

    # If user passes a pre-built optimizer, return it
    if isinstance(optimizer_fn, Optimizer):
        return optimizer_fn

    if model is None:
        raise ValueError("get_optimizer(...): 'model' must be provided when optimizer_fn is a string.")

    name = str(optimizer_fn).strip().lower()

    # Respect explicit values; otherwise fallback
    base_lr = float(lr) if lr is not None else None
    wd = float(weight_decay) if weight_decay is not None else _infer_weight_decay(model)
    cn = float(clipnorm) if (clipnorm is not None and float(clipnorm) > 0) else None

    # Optional internal schedule
    schedule = None
    if internal_schedule is not None:
        sched_name = str(internal_schedule).strip().lower()
        if sched_name in ("warmup_cosine", "warmupcosine", "warmup+cosine"):
            if base_lr is None:
                raise ValueError("internal_schedule requires lr=... (base_lr).")
            if not (num_epochs and steps_per_epoch):
                raise ValueError("warmup_cosine needs num_epochs and steps_per_epoch.")
            total_steps = int(num_epochs) * int(steps_per_epoch)
            schedule = WarmUpCosineDecay(
                base_lr=base_lr,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                final_lr_scale=final_lr_scale,
            )
        else:
            raise ValueError(f"Unknown internal_schedule: {internal_schedule}")

    # ---- Optimizers ----
    if name in ("adam", "adam1"):
        if base_lr is None:
            base_lr = 3e-4
        # "adam1" historically = same Adam but no schedule
        schedule_for_adam = schedule if name == "adam" else None
        return _AdamWithHooks(
            model.parameters(),
            lr=base_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
            clipnorm=cn,
            schedule=schedule_for_adam,
        )

    if name == "adamw":
        if base_lr is None:
            base_lr = 3e-4
        # No hidden schedule/clipping; use training loop + torch schedulers
        return optim.AdamW(
            model.parameters(),
            lr=base_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
        )

    if name in ("adadelta", "adadelta".lower(), "adadelta".upper(), "adadelta".title(), "adadelta".capitalize(), "adadelta".swapcase(), "adadelta"):
        if base_lr is None:
            base_lr = 1.0
        return _AdadeltaWithClip(
            model.parameters(),
            lr=base_lr,
            rho=0.95,
            eps=1e-7,
            weight_decay=wd,
            clipnorm=cn,
        )

    if name == "nadam":
        if base_lr is None:
            base_lr = 2e-3
        return _NAdamWithClip(
            model.parameters(),
            lr=base_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=wd,
            clipnorm=cn,
        )

    if name == "adagrad":
        if base_lr is None:
            base_lr = 1e-2
        return _AdagradWithClip(
            model.parameters(),
            lr=base_lr,
            lr_decay=0.0,
            weight_decay=wd,
            initial_accumulator_value=0.0,
            eps=1e-10,
            clipnorm=cn,
        )

    raise ValueError(f"Unknown optimizer name: {optimizer_fn}")
