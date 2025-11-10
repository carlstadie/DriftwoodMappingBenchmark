# core/common/model_utils.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# project deps used by helpers
from core.Swin_UNetPP import SwinUNet
from core.TerraMind import TerraMind

# Optional runtime config (used for a few defaults if present)
config: Any = None


def set_global_seed(seed: Optional[int] = None):
    """Set global RNG seeds for reproducibility. If seed is None, do nothing."""
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic kernels (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModelEMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]
        self.backup = None

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @contextmanager
    @torch.no_grad()
    def use_ema_weights(self, model: nn.Module):
        """Temporarily swap model params with EMA weights."""
        self.backup = [p.detach().clone() for p in self.params]
        for p, s in zip(self.params, self.shadow):
            p.copy_(s)
        try:
            yield
        finally:
            for p, b in zip(self.params, self.backup):
                p.copy_(b)
            self.backup = None


def _required_multiple(patch_size: int, window: int, levels: int) -> int:
    """Return required multiple in IMAGE pixels for Swin-like hierarchies."""
    return int(patch_size) * (int(window) * (2 ** int(levels)))


def _autopad(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int, int]:
    """Pad right/bottom so H,W are multiples of 'multiple'."""
    H, W = x.shape[-2], x.shape[-1]
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, pad_h, pad_w


def _unpad(y: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_h or pad_w:
        y = y[..., : y.shape[-2] - pad_h, : y.shape[-1] - pad_w]
    return y


def _forward_with_autopad(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward pass with optional autopadding for Swin models (guarded by model type)."""
    base = getattr(model, "_orig_mod", model)  # unwrap compile wrappers
    if isinstance(base, SwinUNet):
        # only Swin needs strict spatial multiples
        swin_window = getattr(config, "swin_window", 4) if config is not None else 4
        swin_levels = getattr(config, "swin_levels", 3) if config is not None else 3
        swin_patch = getattr(config, "swin_patch_size", 16) if config is not None else 16
        multiple = _required_multiple(swin_patch, swin_window, swin_levels)
        x, ph, pw = _autopad(x, multiple)
        y = model(x)
        return _unpad(y, ph, pw)
    else:
        return model(x)


def _ensure_probabilities(y_pred: torch.Tensor) -> torch.Tensor:
    """
    Heuristic: if predictions look like logits (outside [0,1] by margin), apply sigmoid.
    Else assume probabilities and clamp to [0,1].
    """
    with torch.no_grad():
        mn = float(y_pred.min().detach().cpu())
        mx = float(y_pred.max().detach().cpu())
    if (mn < -1e-3) or (mx > 1.0 + 1e-3):
        return torch.sigmoid(y_pred)
    return y_pred.clamp(0.0, 1.0)


def _ensure_nchw(t: torch.Tensor) -> torch.Tensor:
    """Accept [N,H,W] -> [N,1,H,W]; keep [N,C,H,W] as-is."""
    if t.dim() == 3:
        return t.unsqueeze(1)
    return t


def _is_terramind_model(model: nn.Module) -> bool:
    """Robustly detect TerraMind even when wrapped by torch.compile."""
    base = getattr(model, "_orig_mod", model)
    try:
        return isinstance(base, TerraMind) or bool(getattr(base, "_is_terramind", False))
    except Exception:
        return bool(getattr(base, "_is_terramind", False))


def _as_probs_from_terratorch_logits_first(out, num_classes: int = 1) -> torch.Tensor:
    """
    TerraTorch containers -> probability maps. Prefer logits (keeps grad), then float predictions.
    Always return float NCHW.
    """
    from collections.abc import Mapping
    import torch as _torch

    def _activate_from_logits(logits: _torch.Tensor) -> _torch.Tensor:
        logits = _ensure_nchw(logits)
        if num_classes <= 1 or (logits.dim() >= 2 and logits.shape[1] == 1):
            return _torch.sigmoid(logits)
        return _torch.softmax(logits, dim=1)

    # 1) Prefer logits
    if isinstance(out, Mapping) and isinstance(out.get("logits", None), _torch.Tensor):
        return _activate_from_logits(out["logits"])
    if hasattr(out, "logits") and isinstance(out.logits, _torch.Tensor):
        return _activate_from_logits(out.logits)

    # 2) Allow prediction ONLY if it's already a float prob map
    if isinstance(out, Mapping) and isinstance(out.get("prediction", None), _torch.Tensor):
        p = _ensure_nchw(out["prediction"])
        if p.dtype.is_floating_point and p.dim() == 4:
            return p
    if hasattr(out, "prediction") and isinstance(out.prediction, _torch.Tensor):
        p = _ensure_nchw(out.prediction)
        if p.dtype.is_floating_point and p.dim() == 4:
            return p

    # 3) Fallbacks (dict values, attrs, sequences, tensors)
    if isinstance(out, Mapping):
        for v in out.values():
            if isinstance(v, _torch.Tensor):
                return _ensure_probabilities(_ensure_nchw(v))
    try:
        for v in vars(out).values():
            if isinstance(v, _torch.Tensor):
                return _ensure_probabilities(_ensure_nchw(v))
            if isinstance(v, (list, tuple)):
                for t in v:
                    if isinstance(t, _torch.Tensor):
                        return _ensure_probabilities(_ensure_nchw(t))
            if isinstance(v, dict):
                for t in v.values():
                    if isinstance(t, _torch.Tensor):
                        return _ensure_probabilities(_ensure_nchw(t))
    except Exception:
        pass
    if isinstance(out, (list, tuple)) and len(out) > 0:
        first = next((v for v in out if isinstance(v, _torch.Tensor)), None)
        if first is not None:
            return _ensure_probabilities(_ensure_nchw(first))
    if isinstance(out, _torch.Tensor):
        return _ensure_probabilities(_ensure_nchw(out))
    raise TypeError(f"Cannot interpret TerraTorch output of type {type(out)}")


def _as_probs_from_terratorch(out, num_classes: int = 1) -> torch.Tensor:
    """
    TerraTorch ModelOutput / HF-style container / dict / tuple / tensor -> probability maps.

    Order of attempts:
      1) mapping-like: prefer "prediction", then "logits"
      2) attribute-style: .prediction / .logits
      3) special adapters: .to_dict(), .as_dict(), .items(), __iter__()
      4) scan __dict__ for first Tensor
      5) sequence-like: first Tensor
      6) plain Tensor: use heuristic
    """
    from collections.abc import Mapping
    import torch as _torch

    def _activate_from_logits(logits: _torch.Tensor) -> _torch.Tensor:
        if num_classes <= 1 or logits.shape[1] == 1:
            return _torch.sigmoid(logits)
        return _torch.softmax(logits, dim=1)

    # 1) Mapping-like (many ModelOutput types behave like dicts)
    if isinstance(out, Mapping):
        if "prediction" in out and isinstance(out["prediction"], _torch.Tensor):
            return out["prediction"]
        if "logits" in out and isinstance(out["logits"], _torch.Tensor):
            return _activate_from_logits(out["logits"])
        # any first tensor value
        for v in out.values():
            if isinstance(v, _torch.Tensor):
                return _ensure_probabilities(v)

    # 2) Attribute-style
    if hasattr(out, "prediction") and isinstance(out.prediction, _torch.Tensor):
        return out.prediction
    if hasattr(out, "logits") and isinstance(out.logits, _torch.Tensor):
        return _activate_from_logits(out.logits)

    # 3) Adapters common in custom ModelOutput classes
    for to_dict_name in ("to_dict", "as_dict"):
        if hasattr(out, to_dict_name):
            try:
                d = getattr(out, to_dict_name)()
                if isinstance(d, dict):
                    if "prediction" in d and isinstance(d["prediction"], _torch.Tensor):
                        return d["prediction"]
                    if "logits" in d and isinstance(d["logits"], _torch.Tensor):
                        return _activate_from_logits(d["logits"])
                    for v in d.values():
                        if isinstance(v, _torch.Tensor):
                            return _ensure_probabilities(v)
            except Exception:
                pass

    if hasattr(out, "items"):
        try:
            for k, v in out.items():
                if k in ("prediction", "pred"):
                    if isinstance(v, _torch.Tensor):
                        return v
                if k == "logits" and isinstance(v, _torch.Tensor):
                    return _activate_from_logits(v)
            for _, v in out.items():
                if isinstance(v, _torch.Tensor):
                    return _ensure_probabilities(v)
        except Exception:
            pass

    if hasattr(out, "__iter__") and not isinstance(out, (list, tuple, _torch.Tensor)):
        try:
            it = iter(out)
            peek = next(it)
            if isinstance(peek, tuple) and len(peek) == 2:
                k, v = peek
                cands = [v] + [v2 for _, v2 in it]
            else:
                cands = [peek] + list(it)
            for v in cands:
                if isinstance(v, _torch.Tensor):
                    return _ensure_probabilities(v)
        except StopIteration:
            pass
        except Exception:
            pass

    try:
        for v in vars(out).values():
            if isinstance(v, _torch.Tensor):
                return _ensure_probabilities(v)
            if isinstance(v, (list, tuple)):
                for t in v:
                    if isinstance(t, _torch.Tensor):
                        return _ensure_probabilities(t)
            if isinstance(v, dict):
                for t in v.values():
                    if isinstance(t, _torch.Tensor):
                        return _ensure_probabilities(t)
    except Exception:
        pass

    if isinstance(out, (list, tuple)) and len(out) > 0:
        first = next((v for v in out if isinstance(v, _torch.Tensor)), None)
        if first is not None:
            return _ensure_probabilities(first)

    if isinstance(out, _torch.Tensor):
        return _ensure_probabilities(out)

    raise TypeError(f"Cannot interpret TerraTorch output of type {type(out)}")
