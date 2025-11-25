# core/Swin_UNetPP.py
# implementation according to https://github.com/HuCaoFighting/Swin-Unet

from __future__ import annotations

from typing import Optional, Sequence, Tuple, List, Dict

import os
import sys
import warnings

import torch
import torch.nn as nn

try:
    # torchvision >= 0.13
    from torchvision.models.swin_transformer import (
        SwinTransformer,
        SwinTransformerBlock,
    )
    from torchvision.models.feature_extraction import create_feature_extractor
    try:
        # Optional ImageNet-1k weights for Swin-T
        from torchvision.models import Swin_T_Weights
    except Exception:  # pragma: no cover
        Swin_T_Weights = None  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This module requires torchvision with SwinTransformer and "
        "feature_extraction support (torchvision>=0.13)."
    ) from exc


# ----------------------------
# NHWC / NCHW helpers
# ----------------------------
def _to_nhwc(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a NCHW tensor to NHWC if needed.

    Args:
        x: Tensor of shape (N, C, H, W) or (N, H, W, C).

    Returns:
        Tensor in NHWC layout.
    """
    if x.ndim == 4 and x.shape[1] != x.shape[-1]:
        return x.permute(0, 2, 3, 1).contiguous()
    return x


def _to_nchw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a NHWC tensor to NCHW if needed.

    Args:
        x: Tensor of shape (N, H, W, C) or (N, C, H, W).

    Returns:
        Tensor in NCHW layout.
    """
    if x.ndim == 4 and x.shape[-1] != x.shape[1]:
        return x.permute(0, 3, 1, 2).contiguous()
    return x


# ----------------------------
# Band adaptation: inflate first patch-embed conv (3 -> N)
# ----------------------------
def _inflate_patch_embed_conv(backbone: nn.Module, in_chans: int) -> None:
    """
    Replace the first patch-embedding Conv2d to accept `in_chans` inputs.

    If increasing channels (>3), copy RGB kernels and initialize extra bands
    with the mean of RGB kernels. If reducing (<3), average RGB kernels and
    replicate to requested channels.

    Args:
        backbone: TorchVision SwinTransformer module.
        in_chans: Number of desired input channels (bands).
    """
    if in_chans == 3:
        return

    conv = None
    setter = None

    # TorchVision Swin: features[0] is Sequential(Conv2d, Permute, LayerNorm)
    if hasattr(backbone, "features"):
        feat0 = backbone.features[0]
        if isinstance(feat0, nn.Sequential) and isinstance(feat0[0], nn.Conv2d):
            conv = feat0[0]

            def _set(new_conv: nn.Conv2d) -> None:
                backbone.features[0][0] = new_conv

            setter = _set

    if conv is None or setter is None:
        raise RuntimeError("Could not locate patch-embedding Conv2d to inflate.")

    new_conv = nn.Conv2d(
        in_channels=in_chans,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=(conv.bias is not None),
    )

    with torch.no_grad():
        if in_chans > 3:
            new_conv.weight[:, :3] = conv.weight
            mean = conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, 3:] = mean.repeat(1, in_chans - 3, 1, 1)
        else:
            mean = conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight[:, :in_chans] = mean.repeat(1, in_chans, 1, 1)

        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    setter(new_conv)


# ----------------------------
# Patch expansion (NHWC)
# ----------------------------
class PatchExpandNHWC(nn.Module):
    """
    Upsample H,W by 2 using a linear expand + reshape.

    Input:
        Tensor of shape (B, H, W, C).

    Returns:
        Tensor of shape (B, 2H, 2W, C // 2).
    """

    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.dim = int(dim)
        self.expand = nn.Linear(self.dim, 2 * self.dim, bias=False)
        self.norm = norm_layer(self.dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        x = self.expand(x)  # (B, H, W, 2C)  -> rearrange to (B, 2H, 2W, C//2)
        x = x.view(b, h, w, 2, 2, c // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, h * 2, w * 2, c // 2)
        x = self.norm(x)
        return x


class FinalPatchExpandNHWC(nn.Module):
    """
    Final upsampling by 'up_factor' using linear expand + reshape.

    Input:
        Tensor of shape (B, H, W, C).

    Returns:
        Tensor of shape (B, up*H, up*W, C).
    """

    def __init__(self, dim: int, up_factor: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.up = int(up_factor)
        self.dim = int(dim)
        self.expand = nn.Linear(self.dim, (self.up ** 2) * self.dim, bias=False)
        self.norm = norm_layer(self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        x = self.expand(x)  # (B, H, W, up^2*C) -> (B, up*H, up*W, C)
        x = x.view(b, h, w, self.up, self.up, c)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, h * self.up, w * self.up, c)
        x = self.norm(x)
        return x


# ----------------------------
# Decoder Swin stage (NHWC)
# ----------------------------
def _make_swin_stage(
    dim: int,
    depth: int,
    num_heads: int,
    window_size: int,
    dpr_slice: Sequence[float],
) -> nn.Sequential:
    """
    Build a Swin stage (sequence of SwinTransformerBlock) with alternating
    non-shifted / shifted windows.

    Args:
        dim: Channel dimension of the stage.
        depth: Number of blocks in the stage.
        num_heads: Number of attention heads.
        window_size: Local attention window size.
        dpr_slice: Progressive drop-path slice for this stage.

    Returns:
        nn.Sequential of SwinTransformerBlock in NHWC mode.
    """
    blocks: List[nn.Module] = []
    w = int(window_size)
    for i in range(int(depth)):
        shift = [0, 0] if (i % 2 == 0) else [w // 2, w // 2]
        blocks.append(
            SwinTransformerBlock(
                dim=int(dim),
                num_heads=int(num_heads),
                window_size=[w, w],
                shift_size=shift,
                mlp_ratio=4.0,
                dropout=0.0,
                attention_dropout=0.0,
                stochastic_depth_prob=float(dpr_slice[i]) if i < len(dpr_slice) else 0.0,
            )
        )
    return nn.Sequential(*blocks)


# ----------------------------
# TorchVision Swin encoder (returns 4 NHWC features)
# ----------------------------
class _Encoder(nn.Module):
    """
    Wrap TorchVision SwinTransformer and expose (s0, s1, s2, s3) features.

    We instantiate the backbone with 3 input channels for ImageNet compatibility,
    optionally load pretrained Swin-T weights, then (if needed) inflate the
    patch-embedding conv to accept multi-band inputs.
    """

    def __init__(
        self,
        c: int,
        *,
        patch_size: int,
        in_chans: int,
        window_size: int,
        depths: Tuple[int, int, int, int],
        use_imagenet_weights: bool = False,
    ) -> None:
        super().__init__()
        heads = (
            max(1, c // 32),
            max(1, (2 * c) // 32),
            max(1, (4 * c) // 32),
            max(1, (8 * c) // 32),
        )

        # Build canonical Swin backbone (3-channel) so weights match exactly.
        self.backbone = SwinTransformer(
            patch_size=[patch_size, patch_size],
            embed_dim=c,
            depths=list(depths),
            num_heads=list(heads),
            window_size=[window_size, window_size],
            mlp_ratio=4.0,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth_prob=0.1,
            num_classes=0,  # we don't use the classification head
        )

        # Optionally load ImageNet weights (Swin-T only).
        if use_imagenet_weights:
            self._maybe_load_imagenet_weights(c, patch_size, window_size, depths)

        # Inflate for multi-band inputs (after weights are loaded).
        if in_chans != 3:
            _inflate_patch_embed_conv(self.backbone, in_chans)

        # Build feature extractor for the 4 hierarchical stages.
        self.extractor = create_feature_extractor(
            self.backbone,
            return_nodes={
                "features.1": "s0",  # H/patch,     C=c
                "features.3": "s1",  # H/2patch,    C=2c
                "features.5": "s2",  # H/4patch,    C=4c
                "features.7": "s3",  # H/8patch,    C=8c
            },
        )

    def _maybe_load_imagenet_weights(
        self,
        c: int,
        patch_size: int,
        window_size: int,
        depths: Tuple[int, int, int, int],
    ) -> None:
        """
        Load official Swin-T ImageNet-1k weights if architecture matches:
        c=96, patch_size=4, window_size=7, depths=(2,2,6,2).
        """
        ok = (
            Swin_T_Weights is not None
            and int(c) == 96
            and int(patch_size) == 4
            and int(window_size) == 7
            and tuple(map(int, depths)) == (2, 2, 6, 2)
        )
        if not ok:
            warnings.warn(
                "config/use_imagenet_weights=True but the Swin config is not "
                "Swin-T (c=96, patch=4, window=7, depths=(2,2,6,2)). Skipping.",
                stacklevel=2,
            )
            return

        try:
            # Get the state dict and drop classification head params.
            state = Swin_T_Weights.DEFAULT.get_state_dict(progress=True)  # type: ignore[attr-defined]
            # Remove the classification head keys (not present in our backbone)
            unexpected_keys = [k for k in list(state.keys()) if k.startswith("head.")]
            for k in unexpected_keys:
                state.pop(k, None)

            load_ret = self.backbone.load_state_dict(state, strict=False)
            # Count what we loaded vs skipped
            loaded_from_ckpt = len(state)
            skipped = len(load_ret.unexpected_keys) + len(load_ret.missing_keys)
            print(f"[Swin-T] Loaded {loaded_from_ckpt - skipped} keys, skipped {skipped} (head/mismatch).")
        except Exception as e:  # pragma: no cover
            warnings.warn(
                f"Could not load Swin-T weights ({e}). Proceeding without.",
                stacklevel=2,
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats = self.extractor(x)  # NHWC tensors
        s0 = feats["s0"]
        s1 = feats["s1"]
        s2 = feats["s2"]
        s3 = feats["s3"]
        return s3, [s0, s1, s2]


# ----------------------------
# Decoder (symmetric Swin in NHWC)
# ----------------------------
class _Decoder(nn.Module):
    """
    Symmetric Swin decoder with PatchExpand upsampling and progressive
    drop-path schedule across bottleneck + decoder blocks.
    """

    def __init__(
        self,
        c: int,
        window_size: int,
        depths: Tuple[int, int, int, int],
        dpr_decoder: Sequence[float],
    ) -> None:
        super().__init__()

        # Bottleneck (mirror encoder stage 3): 8c channels
        self.bottleneck = _make_swin_stage(
            dim=8 * c,
            depth=depths[3],
            num_heads=max(1, (8 * c) // 32),
            window_size=window_size,
            dpr_slice=dpr_decoder[: depths[3]],
        )

        ofs = depths[3]
        # Up: H/8 -> H/4 (8c -> 4c), then fuse with s2 (4c)
        self.up3 = PatchExpandNHWC(8 * c)
        self.red3 = nn.Linear(8 * c, 4 * c, bias=False)
        self.stage3 = _make_swin_stage(
            dim=4 * c,
            depth=depths[2],
            num_heads=max(1, (4 * c) // 32),
            window_size=window_size,
            dpr_slice=dpr_decoder[ofs : ofs + depths[2]],
        )

        ofs += depths[2]
        # Up: H/4 -> H/2 (4c -> 2c), then fuse with s1 (2c)
        self.up2 = PatchExpandNHWC(4 * c)
        self.red2 = nn.Linear(4 * c, 2 * c, bias=False)
        self.stage2 = _make_swin_stage(
            dim=2 * c,
            depth=depths[1],
            num_heads=max(1, (2 * c) // 32),
            window_size=window_size,
            dpr_slice=dpr_decoder[ofs : ofs + depths[1]],
        )

        ofs += depths[1]
        # Up: H/2 -> H (2c -> c), then fuse with s0 (c)
        self.up1 = PatchExpandNHWC(2 * c)
        self.red1 = nn.Linear(2 * c, c, bias=False)
        self.stage1 = _make_swin_stage(
            dim=c,
            depth=depths[0],
            num_heads=max(1, c // 32),
            window_size=window_size,
            dpr_slice=dpr_decoder[ofs : ofs + depths[0]],
        )

        # Final expansion to input stride (set by SwinUNet)
        self.final_up: Optional[FinalPatchExpandNHWC] = None

    @staticmethod
    def _concat_reduce(x: torch.Tensor, skip: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        """
        Concatenate decoder tensor with skip (NHWC) and reduce channels.

        Args:
            x: Current decoder tensor (B, H, W, Cx).
            skip: Encoder skip tensor (B, H, W, Cs).
            proj: Linear projection to reduce channels.

        Returns:
            Tensor (B, H, W, Cout).
        """
        x = torch.cat([x, skip], dim=-1)
        return proj(x)

    def forward(self, x: torch.Tensor, skips_nhwc: Sequence[torch.Tensor]) -> torch.Tensor:
        # x: deepest stage (s3); skips: [s0, s1, s2] (all NHWC)
        s0, s1, s2 = skips_nhwc

        x = self.bottleneck(x)

        x = self.up3(x)
        x = self._concat_reduce(x, s2, self.red3)
        x = self.stage3(x)

        x = self.up2(x)
        x = self._concat_reduce(x, s1, self.red2)
        x = self.stage2(x)

        x = self.up1(x)
        x = self._concat_reduce(x, s0, self.red1)
        x = self.stage1(x)

        if self.final_up is not None:
            x = self.final_up(x)
        return x


# ----------------------------
# Weight copy: initialize decoder from encoder
# ----------------------------
def _copy_block_params(dst: SwinTransformerBlock, src: SwinTransformerBlock) -> None:
    """
    Copy parameters from an encoder Swin block to a decoder Swin block.

    Args:
        dst: Destination block (decoder).
        src: Source block (encoder).
    """
    with torch.no_grad():
        # Norms
        dst.norm1.weight.copy_(src.norm1.weight)
        dst.norm1.bias.copy_(src.norm1.bias)
        dst.norm2.weight.copy_(src.norm2.weight)
        dst.norm2.bias.copy_(src.norm2.bias)
        # Attention layers
        dst.attn.qkv.weight.copy_(src.attn.qkv.weight)
        if dst.attn.qkv.bias is not None and src.attn.qkv.bias is not None:
            dst.attn.qkv.bias.copy_(src.attn.qkv.bias)
        dst.attn.proj.weight.copy_(src.attn.proj.weight)
        if dst.attn.proj.bias is not None and src.attn.proj.bias is not None:
            dst.attn.proj.bias.copy_(src.attn.proj.bias)
        # Relative position bias (if present)
        if hasattr(dst.attn, "relative_position_bias_table") and hasattr(
            src.attn, "relative_position_bias_table"
        ):
            dst.attn.relative_position_bias_table.data.copy_(
                src.attn.relative_position_bias_table.data
            )
        # MLP (two Linear layers)
        dst_linears = [m for m in dst.mlp.modules() if isinstance(m, nn.Linear)]
        src_linears = [m for m in src.mlp.modules() if isinstance(m, nn.Linear)]
        for dlin, slin in zip(dst_linears, src_linears):
            dlin.weight.copy_(slin.weight)
            if dlin.bias is not None and slin.bias is not None:
                dlin.bias.copy_(slin.bias)


def _init_decoder_from_encoder(decoder: _Decoder, backbone: SwinTransformer) -> None:
    """
    Mirror encoder stage weights into decoder stages at matching scales.

    Works with torchvision's SwinTransformer (stages in backbone.features)
    and, as a fallback, with timm-style Swin that exposes a `.blocks` attr.
    """
    try:
        s0_blocks = s1_blocks = s2_blocks = s3_blocks = None

        # ---- 1) TorchVision Swin: stages are nn.Sequential of SwinTransformerBlock in backbone.features ----
        if hasattr(backbone, "features") and isinstance(backbone.features, nn.Sequential):
            encoder_stages = []
            for m in backbone.features:
                # We treat any nn.Sequential that contains SwinTransformerBlock(s) as a stage
                if isinstance(m, nn.Sequential) and any(
                    isinstance(b, SwinTransformerBlock) for b in m
                ):
                    encoder_stages.append(m)

            if len(encoder_stages) >= 4:
                s0_blocks, s1_blocks, s2_blocks, s3_blocks = encoder_stages[:4]

        # ---- 2) Fallback: timm-style / other Swin with `.blocks` attribute ----
        if s0_blocks is None:
            stage_modules = [m for m in backbone.modules() if hasattr(m, "blocks")]
            if len(stage_modules) < 4:
                raise RuntimeError(
                    f"Expected at least 4 Swin stages, found {len(stage_modules)}."
                )
            s0_blocks = stage_modules[0].blocks
            s1_blocks = stage_modules[1].blocks
            s2_blocks = stage_modules[2].blocks
            s3_blocks = stage_modules[3].blocks

        # Now copy encoder → decoder (deepest → bottleneck)
        # decoder.bottleneck  ~ encoder stage3
        for d_blk, s_blk in zip(decoder.bottleneck, s3_blocks):
            _copy_block_params(d_blk, s_blk)

        # decoder.stage3 ~ encoder stage2
        for d_blk, s_blk in zip(decoder.stage3, s2_blocks):
            _copy_block_params(d_blk, s_blk)

        # decoder.stage2 ~ encoder stage1
        for d_blk, s_blk in zip(decoder.stage2, s1_blocks):
            _copy_block_params(d_blk, s_blk)

        # decoder.stage1 ~ encoder stage0
        for d_blk, s_blk in zip(decoder.stage1, s0_blocks):
            _copy_block_params(d_blk, s_blk)

    except Exception as e:  # pragma: no cover
        warnings.warn(
            f"Could not initialize decoder from encoder weights (layout changed?): {e}. "
            f"Proceeding without explicit mirroring.",
            stacklevel=2,
        )



# ----------------------------
# SwinUNet
# ----------------------------
class SwinUNet(nn.Module):
    """
    UNet-like architecture built from a TorchVision Swin encoder and a symmetric
    Swin decoder (true to the official Swin-UNet design).

    Entry-point signature preserved for compatibility with project scripts.
    """

    def __init__(
        self,
        h: int,
        w: int,
        ch: int,
        c: int,
        num_class: int = 1,
        num_blocks: int = 3,
        patch_size: int = 4,
        window_size: int = 7,

        # included for compatibility; not used as handled inside blocks
        ss_size: Optional[int] = None,  # shift-size (implicit via alternation)
        attn_drop: float = 0.0,         # kept for API compat
        proj_drop: float = 0.0,         # kept for API compat
        mlp_drop: float = 0.0,          # kept for API compat
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        assert int(num_blocks) == 3, "This SwinUNet assumes three downsampling stages."
        self.h = int(h)
        self.w = int(w)
        self.patch_size = int(patch_size)
        depths = (2, 2, 6, 2)  # official Swin-T depths

        # Optional pretrained loading controlled by config.use_imagenet_weights
        use_imagenet_weights = self._resolve_use_imagenet_weights_flag()

        # Encoder (TorchVision Swin)
        self.encoder = _Encoder(
            c=int(c),
            patch_size=int(patch_size),
            in_chans=int(ch),
            window_size=int(window_size),
            depths=depths,
            use_imagenet_weights=use_imagenet_weights,
        )

        # Progressive drop-path schedule across bottleneck + decoder
        total_dec_blocks = sum(depths)
        dpr = torch.linspace(0.0, float(drop_path), steps=total_dec_blocks).tolist()

        # Decoder (symmetric)
        self.decoder = _Decoder(
            c=int(c),
            window_size=int(window_size),
            depths=depths,
            dpr_decoder=dpr,
        )
        self.decoder.final_up = FinalPatchExpandNHWC(int(c), up_factor=int(patch_size))

        # Per-pixel head (NHWC -> logits) + sigmoid in forward
        self.head = nn.Linear(int(c), int(num_class), bias=True)

        # Tell the training loop we already return probabilities (NCHW, in [0,1])
        setattr(self, "_returns_probabilities", True)

        # Initialize decoder from encoder if pretrained were loaded
        if use_imagenet_weights:
            _init_decoder_from_encoder(self.decoder, self.encoder.backbone)

    # ----------------------------
    # Resolve config flag for pretrained
    # ----------------------------
    @staticmethod
    def _resolve_use_imagenet_weights_flag() -> bool:
        """
        Read a runtime flag controlling pretrained loading:
            - Prefer a global 'config.use_imagenet_weights' if present in
              training/tuning/evaluation modules.
            - Else, read env var USE_IMAGENET_WEIGHTS.
            - Default: False.
        """
        for mod_name in ("training", "tuning", "evaluation"):
            mod = sys.modules.get(mod_name, None)
            if mod is not None and hasattr(mod, "config"):
                cfg = getattr(mod, "config")
                if cfg is not None and hasattr(cfg, "use_imagenet_weights"):
                    try:
                        return bool(getattr(cfg, "use_imagenet_weights"))
                    except Exception:
                        pass
        env = os.getenv("USE_IMAGENET_WEIGHTS", "").strip().lower()
        if env in {"1", "true", "yes", "on"}:
            return True
        if env in {"0", "false", "no", "off"}:
            return False
        return False

    # ----------------------------
    # MC Dropout utilities
    # ----------------------------
    def enable_mc_dropout(self) -> None:
        """
        Enable MC dropout by setting all dropout layers to training mode
        while keeping batch norm in eval mode.
        """
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    
    def predict_with_mc_dropout(
        self, 
        x: torch.Tensor, 
        n_samples: int = 10,
        return_uncertainty: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Perform MC dropout prediction by running multiple forward passes.
        
        Args:
            x: Input tensor of shape (N, ch, H, W).
            n_samples: Number of MC dropout samples to collect.
            return_uncertainty: If True, return (mean, std); else just mean.
            
        Returns:
            If return_uncertainty=True: (mean, std) both of shape (N, num_class, H, W)
            If return_uncertainty=False: mean of shape (N, num_class, H, W)
        """
        was_training = self.training
        self.eval()
        self.enable_mc_dropout()
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (n_samples, N, num_class, H, W)
        mean = predictions.mean(dim=0)
        
        if return_uncertainty:
            std = predictions.std(dim=0)
            if was_training:
                self.train()
            return mean, std
        else:
            if was_training:
                self.train()
            return mean

    # ----------------------------
    # Forward
    # ----------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (N, ch, H, W).

        Returns:
            Tensor of shape (N, num_class, H, W) with probabilities in [0, 1].
        """
        # Encoder (TorchVision Swin provides NHWC features)
        s3, skips = self.encoder(x)  # s3 deepest, skips=[s0, s1, s2]

        # Decoder (NHWC)
        y = self.decoder(s3, skips)

        # Head (NHWC -> NCHW) + sigmoid
        logits = self.head(y)
        logits = _to_nchw(logits)
        out = torch.sigmoid(logits)

        # If internal padding occurred, crop back to (H, W)
        if out.shape[-2] != self.h or out.shape[-1] != self.w:
            out = out[..., : self.h, : self.w]
        return out
