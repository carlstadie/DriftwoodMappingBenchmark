# core/Swin_UNetPP.py
# implementation according to https://github.com/HuCaoFighting/Swin-Unet

from __future__ import annotations

from typing import Optional, Sequence, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Window operations (N, C, H, W)
# ----------------------------
def window_partition(x: torch.Tensor, window_size: Tuple[int, int]) -> torch.Tensor:
    """
    Split an image into non-overlapping windows.

    Args:
        x: Tensor of shape (B, C, H, W).
        window_size: Tuple (Wh, Ww).

    Returns:
        Tensor of shape (B * nWh * nWw, Wh * Ww, C).
    """
    b, c, h, w = x.shape
    wh, ww = window_size
    assert h % wh == 0 and w % ww == 0, "H and W must be divisible by window size."
    x = x.view(b, c, h // wh, wh, w // ww, ww)  # B, C, nWh, Wh, nWw, Ww
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # B, nWh, nWw, Wh, Ww, C
    windows = x.view(-1, wh * ww, c)  # B*nW, N, C
    return windows


def window_reverse(
    windows: torch.Tensor,
    window_size: Tuple[int, int],
    h: int,
    w: int,
    c: int,
) -> torch.Tensor:
    """
    Reverse the partition and rebuild the image grid.

    Args:
        windows: Tensor (B * nW, Wh * Ww, C).
        window_size: Tuple (Wh, Ww).
        h: Output height.
        w: Output width.
        c: Channels.

    Returns:
        Tensor of shape (B, C, H, W).
    """
    wh, ww = window_size
    n_wh = h // wh
    n_ww = w // ww
    b = windows.shape[0] // (n_wh * n_ww)
    x = windows.view(b, n_wh, n_ww, wh, ww, c)  # B, nWh, nWw, Wh, Ww, C
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # B, C, nWh, Wh, nWw, Ww
    x = x.view(b, c, h, w)
    return x


# ----------------------------
# Normalisation helper: LayerNorm over channels for 2D maps
# ----------------------------
class LayerNorm2d(nn.Module):
    """
    LayerNorm over channels for tensors shaped (N, C, H, W).

    This applies per-pixel LayerNorm across the channel axis,
    matching LayerNormalization(eps=1e-5) used in the TF version.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)  # N, H, W, C
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # N, C, H, W
        return x


# ----------------------------
# Stochastic depth (DropPath)
# ----------------------------
class DropPath(nn.Module):
    """Stochastic depth layer used inside residual paths."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x / keep_prob * binary_tensor


# ----------------------------
# Windowed self-attention
# ----------------------------
class WindowAttention(nn.Module):
    """Multi-head self-attention applied within local windows."""

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int = 4,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_dropout = nn.Dropout(attn_drop_rate)
        self.proj_dropout = nn.Dropout(proj_drop_rate)

        wh, ww = window_size
        num_rel_positions = (2 * wh - 1) * (2 * ww - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_rel_positions, num_heads)
        )
        # Precompute relative position index (N, N)
        coords_h = torch.arange(wh)
        coords_w = torch.arange(ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords = coords.reshape(2, -1)
        rel_coords = coords[:, :, None] - coords[:, None, :]  # 2, N, N
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        rel_coords[..., 0] += wh - 1
        rel_coords[..., 1] += ww - 1
        rel_index = rel_coords[..., 0] * (2 * ww - 1) + rel_coords[..., 1]
        self.register_buffer(
            "relative_position_index", rel_index.long(), persistent=False
        )

    def forward(
        self,
        x: torch.Tensor,  # (B_, N, C)
        mask: Optional[torch.Tensor] = None,  # (nW, N, N) with {0, -1e9}
    ) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).view(b_, n, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B_, heads, N, dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)  # B_, heads, N, N

        if mask is not None:
            n_w = mask.shape[0]
            attn = attn.view(b_ // n_w, n_w, self.num_heads, n, n)
            attn = attn + mask[:, None, :, :].to(attn.dtype)
            attn = attn.view(-1, self.num_heads, n, n)

        # relative bias (heads, N, N)
        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        bias = bias.reshape(n, n, self.num_heads).permute(2, 0, 1)
        bias = torch.clamp(bias, -5.0, 5.0)
        attn = attn + bias.unsqueeze(0)

        attn = torch.clamp(attn, -10.0, 10.0)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v  # B_, heads, N, dim
        out = out.permute(0, 2, 1, 3).contiguous().view(b_, n, c)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


# ----------------------------
# Swin Transformer Block
# ----------------------------
class SwinTransformerBlock(nn.Module):
    """Two-window attention block with optional shift and MLP."""

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int = 4,
        window_size: int = 4,
        shift_size: int = 0,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        mlp_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        h, w = input_resolution
        ws = min(window_size, h, w)
        self.window_size = ws
        self.shift_size = min(shift_size, ws // 2)

        self.norm1 = LayerNorm2d(dim, eps=1e-5)
        self.attn = WindowAttention(
            dim=dim,
            window_size=(ws, ws),
            num_heads=num_heads,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
        )
        self.attn_dropout = nn.Dropout(attn_drop_rate)
        self.proj_dropout = nn.Dropout(proj_drop_rate)

        self.norm2 = LayerNorm2d(dim, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, 4 * dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout(mlp_drop_rate),
            nn.Conv2d(4 * dim, dim, kernel_size=1, bias=True),
            nn.Dropout(mlp_drop_rate),
        )
        self.drop_path = DropPath(drop_prob=drop_path_rate)

        # Attention mask for shifted windows
        if self.shift_size > 0:
            img_mask = torch.zeros(1, 1, h, w)  # (1, 1, H, W)
            h_slices = (
                slice(0, -ws),
                slice(-ws, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -ws),
                slice(-ws, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for hs in h_slices:
                for ws_ in w_slices:
                    img_mask[:, :, hs, ws_] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, (ws, ws)).squeeze(-1)  # (nW, N)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = torch.where(
                attn_mask != 0, torch.tensor(-1e9), torch.tensor(0.0)
            )
            self.register_buffer(
                "attn_mask", attn_mask.to(torch.float32), persistent=False
            )
        else:
            self.register_buffer("attn_mask", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        n, c, h, w = x.shape
        shortcut = x

        x = self.norm1(x)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1))

        # window attention
        x_w = window_partition(x, (self.window_size, self.window_size))  # (B*, N, C)
        attn_w = self.attn(x_w, mask=self.attn_mask)
        attn_w = self.attn_dropout(attn_w)
        x = window_reverse(attn_w, (self.window_size, self.window_size), h, w, c)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(-2, -1))

        x = self.proj_dropout(x)
        x = shortcut + self.drop_path(x)

        shortcut2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut2 + self.drop_path(x)
        return x


# ----------------------------
# Patch ops
# ----------------------------
class PatchEmbedding(nn.Module):
    """Patchify input and project to the embedding dimension."""

    def __init__(self, in_ch: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_ch, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0, bias=True
        )
        self.norm = LayerNorm2d(embed_dim, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """
    Downsample H,W by 2; channels: C -> 2C (after LN and 1x1 conv).

    Mirrors TF: reshape to 4*C channels, LayerNorm, then Dense(2C) with no bias.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(4 * dim, eps=1e-5)
        self.reduction = nn.Conv2d(4 * dim, 2 * dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert h % 2 == 0 and w % 2 == 0, "H and W must be even for PatchMerging."
        x = x.view(b, c, h // 2, 2, w // 2, 2)  # B, C, H/2, 2, W/2, 2
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # B, 2, 2, C, H/2, W/2
        x = x.view(b, 4 * c, h // 2, w // 2)  # B, 4C, H/2, W/2
        x = self.norm(x)
        x = self.reduction(x)  # B, 2C, H/2, W/2
        # Ich will nicht mehr, ich kann nicht mehr, ich halte das nicht mehr aus
        return x


class PatchExpansion(nn.Module):
    """
    Upsample H,W by 2 via pixel-shuffle; channels: dim -> dim//2 after shuffle.

    Matches TF: Dense(2*dim) + inverse space-to-depth.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.expand = nn.Conv2d(dim, 2 * dim, kernel_size=1, bias=False)
        self.shuffle = nn.PixelShuffle(upscale_factor=2)
        self.norm = LayerNorm2d(dim // 2, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.shuffle(x)
        x = self.norm(x)
        return x


class FinalPatchExpansion(nn.Module):
    """
    Final upsampling by 'up_factor' using pixel-shuffle after 1x1 expansion.

    Dense((up^2)*dim) -> reshape -> transpose -> reshape.
    """

    def __init__(self, dim: int, up_factor: int) -> None:
        super().__init__()
        self.up = up_factor
        self.expand = nn.Conv2d(dim, (up_factor ** 2) * dim, kernel_size=1, bias=False)
        self.shuffle = nn.PixelShuffle(upscale_factor=up_factor)
        self.norm = LayerNorm2d(dim, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.shuffle(x)
        x = self.norm(x)
        return x


# ----------------------------
# Two-block Swin stage
# ----------------------------
class SwinBlock(nn.Module):
    """A pair of shifted-window transformer blocks."""

    def __init__(
        self,
        dims: int,
        ip_res: Tuple[int, int],
        window_size: int = 4,
        ss_size: Optional[int] = None,
        num_heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        if ss_size is None:
            ss_size = window_size // 2
        self.swtb1 = SwinTransformerBlock(
            dim=dims,
            input_resolution=ip_res,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            attn_drop_rate=attn_drop,
            proj_drop_rate=proj_drop,
            mlp_drop_rate=mlp_drop,
            drop_path_rate=drop_path,
        )
        self.swtb2 = SwinTransformerBlock(
            dim=dims,
            input_resolution=ip_res,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=ss_size,
            attn_drop_rate=attn_drop,
            proj_drop_rate=proj_drop,
            mlp_drop_rate=mlp_drop,
            drop_path_rate=drop_path,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.swtb1(x)
        x = self.swtb2(x)
        return x


# ----------------------------
# Encoder / Decoder
# ----------------------------
class Encoder(nn.Module):
    """Three-stage encoder with Swin blocks and patch merging."""

    def __init__(
        self,
        c: int,
        input_resolution: Tuple[int, int],
        window_size: int = 4,
        ss_size: Optional[int] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        h, w = input_resolution
        ws = window_size
        ss = ss_size if ss_size is not None else ws // 2

        self.enc_swin_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    SwinBlock(
                        c,
                        (h, w),
                        window_size=ws,
                        ss_size=ss,
                        num_heads=max(1, c // 32),
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        mlp_drop=mlp_drop,
                        drop_path=drop_path,
                    ),
                    SwinBlock(
                        c,
                        (h, w),
                        window_size=ws,
                        ss_size=ss,
                        num_heads=max(1, c // 32),
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        mlp_drop=mlp_drop,
                        drop_path=drop_path,
                    ),
                ),
                nn.Sequential(
                    SwinBlock(
                        2 * c,
                        (h // 2, w // 2),
                        window_size=ws,
                        ss_size=ss,
                        num_heads=max(1, (2 * c) // 32),
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        mlp_drop=mlp_drop,
                        drop_path=drop_path,
                    ),
                    SwinBlock(
                        2 * c,
                        (h // 2, w // 2),
                        window_size=ws,
                        ss_size=ss,
                        num_heads=max(1, (2 * c) // 32),
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        mlp_drop=mlp_drop,
                        drop_path=drop_path,
                    ),
                ),
                nn.Sequential(
                    SwinBlock(
                        4 * c,
                        (h // 4, w // 4),
                        window_size=ws,
                        ss_size=ss,
                        num_heads=max(1, (4 * c) // 32),
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        mlp_drop=mlp_drop,
                        drop_path=drop_path,
                    ),
                    SwinBlock(
                        4 * c,
                        (h // 4, w // 4),
                        window_size=ws,
                        ss_size=ss,
                        num_heads=max(1, (4 * c) // 32),
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        mlp_drop=mlp_drop,
                        drop_path=drop_path,
                    ),
                ),
            ]
        )

        self.enc_patch_merge_blocks = nn.ModuleList(
            [PatchMerging(c), PatchMerging(2 * c), PatchMerging(4 * c)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skip_conn_ftrs: List[torch.Tensor] = []
        for swin_blocks, patch_merger in zip(
            self.enc_swin_blocks, self.enc_patch_merge_blocks
        ):
            x = swin_blocks(x)
            skip_conn_ftrs.append(x)
            x = patch_merger(x)
        return x, skip_conn_ftrs


class Decoder(nn.Module):
    """Three-stage decoder with patch expansion and skip connections."""

    def __init__(
        self,
        c: int,
        input_resolution: Tuple[int, int],
        window_size: int = 4,
        ss_size: Optional[int] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        h, w = input_resolution
        ws = window_size
        ss = ss_size if ss_size is not None else ws // 2

        self.dec_swin_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    SwinBlock(
                        4 * c,
                        (h // 4, w // 4),
                        window_size=ws,
                        ss_size=ss,
                        num_heads=max(1, (4 * c) // 32),
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        mlp_drop=mlp_drop,
                        drop_path=drop_path,
                    ),
                    SwinBlock(
                        4 * c,
                        (h // 4, w // 4),
                        window_size=ws,
                        ss_size=ss,
                        num_heads=max(1, (4 * c) // 32),
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        mlp_drop=mlp_drop,
                        drop_path=drop_path,
                    ),
                ),
                nn.Sequential(
                    SwinBlock(
                        2 * c,
                        (h // 2, w // 2),
                        window_size=ws,
                        ss_size=ss,
                        num_heads=max(1, (2 * c) // 32),
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        mlp_drop=mlp_drop,
                        drop_path=drop_path,
                    ),
                    SwinBlock(
                        2 * c,
                        (h // 2, w // 2),
                        window_size=ws,
                        ss_size=ss,
                        num_heads=max(1, (2 * c) // 32),
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        mlp_drop=mlp_drop,
                        drop_path=drop_path,
                    ),
                ),
                nn.Sequential(
                    SwinBlock(
                        c,
                        (h, w),
                        window_size=ws,
                        ss_size=ss,
                        num_heads=max(1, c // 32),
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        mlp_drop=mlp_drop,
                        drop_path=drop_path,
                    ),
                    SwinBlock(
                        c,
                        (h, w),
                        window_size=ws,
                        ss_size=ss,
                        num_heads=max(1, c // 32),
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        mlp_drop=mlp_drop,
                        drop_path=drop_path,
                    ),
                ),
            ]
        )

        # After concat with encoder features, reduce back to stage dimensions
        self.dec_patch_expand_blocks = nn.ModuleList(
            [PatchExpansion(8 * c), PatchExpansion(4 * c), PatchExpansion(2 * c)]
        )
        self.skip_conv_layers = nn.ModuleList(
            [
                nn.Conv2d(8 * c, 4 * c, kernel_size=1, bias=False),
                nn.Conv2d(4 * c, 2 * c, kernel_size=1, bias=False),
                nn.Conv2d(2 * c, c, kernel_size=1, bias=False),
            ]
        )

    def forward(
        self, x: torch.Tensor, encoder_features: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        for patch_expand, swin_blocks, enc_ftr, skip_conv in zip(
            self.dec_patch_expand_blocks,
            self.dec_swin_blocks,
            reversed(encoder_features),
            self.skip_conv_layers,
        ):
            x = patch_expand(x)  # upsample 2x
            x = torch.cat([x, enc_ftr], dim=1)  # concat skip
            x = skip_conv(x)  # 1x1 to match stage dim
            x = swin_blocks(x)
        return x


# ----------------------------
# SwinUNet
# ----------------------------
class SwinUNet(nn.Module):
    """UNet-like architecture built from Swin Transformer stages."""

    def __init__(
        self,
        h: int,
        w: int,
        ch: int,
        c: int,
        num_class: int = 1,
        num_blocks: int = 3,
        patch_size: int = 16,
        window_size: int = 4,
        ss_size: Optional[int] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        # Patch embedding (Conv2d stride=patch_size) + LN
        self.patch_embed = PatchEmbedding(ch, c, patch_size)

        enc_ip = (h // patch_size, w // patch_size)

        # Encoder
        self.encoder = Encoder(
            c,
            enc_ip,
            window_size=window_size,
            ss_size=ss_size,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
            drop_path=drop_path,
        )

        # Bottleneck dims / resolution
        bot_dim = c * (2 ** num_blocks)
        bot_ip = (h // (patch_size * (2 ** num_blocks)), w // (patch_size * (2 ** num_blocks)))
        self.bottleneck = SwinBlock(
            bot_dim,
            bot_ip,
            window_size=window_size,
            ss_size=ss_size,
            num_heads=max(1, bot_dim // 32),
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
            drop_path=drop_path,
        )

        # Decoder
        self.decoder = Decoder(
            c,
            enc_ip,
            window_size=window_size,
            ss_size=ss_size,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop,
            drop_path=drop_path,
        )

        # Final expansion back to input stride then 1x1 head + sigmoid
        self.final_expansion = FinalPatchExpansion(c, up_factor=patch_size)
        self.head = nn.Conv2d(c, num_class, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, ch, H, W)
        x = self.patch_embed(x)
        x, skip_ftrs = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_ftrs)
        x = self.final_expansion(x)
        x = torch.sigmoid(self.head(x))
        return x
