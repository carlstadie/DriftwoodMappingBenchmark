# core/UNet.py
from typing import Iterable, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Two 3x3 convs (optional dilation), ReLU after each conv,
    then BatchNorm and optional Dropout (after BN) — matches TF.
    TF BN defaults: epsilon=1e-3, momentum=0.99 (batch weight 0.01).
    PyTorch BN uses 'momentum' = batch weight, so set 0.01.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )
        # Align BN with TF defaults
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01, affine=True, track_running_stats=True)
        self.do = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x), inplace=False)
        x = F.relu(self.conv2(x), inplace=False)
        x = self.bn(x)
        x = self.do(x)
        return x


class AttentionGate2D(nn.Module):
    """
    Attention gate for skip connections.
    """

    def __init__(
        self,
        in_channels_x: int,
        in_channels_g: int,
        inter_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if inter_channels is None:
            inter_channels = max(1, in_channels_g // 4)

        self.theta_x = nn.Conv2d(in_channels_x, inter_channels, kernel_size=1, bias=True)
        self.phi_g   = nn.Conv2d(in_channels_g, inter_channels, kernel_size=1, bias=True)
        self.psi     = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)

    @staticmethod
    def _resize_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        theta_x = self.theta_x(x)
        phi_g   = self.phi_g(g)

        if theta_x.shape[-2:] != phi_g.shape[-2:]:
            phi_g = self._resize_like(phi_g, theta_x)

        f    = F.relu(theta_x + phi_g, inplace=False)
        psi  = self.psi(f)
        rate = torch.sigmoid(psi)  # (N,1,H,W)
        return x * rate


class UNetAttention(nn.Module):
    """
    UNet with attention-based skip connections (PyTorch).
    Matches TF graph structure & activations.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dilation_rate: int = 1,
        layer_count: int = 64,
        dropout: float = 0.0,
        l2_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        lc = layer_count

        # Encoder
        self.enc1 = ConvBlock(in_channels, 1 * lc, dilation=dilation_rate, dropout=dropout)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(1 * lc, 2 * lc, dilation=dilation_rate, dropout=dropout)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(2 * lc, 4 * lc, dilation=dilation_rate, dropout=dropout)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(4 * lc, 8 * lc, dilation=dilation_rate, dropout=dropout)
        self.pool4 = nn.MaxPool2d(2)

        self.center = ConvBlock(8 * lc, 16 * lc, dilation=dilation_rate, dropout=dropout)

        # Attention gates
        self.att6 = AttentionGate2D(in_channels_x=8 * lc, in_channels_g=16 * lc)
        self.att7 = AttentionGate2D(in_channels_x=4 * lc, in_channels_g=8 * lc)
        self.att8 = AttentionGate2D(in_channels_x=2 * lc, in_channels_g=4 * lc)
        self.att9 = AttentionGate2D(in_channels_x=1 * lc, in_channels_g=2 * lc)

        # Decoder
        self.dec6 = ConvBlock(in_channels=24 * lc, out_channels=8 * lc, dilation=1, dropout=dropout)
        self.dec7 = ConvBlock(in_channels=12 * lc, out_channels=4 * lc, dilation=1, dropout=dropout)
        self.dec8 = ConvBlock(in_channels=6  * lc, out_channels=2 * lc, dilation=1, dropout=dropout)
        self.dec9 = ConvBlock(in_channels=3  * lc, out_channels=1 * lc, dilation=1, dropout=dropout)

        # Head
        self.head = nn.Conv2d(1 * lc, num_classes, kernel_size=1, bias=True)

        self.l2_weight = float(l2_weight) if l2_weight is not None else 0.0

    @staticmethod
    def _upsample_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        c1 = self.enc1(x); p1 = self.pool1(c1)
        c2 = self.enc2(p1); p2 = self.pool2(c2)
        c3 = self.enc3(p2); p3 = self.pool3(c3)
        c4 = self.enc4(p3); p4 = self.pool4(c4)
        c5 = self.center(p4)

        # Decoder + attention-gated skips
        u6  = F.interpolate(c5, scale_factor=2.0, mode="bilinear", align_corners=False)
        u6  = self._upsample_like(u6, c4)
        a6  = self.att6(c4, u6)
        c6  = self.dec6(torch.cat([u6, a6], dim=1))

        u7  = F.interpolate(c6, scale_factor=2.0, mode="bilinear", align_corners=False)
        u7  = self._upsample_like(u7, c3)
        a7  = self.att7(c3, u7)
        c7  = self.dec7(torch.cat([u7, a7], dim=1))

        u8  = F.interpolate(c7, scale_factor=2.0, mode="bilinear", align_corners=False)
        u8  = self._upsample_like(u8, c2)
        a8  = self.att8(c2, u8)
        c8  = self.dec8(torch.cat([u8, a8], dim=1))

        u9  = F.interpolate(c8, scale_factor=2.0, mode="bilinear", align_corners=False)
        u9  = self._upsample_like(u9, c1)
        a9  = self.att9(c1, u9)
        c9  = self.dec9(torch.cat([u9, a9], dim=1))

        out = torch.sigmoid(self.head(c9))
        return out


def _normalize_num_classes(input_label_channels: Union[int, Iterable[int]]) -> int:
    if isinstance(input_label_channels, Iterable) and not isinstance(input_label_channels, (str, bytes)):
        try:
            return len(list(input_label_channels))
        except Exception:
            return int(input_label_channels)
    return int(input_label_channels)


def UNet(
    input_shape,
    input_label_channels: Union[int, Iterable[int]],
    dilation_rate: int = 1,
    layer_count: int = 64,
    l2_weight: float = 1e-4,
    dropout: float = 0.0,
    weight_file: str = None,
    summary: bool = False,
) -> nn.Module:
    """Factory that mirrors the TF signature."""
    in_channels = int(input_shape[-1])
    num_classes = _normalize_num_classes(input_label_channels)

    model = UNetAttention(
        in_channels=in_channels,
        num_classes=num_classes,
        dilation_rate=dilation_rate,
        layer_count=layer_count,
        dropout=dropout,
        l2_weight=l2_weight,
    )

    if weight_file:
        try:
            state = torch.load(weight_file, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"])
            elif isinstance(state, dict):
                model.load_state_dict(state)
            else:
                model = state
        except Exception as exc:
            raise RuntimeError(f"Failed to load PyTorch weights from '{weight_file}': {exc}") from exc

    if summary:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(model)
        print(f"Total params: {total:,}  •  Trainable: {trainable:,}")

    return model
