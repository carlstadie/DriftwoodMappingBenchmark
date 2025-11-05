import torch
import torch.nn as nn

# --- TerraTorch: make sure registries are populated ---
import terratorch
try:
    # Loads core registrations
    import terratorch.models  # noqa: F401
    # Ensures TerraMind backbones ('terramind_v1_*') are registered
    import terratorch.models.backbones.terramind  # noqa: F401
except Exception:
    pass

# If you use UPerNetDecoder (mmseg), also register mmseg decoders (safe no-op if not installed)
try:
    import mmseg  # noqa: F401
    import terratorch.models.decoders.mmseg  # noqa: F401
except Exception:
    pass

from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory


def _tm_indices_for_size(size: str):
    """
    Recommended transformer layer taps for hierarchical decoders.
    Base: [2,5,8,11], Large: [5,11,17,23].
    """
    size = (size or "base").lower()
    if size == "large":
        return [5, 11, 17, 23]
    # tiny/small/base -> default to base taps (12 layers total)
    return [2, 5, 8, 11]


def _bands_for_modality(modality: str, n_channels: int, explicit_bands=None):
    """
    Map your inputs to TerraMind's pre-trained band names.
    If explicit_bands is provided, pass-through that.
    """
    if explicit_bands is not None:
        return explicit_bands

    m = (modality or "S2").upper()

    # 4-band PlanetScope / Aerial assumed order: [B, G, R, NIR]
    if m in ("PS", "AE"):
        return ["BLUE", "GREEN", "RED", "NIR_NARROW"]

    # S2: infer common subsets by channel count, else None -> all bands
    if m == "S2":
        if n_channels == 3:
            return ["BLUE", "GREEN", "RED"]
        if n_channels == 4:
            return ["BLUE", "GREEN", "RED", "NIR_NARROW"]
        # 10–12 bands -> use all pretrained bands (no subset)
        return None

    # Fallback: no subset mapping
    return None


class TerraMind(nn.Module):
    """
    Thin wrapper that builds a TerraMind backbone + necks + decoder for segmentation.
    Uses TerraTorch's EncoderDecoderFactory under the hood.
    NOTE: We keep TerraTorch's ModelOutput (tutorial style).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        modality: str = "S2",                 # "S2", "PS", or "AE"
        tm_size: str = "large",               # "tiny" | "small" | "base" | "large"
        merge_method: str = "mean",           # mean | max | concat
        pretrained: bool = True,
        ckpt_path: str = None,
        indices_override=None,                # e.g. [5, 11, 17, 23]
        bands_override=None,                  # e.g. ["BLUE","GREEN","RED","NIR_NARROW"]
        decoder: str = "UperNetDecoder",      # <-- configurable (was hard-coded)
        decoder_channels=256,                 # int for UPerNet, list[int] for UNetDecoder
        decoder_kwargs: dict | None = None,   # <-- extra decoder-specific args
        backbone: str | None = None,          # <-- optional direct backbone name
        rescale: bool = True,                 # upsample logits to input size
    ):
        super().__init__()

        # 1) Choose the TerraMind variant (or accept explicit backbone override)
        backbone_name = backbone or f"terramind_v1_{tm_size.lower()}"

        # 2) Single input modality; reuse S2 embeddings with band subsets as needed.
        modality = (modality or "S2").upper()
        backbone_modalities = ["S2L2A"]  # always reuse S2 embeddings
        bands = _bands_for_modality(modality, in_channels, bands_override)
        backbone_bands = {"S2L2A": bands} if bands is not None else None

        # 3) Neck recipe for ViT-like encoders → hierarchical decoders
        indices = indices_override or _tm_indices_for_size(tm_size)
        necks = [
            {"name": "ReshapeTokensToImage", "remove_cls_token": False},
            {"name": "SelectIndices", "indices": indices},
            {"name": "LearnedInterpolateToPyramidal"},
        ]

        # 4) Build encoder+decoder model
        factory = EncoderDecoderFactory()
        extra = dict(decoder_kwargs or {})
        # keep backwards-compat: always pass decoder_channels if caller provided it
        if "decoder_channels" not in extra:
            extra["decoder_channels"] = decoder_channels

        self.model = factory.build_model(
            task="segmentation",
            backbone=backbone_name,
            decoder=decoder,                 # <-- now configurable
            num_classes=num_classes,
            necks=necks,
            rescale=rescale,

            # --- args routed to backbone ---
            backbone_pretrained=pretrained,
            backbone_modalities=backbone_modalities,
            backbone_merge_method=merge_method,
            backbone_bands=backbone_bands,
            backbone_ckpt_path=ckpt_path,

            # --- decoder-specific kwargs (channels, etc.) ---
            **extra,
        )

    def forward(self, x):
        # Return TerraTorch's ModelOutput; training loop will interpret it.
        return self.model(x)


    def forward(self, x):
        # Return TerraTorch's ModelOutput; training loop will interpret it.
        return self.model(x)
