# pretrain_terramind_continual_mim_nce_tokence.py
# Continual pretraining for AE/PS (4-band) + S2 (10-band) using TerraMind (TerraTorch)
# Objectives:
#   - Masked Image Modeling (per modality; pixel L2 on masked patches)
#   - Cross-modal InfoNCE on pooled features (starts after warmup)
#   - Optional Token-CE on S2 (predict masked S2 code IDs from AE+PS fused tokens)
#
# Notes:
#   - Planet/Aerial were not in the original TerraMind training. We unfreeze slowly,
#     use a small trunk LR, warm up, and schedule the mask ratio.
#   - Geometry augs come from Albumentations (channel-agnostic). Photometric jitter is
#     custom and channel-safe. You can phase in photo jitter & channel drop after a few epochs.
#   - Every __getitem__ draws a fresh random window. Dataset length is scaled via:
#       repeats_per_epoch * crops_per_file_per_epoch.

import os, random, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import bounds as win_bounds
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Callback

# -----------------------------
# 1) CONFIG
# -----------------------------
class Configuration:
    def __init__(self):
        # Data
        self.aerial_dir   = '/isipd/projects/p_planetdw/data/methods_test/training_images/FN/AE'
        self.ps_dir       = '/isipd/projects/p_planetdw/data/methods_test/training_images/FN/PS'
        self.s2_dir       = '/isipd/projects/p_planetdw/data/methods_test/training_images/FN/S2/temp'
        self.pretrain_ext = '.tif'

        # Training
        self.patch_size   = (256, 256)    # crop size (H, W)
        self.batch_size   = 32
        self.epochs       = 50
        self.lr_trunk     = 1e-4          # trunk LR (adapt to new sensors)
        self.lr_heads     = 5e-4          # MIM/CE heads + projection head
        self.weight_decay = 1e-4
        self.fp16         = True
        self.seed         = 1701
        self.num_workers  = 1

        # Coverage / steps per epoch
        self.repeats_per_epoch        = 4
        self.crops_per_file_per_epoch = 6

        # Hardware
        self.devices  = 1
        self.strategy = "auto"
        self.gpu_index = 7

        # Objectives
        self.mim_mask_ratio_min = 0.25
        self.mim_mask_ratio_max = 0.55
        self.lambda_mim     = 1.0
        self.lambda_nce     = 0.0   # switched on after nce_start_epoch (below)
        self.lambda_tokence = 1.0
        self.temperature    = 0.1

        # Curriculum & schedules
        self.warmup_epochs       = 3
        self.freeze_trunk_epochs = 1
        self.min_lr_ratio        = 0.1
        self.proj_dropout        = 0.10
        self.repr_dropout_prob   = 0.05
        self.nce_start_epoch     = 5            # start InfoNCE after this epoch
        self.photo_enable_epoch  = 2            # enable photo jitter & channel drop after this epoch

        # Augment / normalize
        self.nodata_val            = None
        self.augmenter_strength    = 0.5        # geometry only
        self.photo_jitter_strength = 0.5        # custom, channel-safe
        self.channel_drop_prob     = 0.05

        # Visualization
        self.viz_every_steps = 200              # TB images frequency

        # Run dirs
        self.run_id       = 'terramind_continual_shared_mim_nce_tokence'
        self.logs_dir     = f'/isipd/projects/p_planetdw/data/methods_test/logs/{self.run_id}'
        self.models_dir   = f'/isipd/projects/p_planetdw/data/methods_test/models/{self.run_id}'
        self.continue_model_path = None

        # S2 band configuration (we train with 10 bands). If a file has 12, keep:
        # On-disk order: [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12]
        self.s2l2a_expect_channels = 10
        self.s2l2a_from12_keep_idx = [1,2,3,4,5,6,7,8,10,11]  # B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12
        self.s2l2a_bands = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]

        # Optional S2 tokenizer for Token-CE
        self.s2_tokenizer_path: Optional[str] = None
        self.s2_tokenizer_callable: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self.s2_tokenizer_num_codes: Optional[int] = None
        self.s2_tokenizer_input_scale_mode = "minmax"

config = Configuration()

# -----------------------------
# 2) FILE DISCOVERY & FILTERS
# -----------------------------
def list_files(folder, ext):
    print(f"Listing *{ext} files in {folder}...")
    ext = ext.lower()
    try:
        names = os.listdir(folder)
    except FileNotFoundError:
        print(f"[WARN] Folder not found: {folder}")
        return []
    return sorted(os.path.join(folder, f) for f in tqdm(names) if f.lower().endswith(ext))

def _can_read_random_window(fp, patch=128):
    try:
        with rasterio.open(fp) as src:
            H, W = src.height, src.width
            ph = min(H, patch); pw = min(W, patch)
            top  = 0 if H == ph else np.random.randint(0, max(1, H - ph + 1))
            left = 0 if W == pw else np.random.randint(0, max(1, W - pw + 1))
            win = Window(col_off=left, row_off=top, width=pw, height=ph)
            x = src.read(window=win, out_dtype='float32')
            return np.isfinite(x).any()
    except Exception:
        return False

def drop_unreadable_images(paths, max_workers=8, probe_patch=128):
    if not paths: return []
    kept = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut2fp = {ex.submit(_can_read_random_window, fp, probe_patch): fp for fp in paths}
        for fut in tqdm(as_completed(fut2fp), total=len(fut2fp), desc="filter(unreadable)"):
            fp = fut2fp[fut]
            try:
                if fut.result(): kept.append(fp)
            except Exception:
                pass
    kept.sort()
    return kept

# -------- filename keys (KEEP AS IS) --------
def _basename_key_a(fp):
    b = os.path.splitext(os.path.basename(fp))[0]
    parts = b.split('_')
    return parts[1] if len(parts) > 1 else b

def _basename_key_b(fp):
    b = os.path.splitext(os.path.basename(fp))[0]
    parts = b.split('_')
    return parts[0] if len(parts) > 0 else b
# -------------------------------------------

def find_triplets(ae_paths, ps_paths, s2_paths):
    print('Aerial'); d_ae = {_basename_key_a(p): p for p in ae_paths}
    print('PS');     d_ps = {_basename_key_a(p): p for p in ps_paths}
    print('S2');     d_s2 = {_basename_key_b(p): p for p in s2_paths}
    keys = sorted(set(d_ae) & set(d_ps) & set(d_s2))
    return [(d_ae[k], d_ps[k], d_s2[k]) for k in keys]

# -----------------------------
# 3) IMAGE UTILS & AUG
# -----------------------------
def make_finite_np(x: np.ndarray) -> np.ndarray:
    if not np.isfinite(x).all():
        x = np.where(np.isfinite(x), x, 0.0).astype(np.float32, copy=False)
    return x

def make_finite_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def normalise_patch(img, nodata_val=None, eps=1e-6):
    # per-patch z-score (per channel); replace non-finites and small-std
    x = img.astype(np.float32, copy=False)
    x = np.where(np.isfinite(x), x, np.nan)
    if nodata_val is not None:
        mask = np.any(x == nodata_val, axis=2)
        if mask.any():
            x = x.copy(); x[mask] = np.nan
    if not np.isfinite(x).any():
        return np.zeros_like(x, dtype=np.float32)
    mu = np.nanmean(x, axis=(0, 1))
    sd = np.nanstd( x, axis=(0, 1))
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd), sd, 1.0)
    sd = np.where(sd < eps, 1.0, sd)
    x = (x - mu) / sd
    x = np.where(np.isfinite(x), x, 0.0).astype(np.float32, copy=False)
    return x

def scale01_per_band(x_hwc: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x_hwc.astype(np.float32, copy=False)
    flat = x.reshape(-1, x.shape[2])
    mins = np.nanmin(flat, axis=0)
    maxs = np.nanmax(flat, axis=0)
    mins = np.where(np.isfinite(mins), mins, 0.0)
    maxs = np.where(np.isfinite(maxs), maxs, 1.0)
    denom = np.maximum(maxs - mins, eps)
    x = (x - mins) / denom
    x = np.clip(x, 0.0, 1.0)
    return x

def build_geo_augmenter(strength=1.0):
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0.0: return None
    return A.Compose([
        A.HorizontalFlip(p=0.5 * s),
        A.VerticalFlip(p=0.5 * s),
        A.RandomRotate90(p=0.3 * s),
        A.ElasticTransform(alpha=10 * s, sigma=10 * s, alpha_affine=0.0, p=0.20 * s),
    ])

def photometric_jitter_np(img: np.ndarray, s: float) -> np.ndarray:
    if s <= 0.0: return img
    x = img.astype(np.float32, copy=False)
    C = x.shape[2]
    gain = 1.0 + np.random.uniform(-0.15 * s, 0.15 * s, size=(1, 1, C)).astype(np.float32)
    bias = np.random.uniform(-0.10 * s, 0.10 * s, size=(1, 1, C)).astype(np.float32)
    chan_std = np.nanstd(x.reshape(-1, C), axis=0, keepdims=True).reshape(1, 1, C).astype(np.float32)
    noise = np.random.normal(0.0, (0.02 * s) * (chan_std + 1e-6), size=x.shape).astype(np.float32)
    return x * gain + bias + noise

def maybe_drop_channel_np(img: np.ndarray, p: float) -> np.ndarray:
    if p <= 0.0 or np.random.rand() > p: return img
    C = img.shape[2]
    c = int(np.random.randint(0, C))
    out = img.copy()
    out[..., c] = 0.0
    return out

# S2 downselect (12 → 10)
_first_s2_downselect_notice = False
def s2_downselect_if_needed(x_hwc: np.ndarray, expect_channels: int, keep_idx_from12: list[int]) -> np.ndarray:
    global _first_s2_downselect_notice
    C = x_hwc.shape[2]
    if C == expect_channels:
        return x_hwc
    if C == 12 and expect_channels == 10:
        x_hwc = x_hwc[..., keep_idx_from12]
        if not _first_s2_downselect_notice:
            print(f"[INFO] S2 downselect: detected 12 bands; keeping indices {keep_idx_from12} -> 10 bands "
                  f"(B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12).")
            _first_s2_downselect_notice = True
        return x_hwc
    raise RuntimeError(f"S2 has {C} channels; expected {expect_channels} (or 12→downselect→{expect_channels}).")

# Robust aligned windows with CRS handling
def _clamp_window_to_src(win: Window, src) -> Window:
    col_off = int(np.floor(win.col_off))
    row_off = int(np.floor(win.row_off))
    col_off = max(0, min(col_off, max(0, src.width - 1)))
    row_off = max(0, min(row_off, max(0, src.height - 1)))
    w_req = int(np.ceil(win.width))
    h_req = int(np.ceil(win.height))
    w = max(1, min(w_req, src.width - col_off))
    h = max(1, min(h_req, src.height - row_off))
    return Window(col_off, row_off, w, h)

def _read_resampled(src, win: Window, out_h: int, out_w: int, resampling=Resampling.bilinear):
    win_c = _clamp_window_to_src(win, src)
    try:
        return src.read(
            window=win_c,
            out_shape=(src.count, out_h, out_w),
            resampling=resampling,
            out_dtype='float32',
        )
    except Exception:
        try:
            return src.read(
                window=win,
                out_shape=(src.count, out_h, out_w),
                resampling=resampling,
                boundless=True,
                fill_value=np.nan,
                out_dtype='float32',
            )
        except Exception:
            return np.zeros((src.count, out_h, out_w), dtype=np.float32)

def _aligned_triplet_window(ae_fp, ps_fp, s2_fp, ph, pw):
    with rasterio.open(ae_fp) as ae, rasterio.open(ps_fp) as ps, rasterio.open(s2_fp) as s2:
        H, W = ae.height, ae.width
        ph_ = min(H, ph); pw_ = min(W, pw)
        top  = 0 if H == ph_ else np.random.randint(0, H - ph_ + 1)
        left = 0 if W == pw_ else np.random.randint(0, W - pw_ + 1)
        win_ae = Window(left, top, pw_, ph_)
        ae_bounds = win_bounds(win_ae, ae.transform)
        if ps.crs and ae.crs and ps.crs != ae.crs:
            ps_bounds = transform_bounds(ae.crs, ps.crs, *ae_bounds, densify_pts=21)
        else:
            ps_bounds = ae_bounds
        if s2.crs and ae.crs and s2.crs != ae.crs:
            s2_bounds = transform_bounds(ae.crs, s2.crs, *ae_bounds, densify_pts=21)
        else:
            s2_bounds = ae_bounds
        win_ps = from_bounds(*ps_bounds, transform=ps.transform)
        win_s2 = from_bounds(*s2_bounds, transform=s2.transform)
        x_ae = ae.read(window=win_ae, out_shape=(ae.count, ph, pw),
                       resampling=Resampling.bilinear, out_dtype='float32')
        x_ps = _read_resampled(ps, win_ps, ph, pw, resampling=Resampling.bilinear)
        x_s2 = _read_resampled(s2, win_s2, ph, pw, resampling=Resampling.bilinear)
    x_ae = np.transpose(x_ae, (1,2,0)).astype(np.float32, copy=False)
    x_ps = np.transpose(x_ps, (1,2,0)).astype(np.float32, copy=False)
    x_s2 = np.transpose(x_s2, (1,2,0)).astype(np.float32, copy=False)
    return x_ae, x_ps, x_s2

# -----------------------------
# 4) DATASET
# -----------------------------
class MultiSensorTriplets(Dataset):
    """
    Each __getitem__ returns a fresh random aligned crop from one triplet.

    Returns:
      {
        "basename": <str>,
        "AE":         FloatTensor [4,H,W]
        "PS":         FloatTensor [4,H,W]
        "S2L2A":      FloatTensor [10,H,W]
        "S2L2A_raw":  FloatTensor [10,H,W]  (unnormalized; for tokenizer)
      }
    """
    def __init__(self, triplets, patchsize, augmenter_strength, photo_strength, chan_drop_p, nodata_val,
                 expected_s2_channels: int, s2_keep_idx_from12: list[int],
                 repeats_per_epoch: int = 1, crops_per_file_per_epoch: int = 1,
                 photo_enable_epoch: int = 0):
        self.triplets = triplets
        self.ph, self.pw = patchsize
        self.nodata_val = nodata_val

        self.geo_aug  = build_geo_augmenter(augmenter_strength)
        self.photo_s  = float(np.clip(photo_strength, 0.0, 1.0))
        self.chan_drop_p = float(np.clip(chan_drop_p, 0.0, 1.0))

        self.expected_s2_channels = expected_s2_channels
        self.s2_keep_idx_from12   = s2_keep_idx_from12

        self.repeats = max(1, int(repeats_per_epoch))
        self.crops_per_file = max(1, int(crops_per_file_per_epoch))
        self._mult = self.repeats * self.crops_per_file

        # phase-in knobs
        self.cur_epoch = 0
        self.photo_enable_epoch = int(photo_enable_epoch)

    def __len__(self):
        return len(self.triplets) * self._mult

    def _apply_geo(self, img):
        if self.geo_aug is None: return img
        return self.geo_aug(image=img)["image"]

    def _apply_photo(self, img):
        # enable only after some epochs
        if self.cur_epoch < self.photo_enable_epoch: return img
        if self.photo_s <= 0.0: return img
        return photometric_jitter_np(img, self.photo_s)

    def _apply_chan_drop(self, img):
        if self.cur_epoch < self.photo_enable_epoch: return img
        return maybe_drop_channel_np(img, self.chan_drop_p)

    def __getitem__(self, idx):
        base_idx = idx % len(self.triplets)
        ae_fp, ps_fp, s2_fp = self.triplets[base_idx]
        base = os.path.splitext(os.path.basename(ae_fp))[0]  # (KEEP basename behavior!)

        x_ae_raw, x_ps_raw, x_s2_raw_full = _aligned_triplet_window(ae_fp, ps_fp, s2_fp, self.ph, self.pw)

        x_ae_raw      = self._apply_geo(x_ae_raw)
        x_ps_raw      = self._apply_geo(x_ps_raw)
        x_s2_raw_full = self._apply_geo(x_s2_raw_full)

        x_ae_raw      = self._apply_chan_drop(self._apply_photo(x_ae_raw))
        x_ps_raw      = self._apply_chan_drop(self._apply_photo(x_ps_raw))
        x_s2_raw_full = self._apply_chan_drop(self._apply_photo(x_s2_raw_full))

        x_ae_norm = normalise_patch(x_ae_raw, nodata_val=self.nodata_val)
        x_ps_norm = normalise_patch(x_ps_raw, nodata_val=self.nodata_val)
        x_s2_norm = normalise_patch(x_s2_raw_full, nodata_val=self.nodata_val)

        x_s2_norm = s2_downselect_if_needed(x_s2_norm, self.expected_s2_channels, self.s2_keep_idx_from12)
        x_s2_raw  = s2_downselect_if_needed(x_s2_raw_full, self.expected_s2_channels, self.s2_keep_idx_from12)

        if x_ae_norm.shape[2] != 4:  raise RuntimeError(f"AE {ae_fp} has {x_ae_norm.shape[2]} channels; expected 4.")
        if x_ps_norm.shape[2] != 4:  raise RuntimeError(f"PS {ps_fp} has {x_ps_norm.shape[2]} channels; expected 4.")
        if x_s2_norm.shape[2] != self.expected_s2_channels:
            raise RuntimeError(f"S2 {s2_fp} has {x_s2_norm.shape[2]} channels; expected {self.expected_s2_channels}.")

        x_ae  = make_finite_torch(torch.from_numpy(np.transpose(x_ae_norm, (2,0,1)).copy()).float())
        x_ps  = make_finite_torch(torch.from_numpy(np.transpose(x_ps_norm, (2,0,1)).copy()).float())
        x_s2  = make_finite_torch(torch.from_numpy(np.transpose(x_s2_norm, (2,0,1)).copy()).float())
        x_s2r = make_finite_torch(torch.from_numpy(np.transpose(x_s2_raw,  (2,0,1)).copy()).float())

        return {"basename": base, "AE": x_ae, "PS": x_ps, "S2L2A": x_s2, "S2L2A_raw": x_s2r}

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)

def make_loader(cfg, triplets):
    ds = MultiSensorTriplets(
        triplets=triplets,
        patchsize=cfg.patch_size,
        augmenter_strength=cfg.augmenter_strength,
        photo_strength=cfg.photo_jitter_strength,
        chan_drop_p=cfg.channel_drop_prob,
        nodata_val=cfg.nodata_val,
        expected_s2_channels=cfg.s2l2a_expect_channels,
        s2_keep_idx_from12=cfg.s2l2a_from12_keep_idx,
        repeats_per_epoch=cfg.repeats_per_epoch,
        crops_per_file_per_epoch=cfg.crops_per_file_per_epoch,
        photo_enable_epoch=cfg.photo_enable_epoch
    )
    g = torch.Generator().manual_seed(cfg.seed)
    dl_kwargs = dict(dataset=ds, batch_size=cfg.batch_size, shuffle=True,
                     num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                     worker_init_fn=seed_worker, generator=g)
    if cfg.num_workers > 0:
        dl_kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**dl_kwargs)

# -----------------------------
# 5) SHARED BACKBONE WRAPPER (per-mod tokens)
# -----------------------------
from terratorch.registry import BACKBONE_REGISTRY

def build_shared_backbone_s2l1c_s2l2a(bands_10):
    bands = {
        "S2L1C": ["BLUE", "GREEN", "RED", "NIR_NARROW"],  # AE/PS (4)
        "S2L2A": bands_10                                 # S2 (10)
    }
    model = BACKBONE_REGISTRY.build(
        "terramind_v1_base",
        pretrained=True,
        modalities=["S2L1C", "S2L2A"],
        bands=bands,
        merge_method="mean"
    )
    return model

class TerraMindPerModEncoder(nn.Module):
    """
    Shared TerraMind backbone that exposes per-modality tokens.
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.stems = self.backbone.encoder_embeddings
        self.mod_map = self.backbone.mod_name_mapping

        # locate trunk (encoder)
        self.trunk = None
        for cand in ["encoder", "trunk", "transformer", "encoder_layers", "blocks", "layers"]:
            if hasattr(self.backbone, cand):
                self.trunk = getattr(self.backbone, cand)
                break
        if self.trunk is None:
            raise AttributeError("Could not locate TerraMind trunk (encoder).")

        self.final_norm = getattr(self.backbone, "final_norm", None)

        # per-mod patch sizes
        self.patch_size_by_mod = {}
        for mod in ["S2L1C", "S2L2A"]:
            key = self.mod_map[mod]
            ps = getattr(self.stems[key], "patch_size", (16, 16))
            self.patch_size_by_mod[mod] = ps

        # learned mask tokens (init with small noise; width corrected at first use)
        self.mask_token = nn.ParameterDict({
            "S2L1C": nn.Parameter(torch.zeros(1, 1, 1024)),
            "S2L2A": nn.Parameter(torch.zeros(1, 1, 1024)),
        })
        for k in self.mask_token:
            nn.init.normal_(self.mask_token[k], std=0.02)
        self._mask_dim_inited = False

    @staticmethod
    def _to_tokens(out) -> torch.Tensor:
        if torch.is_tensor(out): return out
        if isinstance(out, dict):
            for k in ("tokens", "x", "last_hidden_state"):
                v = out.get(k, None)
                if torch.is_tensor(v): return v
        if isinstance(out, (list, tuple)) and len(out) > 0:
            last = out[-1]
            if torch.is_tensor(last): return last
            if hasattr(last, "last_hidden_state") and torch.is_tensor(last.last_hidden_state):
                return last.last_hidden_state
        if hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
            return out.last_hidden_state
        raise TypeError(f"Cannot extract tokens from object of type {type(out)}")

    def _ensure_mask_dim(self, E: int):
        if not self._mask_dim_inited:
            for mod in self.mask_token:
                if self.mask_token[mod].shape[-1] != E:
                    new_param = torch.zeros(1, 1, E,
                                            device=self.mask_token[mod].device,
                                            dtype=self.mask_token[mod].dtype)
                    nn.init.normal_(new_param, std=0.02)
                    self.mask_token[mod] = nn.Parameter(new_param)
            self._mask_dim_inited = True

    def stem_tokens(self, mod_name: str, x: torch.Tensor) -> torch.Tensor:
        key = self.mod_map[mod_name]
        out = self.stems[key](x)
        return self._to_tokens(out)  # [B,T,E]

    def _forward_block(self, layer: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        out = layer(tokens)
        return self._to_tokens(out) if not torch.is_tensor(out) else out

    def run_trunk(self, tokens: torch.Tensor) -> torch.Tensor:
        if callable(self.trunk) and not isinstance(self.trunk, (nn.ModuleList, list, tuple)):
            out = self.trunk(tokens)
            toks = self._to_tokens(out) if not torch.is_tensor(out) else out
        else:
            toks = tokens
            for layer in self.trunk:
                toks = self._forward_block(layer, toks)
        if self.final_norm is not None:
            toks = self.final_norm(toks)
        return toks  # [B,T,E]

    def forward_tokens(self, mod_name: str, x: torch.Tensor, mask_bool: torch.Tensor | None = None) -> torch.Tensor:
        toks = self.stem_tokens(mod_name, x)  # [B,T,E]
        B, T, E = toks.shape
        self._ensure_mask_dim(E)
        if mask_bool is not None:
            mtok = self.mask_token[mod_name].expand(B, T, E)
            toks = torch.where(mask_bool.unsqueeze(-1), mtok, toks)
        toks = self.run_trunk(toks)
        return toks

# -----------------------------
# 6) TOKEN GRID HELPERS
# -----------------------------
def token_grid_from_hw_ps(H: int, W: int, ps: Tuple[int,int]) -> Tuple[int,int,int]:
    ph, pw = ps
    pad_h = (ph - H % ph) % ph
    pad_w = (pw - W % pw) % pw
    Hpad, Wpad = H + pad_h, W + pad_w
    h_p, w_p = Hpad // ph, Wpad // pw
    T = h_p * w_p
    return h_p, w_p, T

def tokens_align_to_grid(toks: torch.Tensor, src_hw: Tuple[int,int], src_ps: Tuple[int,int],
                         tgt_hw: Tuple[int,int], tgt_ps: Tuple[int,int]) -> torch.Tensor:
    B, Tsrc, E = toks.shape
    hs, ws, Tsrc_check = token_grid_from_hw_ps(src_hw[0], src_hw[1], src_ps)
    ht, wt, Ttgt       = token_grid_from_hw_ps(tgt_hw[0], tgt_hw[1], tgt_ps)
    if Tsrc != Tsrc_check:
        Tmin = min(Tsrc, Tsrc_check)
        toks = toks[:, :Tmin, :]
        hs = int(np.sqrt(Tmin)); ws = max(1, Tmin // hs)
    feat = toks.view(B, hs, ws, E).permute(0, 3, 1, 2)      # [B,E,hs,ws]
    feat = F.interpolate(feat, size=(ht, wt), mode="nearest")
    out  = feat.permute(0, 2, 3, 1).contiguous().view(B, ht*wt, E)
    return out

# -----------------------------
# 7) OPTIONAL S2 TOKENIZER
# -----------------------------
class S2Tokenizer:
    def __init__(self, path: Optional[str], func: Optional[Callable], num_codes: Optional[int]):
        self._jit = None
        self._func = None
        self.num_codes = num_codes
        if path is not None:
            try:
                self._jit = torch.jit.load(path, map_location="cpu")
                self._jit.eval()
                if hasattr(self._jit, "num_codes"):
                    try:
                        self.num_codes = int(self._jit.num_codes)
                    except Exception:
                        pass
                print(f"[Tokenizer] Loaded TorchScript tokenizer from: {path} (K={self.num_codes})")
            except Exception as e:
                warnings.warn(f"Failed to load tokenizer from {path}: {e}")
        if func is not None:
            if callable(func):
                self._func = func
                print(f"[Tokenizer] Using provided Python callable (K={self.num_codes})")
            else:
                warnings.warn("s2_tokenizer_callable is not callable; ignoring.")

        if self._jit is None and self._func is None:
            print("[Tokenizer] No tokenizer configured; Token-CE will be disabled.")

        if self.is_available() and self.num_codes is None:
            warnings.warn("Tokenizer loaded but num_codes is unknown. Please set cfg.s2_tokenizer_num_codes.")
    
    def is_available(self) -> bool:
        return (self._jit is not None) or (self._func is not None)

    @torch.inference_mode()
    def encode(self, x_hwc01: np.ndarray) -> np.ndarray:
        if self._jit is not None:
            inp = torch.from_numpy(x_hwc01).permute(2,0,1).unsqueeze(0).float()
            out = self._jit(inp)
            if isinstance(out, dict):
                out = next((v for v in out.values() if torch.is_tensor(v)), None)
            if out is None or not torch.is_tensor(out):
                raise RuntimeError("Tokenizer TorchScript returned unsupported type.")
            codes = out.squeeze(0).detach().cpu()
            if codes.ndim == 2: codes = codes.reshape(-1)
            elif codes.ndim != 1: codes = codes.view(-1)
            return codes.to(torch.long).numpy()
        elif self._func is not None:
            codes = self._func(x_hwc01)
            if isinstance(codes, np.ndarray):
                if codes.ndim == 2: codes = codes.reshape(-1)
                elif codes.ndim != 1: codes = codes.reshape(-1)
                return codes.astype(np.int64, copy=False)
            else:
                codes = np.array(codes, dtype=np.int64).reshape(-1)
                return codes
        else:
            raise RuntimeError("Tokenizer not available")

# -----------------------------
# 8) PATCHIFY / UNPATCHIFY
# -----------------------------
def patchify_images(x: torch.Tensor, ps: Tuple[int,int]) -> Tuple[torch.Tensor, Tuple[int,int]]:
    B, C, H, W = x.shape
    ph, pw = ps
    pad_h = (ph - H % ph) % ph
    pad_w = (pw - W % pw) % pw
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w if W % pw != 0 else 0, 0, pad_h if H % ph != 0 else 0), mode='reflect')
        H, W = x.shape[-2:]
    h_p, w_p = H // ph, W // pw
    patches = x.unfold(2, ph, ph).unfold(3, pw, pw)          # [B,C,h_p,w_p,ph,pw]
    patches = patches.contiguous().view(B, C, h_p*w_p, ph*pw) # [B,C,T,ph*pw]
    patches = patches.permute(0,2,3,1).contiguous().view(B, h_p*w_p, ph*pw*C)  # [B,T,ph*pw*C]
    return patches, (h_p, w_p)

def unpatchify_images(patches: torch.Tensor, grid_hw: Tuple[int,int], ps: Tuple[int,int], C: int) -> torch.Tensor:
    # patches: [B,T,ph*pw*C] -> [B,C,H,W] (padded HW)
    B, T, PPC = patches.shape
    ph, pw = ps
    h_p, w_p = grid_hw
    patches = patches.view(B, T, ph*pw, C)                    # [B,T,ph*pw,C]
    patches = patches.permute(0, 3, 1, 2).contiguous()        # [B,C,T,ph*pw]
    patches = patches.view(B, C, h_p, w_p, ph, pw)            # [B,C,h_p,w_p,ph,pw]
    x = patches.permute(0,1,2,4,3,5).contiguous().view(B, C, h_p*ph, w_p*pw)
    return x

# -----------------------------
# 9) LIGHTNING MODULE
# -----------------------------
class CrossModalMIMNCETokenCE(L.LightningModule):
    def __init__(self, cfg: Configuration, train_dataset: Optional[Dataset] = None):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg", "train_dataset"])
        self.cfg = cfg
        self._train_dataset_ref = train_dataset  # so we can update epoch inside dataset

        # Backbone
        shared = build_shared_backbone_s2l1c_s2l2a(cfg.s2l2a_bands)
        self.enc = TerraMindPerModEncoder(shared)

        # Projection head for InfoNCE
        self.proj = nn.Sequential(
            nn.LazyLinear(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.cfg.proj_dropout),
            nn.Linear(2048, 128)
        )

        # Stronger decoder head per modality (tiny MLP)
        self.recon_heads = nn.ModuleDict()
        self.C_by_mod = {"S2L1C": 4, "S2L2A": len(cfg.s2l2a_bands)}
        for mod in ["S2L1C", "S2L2A"]:
            ph, pw = self.enc.patch_size_by_mod[mod]
            C = self.C_by_mod[mod]
            out_dim = ph * pw * C
            self.recon_heads[mod] = nn.Sequential(
                nn.LazyLinear(4 * 1024),
                nn.GELU(),
                nn.Linear(4 * 1024, out_dim)
            )

        # Optional tokenizer + token-CE head
        self.tokenizer = S2Tokenizer(cfg.s2_tokenizer_path, cfg.s2_tokenizer_callable, cfg.s2_tokenizer_num_codes)
        self.token_ce_head: Optional[nn.Module] = None
        if self.tokenizer.is_available():
            K = self.cfg.s2_tokenizer_num_codes if self.cfg.s2_tokenizer_num_codes is not None else 1024
            self.token_ce_head = nn.LazyLinear(K)

        # schedules
        self._optim_groups_built = False

        # optionally freeze trunk initially
        if self.cfg.freeze_trunk_epochs > 0:
            for p in self.enc.backbone.parameters():
                p.requires_grad = False

    # ----- schedules -----
    @property
    def current_mask_ratio(self) -> float:
        e = max(0, self.current_epoch)
        E = max(1, self.cfg.epochs - 1)
        r = self.cfg.mim_mask_ratio_min + (self.cfg.mim_mask_ratio_max - self.cfg.mim_mask_ratio_min) * (e / E)
        return float(np.clip(r, 0.0, 0.95))

    def _lr_lambda(self, epoch: int):
        if epoch < self.cfg.warmup_epochs:
            return max(1e-6, (epoch + 1) / max(1, self.cfg.warmup_epochs))
        t = (epoch - self.cfg.warmup_epochs) / max(1, (self.cfg.epochs - self.cfg.warmup_epochs))
        cos = 0.5 * (1 + np.cos(np.pi * t))
        return self.cfg.min_lr_ratio + (1.0 - self.cfg.min_lr_ratio) * cos

    # ----- helpers -----
    @staticmethod
    def _make_mask(B, T, ratio, device):
        num_mask = max(1, int(T * ratio))
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(T, device=device)[:num_mask]
            mask[b, idx] = True
        return mask

    def _encode_pool(self, toks: torch.Tensor) -> torch.Tensor:
        emb = toks.mean(dim=1)
        z = self.proj(emb)
        if self.cfg.repr_dropout_prob > 0.0 and self.training:
            drop = torch.rand_like(z[..., :1]) < self.cfg.repr_dropout_prob
            z = torch.where(drop, torch.zeros_like(z), z)
        z = F.normalize(z, dim=-1, eps=1e-6)
        return z

    @staticmethod
    def _nce(z1, z2, temperature):
        z1f, z2f = z1.float(), z2.float()
        logits = (z1f @ z2f.t()) / float(temperature)
        labels = torch.arange(z1f.size(0), device=z1f.device)
        loss = F.cross_entropy(logits, labels)
        return loss.to(z1.dtype)

    @staticmethod
    def _psnr_from_mse(mse: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # assume unit dynamic range after standardization for relative tracking
        return -10.0 * torch.log10(mse + eps)

    # ----- losses -----
    def _mim_loss_for_mod(self, mod: str, x: torch.Tensor, mask: torch.Tensor):
        ps = self.enc.patch_size_by_mod[mod]
        tgt_patches, grid_hw = patchify_images(x, ps)                # [B,T,ph*pw*C]
        toks = self.enc.forward_tokens(mod, x, mask)                 # [B,T,E]
        pred = self.recon_heads[mod](toks)                            # [B,T,ph*pw*C]

        loss = torch.tensor(0.0, device=x.device, dtype=pred.dtype)
        mse_baseline = torch.tensor(0.0, device=x.device, dtype=pred.dtype)
        psnr = torch.tensor(0.0, device=x.device, dtype=pred.dtype)

        if mask.any():
            diff = (pred.float()[mask] - tgt_patches.float()[mask])
            loss = (diff ** 2).mean().to(pred.dtype)

            # baseline (zero predictor due to z-score targets)
            mse_baseline = (tgt_patches.float()[mask] ** 2).mean().to(pred.dtype)
            psnr = self._psnr_from_mse(loss)

        return loss, mse_baseline, psnr, (tgt_patches, pred, grid_hw, ps)

    def _token_ce_loss(self, batch, toks_ae_full: torch.Tensor, toks_ps_full: torch.Tensor) -> torch.Tensor:
        if not (self.tokenizer.is_available() and self.token_ce_head is not None):
            return torch.tensor(0.0, device=toks_ae_full.device, dtype=toks_ae_full.dtype)

        B, C, Hs2, Ws2 = batch["S2L2A"].shape
        ps_s2 = self.enc.patch_size_by_mod["S2L2A"]
        hs2, ws2, T_s2 = token_grid_from_hw_ps(Hs2, Ws2, ps_s2)
        mask_s2 = self._make_mask(B, T_s2, self.current_mask_ratio, batch["S2L2A"].device)

        # Align AE/PS tokens to S2 grid
        H4, W4 = batch["AE"].shape[-2:]
        ps_4 = self.enc.patch_size_by_mod["S2L1C"]
        toks_ae_aln = tokens_align_to_grid(toks_ae_full, (H4, W4), ps_4, (Hs2, Ws2), ps_s2)
        toks_ps_aln = tokens_align_to_grid(toks_ps_full, (H4, W4), ps_4, (Hs2, Ws2), ps_s2)
        fused = 0.5 * (toks_ae_aln + toks_ps_aln)  # [B,T_s2,E]

        # Token targets from RAW S2
        x_s2r = batch["S2L2A_raw"].permute(0,2,3,1).detach().cpu().numpy()
        targets = []
        for b in range(B):
            patch = x_s2r[b]
            patch01 = scale01_per_band(patch) if self.cfg.s2_tokenizer_input_scale_mode == "minmax" \
                                               else np.clip(patch, 0.0, 1.0).astype(np.float32, copy=False)
            try:
                codes_flat = self.tokenizer.encode(patch01)
            except Exception as e:
                warnings.warn(f"Tokenizer encode failed: {e}")
                return torch.tensor(0.0, device=fused.device, dtype=fused.dtype)
            codes_t = torch.from_numpy(codes_flat).to(fused.device, non_blocking=True)
            if codes_t.numel() != T_s2:
                side = int(np.sqrt(float(codes_t.numel())))
                if side * side == codes_t.numel():
                    grid = codes_t.view(1, 1, side, side).float()
                    grid = F.interpolate(grid, size=(hs2, ws2), mode="nearest").long().squeeze(0).squeeze(0)
                    codes_t = grid.reshape(-1)
                else:
                    Tmin = min(T_s2, codes_t.numel())
                    codes_t = codes_t[:Tmin]
                    if Tmin < T_s2:
                        pad = torch.zeros(T_s2 - Tmin, dtype=codes_t.dtype, device=codes_t.device)
                        codes_t = torch.cat([codes_t, pad], dim=0)
            targets.append(codes_t)
        targets = torch.stack(targets, dim=0)  # [B,T_s2]

        logits = self.token_ce_head(fused)  # [B,T_s2,K]

        if mask_s2.any():
            loss = F.cross_entropy(logits.float()[mask_s2], targets.long()[mask_s2], ignore_index=-100)
            return loss.to(logits.dtype)
        else:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # ----- optim -----
    def configure_optimizers(self):
        if not self._optim_groups_built:
            trunk_params, head_params = [], []
            trunk_ids = set()
            for m in [self.enc.backbone]:
                for p in m.parameters():
                    trunk_params.append(p); trunk_ids.add(id(p))
            for p in self.parameters():
                if id(p) not in trunk_ids:
                    head_params.append(p)
            self._optim_groups_built = True

            self._opt_groups = [
                dict(params=[p for p in trunk_params if p.requires_grad], lr=self.cfg.lr_trunk, weight_decay=self.cfg.weight_decay),
                dict(params=[p for p in head_params if p.requires_grad],  lr=self.cfg.lr_heads, weight_decay=self.cfg.weight_decay),
            ]

        opt = torch.optim.AdamW(self._opt_groups, betas=(0.9, 0.999), eps=1e-8)
        lr_lambda = lambda epoch: self._lr_lambda(epoch)  # noqa: E731
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    # ----- hooks -----
    def on_train_epoch_start(self):
        # unfreeze trunk
        if self.current_epoch == self.cfg.freeze_trunk_epochs:
            for p in self.enc.backbone.parameters(): p.requires_grad = True

        # forward epoch info into dataset (for phased-in augs)
        try:
            if self._train_dataset_ref is not None:
                self._train_dataset_ref.cur_epoch = self.current_epoch
            else:
                ds = self.trainer.train_dataloader.dataset
                if hasattr(ds, "cur_epoch"):
                    ds.cur_epoch = self.current_epoch
        except Exception:
            pass

        # enable InfoNCE once we reach the start epoch
        if self.current_epoch == self.cfg.nce_start_epoch and self.cfg.lambda_nce == 0.0:
            self.cfg.lambda_nce = 0.1  # switch on

    # ----- viz helpers (fixed) -----
    @staticmethod
    def _band3_quick_rgb(
        x: torch.Tensor,
        mod: str,
        use_quantiles: bool = True,
        q_lo: float = 0.02,
        q_hi: float = 0.98,
        gamma: float = 1.0,
    ) -> torch.Tensor:
        """
        x: [B,C,H,W]  -> [B,3,H,W] in [0,1]
        AE/PS: channels [BLUE, GREEN, RED, NIR] -> RGB = [2,1,0]
        S2:    [B02,B03,B04,...]                -> RGB = [2,1,0] (B04,B03,B02)
        """
        r, g, b = (2, 1, 0)
        sel = torch.stack([x[:, r], x[:, g], x[:, b]], dim=1)  # [B,3,H,W]

        if use_quantiles:
            B = sel.shape[0]
            flat = sel.view(B, 3, -1)
            lo = torch.quantile(flat, q_lo, dim=-1).view(B, 3, 1, 1)
            hi = torch.quantile(flat, q_hi, dim=-1).view(B, 3, 1, 1)
            out = (sel - lo) / (hi - lo + 1e-6)
        else:
            B = sel.shape[0]
            flat = sel.view(B, 3, -1)
            mn = flat.min(-1, keepdim=True).values.view(B, 3, 1, 1)
            mx = flat.max(-1, keepdim=True).values.view(B, 3, 1, 1)
            out = (sel - mn) / (mx - mn + 1e-6)

        out = out.clamp(0, 1)
        if abs(gamma - 1.0) > 1e-6:
            out = out.pow(1.0 / gamma)
        return out

    @staticmethod
    def _overlay_mask(rgb_01: torch.Tensor, mask_01: torch.Tensor,
                      color=(1.0, 0.0, 0.0), alpha: float = 0.35) -> torch.Tensor:
        """rgb_01: [B,3,H,W], mask_01: [B,1,H,W] (0..1), returns [B,3,H,W]."""
        col = torch.tensor(color, device=rgb_01.device, dtype=rgb_01.dtype).view(1, 3, 1, 1)
        m3 = mask_01.clamp(0, 1).repeat(1, 3, 1, 1)
        return (1 - alpha * m3) * rgb_01 + (alpha * m3) * col

    def _log_viz(self, batch, mask_ae, mask_ps, mask_s2,
                 aux_ae, aux_ps, aux_s2):
        """
        Per modality: [input | input+mask overlay | target(masked) | pred(masked)].
        Reconstructions/masks are cropped back to original H×W for proper alignment.
        """
        try:
            if not hasattr(self.logger, "experiment"):
                return
            tb = self.logger.experiment
        except Exception:
            return

        with torch.no_grad():
            b = 0  # visualize first item

            def make_panel(mod, x_full, mask_tokens, aux):
                # unpack
                tgt_patches, pred_patches, grid_hw, ps = aux
                B = x_full.shape[0]
                C, H, W = x_full.shape[1], x_full.shape[2], x_full.shape[3]
                ph, pw = ps
                h_p, w_p = grid_hw
                Hpad, Wpad = h_p * ph, w_p * pw

                # unpatchify to padded size, then crop to H×W
                tgt_img  = unpatchify_images(tgt_patches, grid_hw, ps, C)[..., :H, :W]
                pred_img = unpatchify_images(pred_patches, grid_hw, ps, C)[..., :H, :W]

                # token mask -> pixel mask at padded size, then crop to H×W
                mask_img = mask_tokens.view(B, 1, h_p, w_p).float().repeat(1, 1, ph, pw)
                if (Hpad, Wpad) != (H, W):
                    mask_img = mask_img[..., :H, :W]
                mask_img = mask_img.clamp(0, 1)

                # masked targets/preds
                tgt_m = tgt_img * mask_img
                pre_m = pred_img * mask_img

                # to RGB + overlay
                rgb_in  = self._band3_quick_rgb(x_full,   "S2L2A" if mod == "S2L2A" else "S2L1C")[b:b+1]
                rgb_msk = self._overlay_mask(rgb_in, mask_img[b:b+1])
                rgb_tgt = self._band3_quick_rgb(tgt_m,   "S2L2A" if mod == "S2L2A" else "S2L1C")[b]
                rgb_pre = self._band3_quick_rgb(pre_m,   "S2L2A" if mod == "S2L2A" else "S2L1C")[b]

                panel = torch.cat([rgb_in[0], rgb_msk[0], rgb_tgt, rgb_pre], dim=-1).clamp(0, 1)
                return panel

            pa  = make_panel("S2L1C", batch["AE"],    mask_ae, aux_ae)
            pp  = make_panel("S2L1C", batch["PS"],    mask_ps, aux_ps)
            ps2 = make_panel("S2L2A", batch["S2L2A"], mask_s2, aux_s2)

            step = int(self.global_step)
            tb.add_image("viz/AE/S2L1C",  pa,  step)
            tb.add_image("viz/PS/S2L1C",  pp,  step)
            tb.add_image("viz/S2/S2L2A",  ps2, step)

    # ----- training -----
    def training_step(self, batch, batch_idx):
        for k in ("AE", "PS", "S2L2A", "S2L2A_raw"):
            batch[k] = torch.nan_to_num(batch[k], nan=0.0, posinf=0.0, neginf=0.0)

        assert batch["AE"].shape[1] == 4 and batch["PS"].shape[1] == 4, "AE/PS must be 4-band"
        assert batch["S2L2A"].shape[1] == len(self.cfg.s2l2a_bands), f"S2 must be {len(self.cfg.s2l2a_bands)}-band after downselect"

        # Build masks with scheduled ratio
        ratio = self.current_mask_ratio
        toks_probe_ae = self.enc.stem_tokens("S2L1C", batch["AE"]);   B, T_ae, _ = toks_probe_ae.shape
        toks_probe_ps = self.enc.stem_tokens("S2L1C", batch["PS"]);   T_ps = toks_probe_ps.shape[1]
        toks_probe_s2 = self.enc.stem_tokens("S2L2A", batch["S2L2A"]); T_s2 = toks_probe_s2.shape[1]
        mask_ae = self._make_mask(B, T_ae, ratio, batch["AE"].device)
        mask_ps = self._make_mask(B, T_ps, ratio, batch["PS"].device)
        mask_s2 = self._make_mask(B, T_s2, ratio, batch["S2L2A"].device)

        # MIM per modality (returns loss, zero-baseline, PSNR, aux for viz)
        loss_mim_ae, base_ae, psnr_ae, aux_ae = self._mim_loss_for_mod("S2L1C", batch["AE"],    mask_ae)
        loss_mim_ps, base_ps, psnr_ps, aux_ps = self._mim_loss_for_mod("S2L1C", batch["PS"],    mask_ps)
        loss_mim_s2, base_s2, psnr_s2, aux_s2 = self._mim_loss_for_mod("S2L2A", batch["S2L2A"], mask_s2)
        loss_mim = loss_mim_ae + loss_mim_ps + loss_mim_s2
        zero_baseline = base_ae + base_ps + base_s2
        mim_improvement = zero_baseline - loss_mim

        # Full tokens (unmasked) for InfoNCE & Token-CE
        toks_ae_full = self.enc.forward_tokens("S2L1C", batch["AE"], mask_bool=None)
        toks_ps_full = self.enc.forward_tokens("S2L1C", batch["PS"], mask_bool=None)
        toks_s2_full = self.enc.forward_tokens("S2L2A", batch["S2L2A"], mask_bool=None)

        loss_nce = torch.tensor(0.0, device=loss_mim.device, dtype=loss_mim.dtype)
        if self.current_epoch >= self.cfg.nce_start_epoch and self.cfg.lambda_nce > 0.0:
            z_ae, z_ps, z_s2 = self._encode_pool(toks_ae_full), self._encode_pool(toks_ps_full), self._encode_pool(toks_s2_full)
            if batch_idx % 3 == 0:
                loss_nce = self._nce(z_ae, z_ps, self.cfg.temperature) + self._nce(z_ps, z_ae, self.cfg.temperature)
            elif batch_idx % 3 == 1:
                loss_nce = self._nce(z_ae, z_s2, self.cfg.temperature) + self._nce(z_s2, z_ae, self.cfg.temperature)
            else:
                loss_nce = self._nce(z_ps, z_s2, self.cfg.temperature) + self._nce(z_s2, z_ps, self.cfg.temperature)

        loss_tokence = torch.tensor(0.0, device=loss_mim.device, dtype=loss_mim.dtype)
        if self.tokenizer.is_available() and self.token_ce_head is not None:
            loss_tokence = self._token_ce_loss(batch, toks_ae_full, toks_ps_full)

        loss = self.cfg.lambda_mim * loss_mim + self.cfg.lambda_nce * loss_nce
        if self.tokenizer.is_available() and self.token_ce_head is not None:
            loss = loss + self.cfg.lambda_tokence * loss_tokence

        if torch.isnan(loss):
            warnings.warn("NaN in loss; clamping to zero for this step.")
            loss = torch.nan_to_num(loss, nan=0.0)

        # Logs (make improvement visible vs zero baseline)
        self.log_dict({
            "train/mim": loss_mim,
            "train/mim_zero_baseline": zero_baseline,
            "train/mim_improvement": mim_improvement,
            "train/mim_psnr_ae": psnr_ae,
            "train/mim_psnr_ps": psnr_ps,
            "train/mim_psnr_s2": psnr_s2,
            "train/nce": loss_nce,
            "train/tokence": loss_tokence,
            "train/total": loss,
            "meta/mask_ratio": torch.tensor(self.current_mask_ratio, device=loss.device),
        }, prog_bar=True, on_step=True, on_epoch=True)

        # TB images every N steps
        if (self.global_step % max(1, int(self.cfg.viz_every_steps))) == 0:
            self._log_viz(batch, mask_ae, mask_ps, mask_s2, aux_ae, aux_ps, aux_s2)

        return loss

# -----------------------------
# 10) DRIVER
# -----------------------------
class PrintEveryStep(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss_val = float('nan')
        try:
            meter = trainer.callback_metrics.get("train/total", None)
            if isinstance(meter, torch.Tensor):
                loss_val = float(meter.detach().cpu())
            elif meter is not None:
                loss_val = float(meter)
        except Exception:
            pass
        names = batch.get("basename", "")
        names = names if isinstance(names, list) else [names]
        print(f"[epoch {trainer.current_epoch} | step {trainer.global_step}] "
              f"loss={loss_val:.4f} batch_idx={batch_idx} samples={names[:3]}")

class SetDatasetEpoch(Callback):
    """Keep dataset.cur_epoch in sync with trainer.current_epoch (for phased-in augs)."""
    def __init__(self, dataset: Dataset):
        super().__init__()
        self.ds = dataset
    def on_train_epoch_start(self, trainer, pl_module):
        try:
            self.ds.cur_epoch = trainer.current_epoch
        except Exception:
            pass

def continue_pretraining(cfg: Configuration):
    # single GPU pin
    if cfg.devices == 1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_index)

    os.environ.setdefault('OMP_NUM_THREADS', '1')
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    os.makedirs(cfg.logs_dir, exist_ok=True)
    os.makedirs(cfg.models_dir, exist_ok=True)

    print('Starting continual pretraining...\nDiscovering images...')
    ae_paths = list_files(cfg.aerial_dir, cfg.pretrain_ext)
    ps_paths = list_files(cfg.ps_dir,     cfg.pretrain_ext)
    s2_paths = list_files(cfg.s2_dir,     cfg.pretrain_ext)

    ae_paths_f = drop_unreadable_images(ae_paths)
    ps_paths_f = drop_unreadable_images(ps_paths)
    s2_paths_f = drop_unreadable_images(s2_paths)
    print(f"Readable: AE {len(ae_paths_f)}/{len(ae_paths)}, PS {len(ps_paths_f)}/{len(ps_paths)}, S2 {len(s2_paths_f)}/{len(s2_paths)}")

    print('Matching triplets...')
    triplets = find_triplets(ae_paths_f, ps_paths_f, s2_paths_f)
    print(f'Using {len(triplets)} triplets.')
    if len(triplets) == 0:
        raise RuntimeError("No matched AE/PS/S2 triplets found. Check filenames and folders.")

    print('Building dataloader...')
    train_loader = make_loader(cfg, triplets)
    train_dataset = train_loader.dataset  # for phased-in augment

    print('Building model...')
    model = CrossModalMIMNCETokenCE(cfg, train_dataset=train_dataset)

    ckpt = ModelCheckpoint(
        dirpath=cfg.models_dir,
        filename="cm_mimnce_tokence-{epoch:02d}-{global_step}",
        save_top_k=3,
        monitor="train/total",
        mode="min",
        save_last=True
    )
    lrmon = LearningRateMonitor(logging_interval='step')

    print('Starting training...')
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu",
        devices=(1 if cfg.devices == 1 else cfg.devices),
        strategy=("auto" if cfg.devices == 1 else cfg.strategy),
        precision="16-mixed" if cfg.fp16 else "32-true",
        default_root_dir=cfg.logs_dir,
        callbacks=[ckpt, lrmon, PrintEveryStep(), SetDatasetEpoch(train_dataset)],
        log_every_n_steps=1,
        enable_progress_bar=True,
        gradient_clip_val=1.0
    )

    ckpt_path = cfg.continue_model_path if cfg.continue_model_path else None
    trainer.fit(model, train_loader, ckpt_path=ckpt_path)

    shared_path = os.path.join(cfg.models_dir, "terramind_shared_backbone_continued.pth")
    torch.save(model.enc.backbone.state_dict(), shared_path)
    print("Training complete.")
    print(f"Checkpoints saved in: {cfg.models_dir}")
    print(f"Shared TerraMind backbone (continued) saved to: {shared_path}")
    return model

# -----------------------------
# 11) ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    if config.devices == 1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_index)
    continue_pretraining(config)
