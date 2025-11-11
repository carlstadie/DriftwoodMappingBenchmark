# core/split_frames.py
# Train/val/test splitting utilities
# - cross_validation_split: K-fold indices (saved/loaded from JSON)
# - split_dataset: stratified-by-positive-rate (with safe fallbacks), saved/loaded from JSON

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import KFold, train_test_split

# Console style similar to training banner output
try:
    from core.common.console import _C, _col
except Exception:  # fallback if console helpers aren't available
    class _C:  # noqa: N801 (kept to mirror existing helper names)
        RED = ""
        YELLOW = ""
        GREEN = ""
        RESET = ""

    def _col(s, _color):
        return s


def _ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist (no-op for empty)."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def summarize_positive_rates(
    frames: Sequence[Any],
    sets: Dict[str, Sequence[int]],
) -> Dict[str, Dict[str, float]]:
    """
    Return summary stats (in percent) of per-frame positive rates for each set.

    Why:
        Helpful sanity check to see whether train/val/test are similarly
        distributed with respect to foreground prevalence.
    """
    pr = _pos_rate(frames) * 100.0
    out: Dict[str, Dict[str, float]] = {}
    for name, idx in sets.items():
        idx = np.asarray(idx, dtype=int)
        vals = pr[idx] if idx.size > 0 else np.asarray([], dtype=float)
        if vals.size == 0:
            out[name] = {
                "n": 0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        else:
            out[name] = {
                "n": int(vals.size),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    return out


# ---------- K-fold CV helper (not used right now) ----------


def cross_validation_split(
    frames: Sequence[Any],
    frames_json: str,
    patch_dir: str,
    n: int = 10,
) -> List[List[List[int]]]:
    """
    n-times divide the frames into training and test (no val inside each fold).

    Args:
        frames: list of frames (FrameInfo or similar)
        frames_json: path to JSON file where splits are stored (and read from
                     if present)
        patch_dir: directory where the JSON is stored (created if missing)
        n: number of folds

    Returns:
        splits: list of [train_index_list, test_index_list] for each fold
    """
    if os.path.isfile(frames_json):
        print("[SPLIT][CV] Reading n-splits from file")
        with open(frames_json, "r") as file:
            fjson = json.load(file)
            splits = fjson.get("splits", [])
            return splits

    print("[SPLIT][CV] Creating and writing n-splits to file")
    frames_list = list(range(len(frames)))
    kf = KFold(n_splits=n, shuffle=True, random_state=1117)
    print(f"[SPLIT][CV] Number of splitting iterations: {kf.get_n_splits(frames_list)}")

    splits: List[List[List[int]]] = []
    for train_index, test_index in kf.split(frames_list):
        splits.append([train_index.tolist(), test_index.tolist()])

    frame_split = {"splits": splits}
    _ensure_dir(patch_dir)
    with open(frames_json, "w") as f:
        json.dump(frame_split, f, indent=2)

    print(_col("[SPLIT][CV] Saved cross-validation splits.", _C.GREEN))
    return splits


# ---------- helpers for stratified splitting by positive-pixel rate ----------


def _pos_rate(frames: Sequence[Any]) -> np.ndarray:
    """
    Compute fraction of positive pixels per frame (used to stratify).

    Expects each frame to have `.annotations` (HxW) with positives > 0.
    """
    _pos_rate = []
    for fr in frames:
        label_annotation = getattr(fr, "annotations", None)
        if label_annotation is None:
            _pos_rate.append(0.0)
        else:
            lab = (np.asarray(label_annotation) > 0).astype(np.uint8)
            _pos_rate.append(float(lab.sum()) / float(lab.size + 1e-6))
    return np.asarray(_pos_rate, dtype=np.float32)

def _make_strata(pos_rates: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """
    Quantile-bin positive rates into strata labels for stratification.

    Falls back to a single stratum if all rates are identical.
    """
    if pos_rates.size == 0:
        return np.zeros(0, dtype=int)

    if np.allclose(pos_rates, pos_rates[0]):
        return np.zeros_like(pos_rates, dtype=int)

    qs = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(pos_rates, qs)
    # Ensure strictly increasing bins to avoid empty/duplicate bins
    bins[0] = -1e-12
    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1e-12
    strata = np.digitize(pos_rates, bins[1:-1], right=False)
    return strata.astype(int)


# ---------- main split (backward-compatible signature) ----------


def split_dataset(
    frames: Sequence[Any],
    frames_json: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    *,
    n_bins: int = 5,
    random_state: int = 1337,
    stratify_by_positives: bool = True,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Divide frames into training, validation, and test. If `frames_json` exists,
    read the split from it; otherwise create a new split (optionally stratified).

    Args:
        frames: list of frames (FrameInfo or similar)
        frames_json: path to JSON file for storing the split
        test_size: fraction of total frames to allocate to test
        val_size: fraction of total frames to allocate to validation
                  (of the overall dataset; we convert to a fraction of the
                  remaining train+val pool)
        n_bins: number of quantile bins for stratification by positive rate
        random_state: RNG seed
        stratify_by_positives: if True, stratify by per-frame positive rate;
                               else random splits

    Returns:
        (training_frames, validation_frames, testing_frames) as lists of indices
    """
    if os.path.isfile(frames_json):
        print("[SPLIT][DATA] Reading train/val/test split from file")
        with open(frames_json, "r") as file:
            fjson = json.load(file)
            training_frames = fjson["training_frames"]
            testing_frames = fjson["testing_frames"]
            validation_frames = fjson["validation_frames"]

        print(f"[SPLIT][DATA] training_frames={len(training_frames)}")
        print(f"[SPLIT][DATA] validation_frames={len(validation_frames)}")
        print(f"[SPLIT][DATA] testing_frames={len(testing_frames)}")
        return training_frames, validation_frames, testing_frames

    print(
        "[SPLIT][DATA] Creating and writing train/val/test split "
        f"({'stratified' if stratify_by_positives else 'random'}) to file"
    )

    idx = np.arange(len(frames))
    strat_labels = None

    if stratify_by_positives:
        pos_rates = _pos_rate(frames)
        try:
            strat_labels = _make_strata(pos_rates, n_bins=n_bins)
        except Exception:
            strat_labels = None

    # Split off test
    try:
        train_val_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            stratify=strat_labels,
        )
    except ValueError:
        # Fallback if strata too imbalanced
        train_val_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=random_state, stratify=None
        )

    # Split train vs val inside remaining pool
    # Convert val_size (of full set) to fraction inside train_val pool
    remaining = max(1e-12, 1.0 - float(test_size))
    val_share_in_tv = float(val_size) / remaining

    tv_strata = None
    if stratify_by_positives and strat_labels is not None:
        tv_strata = strat_labels[train_val_idx]
        # Reduce bins if tiny sample
        n_bins_tv = max(2, min(n_bins, len(train_val_idx) // 5))
        try:
            tv_strata = _make_strata(tv_strata.astype(np.float32), n_bins=n_bins_tv)
        except Exception:
            tv_strata = None

    try:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_share_in_tv,
            random_state=random_state + 1,
            stratify=tv_strata,
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_share_in_tv,
            random_state=random_state + 1,
            stratify=None,
        )

    training_frames = train_idx.tolist()
    validation_frames = val_idx.tolist()
    testing_frames = test_idx.tolist()

    frame_split: Dict[str, Any] = {
        "training_frames": training_frames,
        "testing_frames": testing_frames,
        "validation_frames": validation_frames,
    }
    _ensure_dir(os.path.dirname(frames_json))
    with open(frames_json, "w") as f:
        json.dump(frame_split, f, indent=2)

    print(f"[SPLIT][DATA] training_frames={len(training_frames)}")
    print(f"[SPLIT][DATA] validation_frames={len(validation_frames)}")
    print(f"[SPLIT][DATA] testing_frames={len(testing_frames)}")
    return training_frames, validation_frames, testing_frames
