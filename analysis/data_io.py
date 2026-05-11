# data_io.py
from __future__ import annotations

from typing import Sequence, Tuple
import os
import pandas as pd


def load_checkpoint_metrics(csv_specs, metrics_):
    dfs = []
    for path, group, dataset, arch in csv_specs:
        if os.path.exists(path) is False:
            print(f"Warning: path {path} does not exist, skipping.")
            continue

        df = pd.read_csv(path)

        # Optional sanity checks (won't break anything)
        if "modality" in df.columns:
            mods = set(df["modality"].astype(str).unique())
            if len(mods) == 1 and dataset not in mods:
                print(f"Warning: {path} modality={list(mods)[0]} but CSV_SPECS dataset={dataset}")
        if "model" in df.columns:
            mods = set(df["model"].astype(str).unique())
            # you may see UNET/SWIN/TERRAMIND here; that's okay

        needed = [m for m in metrics_ if m in df.columns]
        if not needed:
            print(f"Warning: none of {metrics_} found in {path}, skipping.")
            continue

        # If this file is frame-level (has frame_id), aggregate to checkpoint.
        # If it's already dataset-level (no frame_id), this will be skipped.
        if "frame_id" in df.columns and "checkpoint" in df.columns:
            df = df.groupby("checkpoint", as_index=False)[needed].mean(numeric_only=True)

        subset = df[needed].copy()
        subset["group"] = group
        subset["dataset"] = dataset
        subset["arch"] = arch
        dfs.append(subset)

    return pd.concat(dfs, ignore_index=True)
