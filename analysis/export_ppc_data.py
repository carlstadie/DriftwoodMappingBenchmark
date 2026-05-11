# export_ppc_data.py
"""
Export raw data from the group model's predictive checks to .npz files
so the figure script can build publication-quality plots independently.

Call export_ppc_data() once per metric inside the main.py metric loop,
immediately after fit_group_model_with_ranking() returns.

Add to main.py imports:
    from export_ppc_data import export_ppc_data

Add to main.py metric loop (after fit_group_model_with_ranking returns):
    export_ppc_data(
        metric=metric,
        prior_mu_draws=prior_mu_draws,
        y_obs_raw=y_obs_raw,
        obs_groups=obs_groups,
        groups=list(groups_this),
        post_pred=post_pred,
        inv_transform=inv,
        out_dir=assets_dir / "ppc_data",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def export_ppc_data(
    metric: str,
    prior_mu_draws: np.ndarray,
    y_obs_raw: np.ndarray,
    obs_groups: np.ndarray,
    groups: Sequence[str],
    post_pred,
    inv_transform: Callable,
    out_dir: Path,
) -> None:
    """
    Save predictive check data for one metric to a compressed numpy archive.

    Args:
        metric:           metric name (used as filename slug)
        prior_mu_draws:   (n_samples, n_groups) prior draws of group means,
                          already on original scale (from fit_group_model_with_ranking)
        y_obs_raw:        (n_obs,) observed run-level values on original scale
        obs_groups:       (n_obs,) group label for each observation
        groups:           ordered list of group names matching prior_mu_draws columns
        post_pred:        ArviZ InferenceData posterior predictive object
                          (from pm.sample_posterior_predictive)
        inv_transform:    callable that maps transformed scale -> original scale
        out_dir:          directory to write the .npz file
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Posterior predictive: stack chains × draws, back-transform           #
    # post_pred.posterior_predictive["y"] shape: (chains, draws, n_obs)   #
    # ------------------------------------------------------------------ #
    ppy_t = post_pred.posterior_predictive["y"].values  # (C, D, n_obs)
    n_chains, n_draws, n_obs = ppy_t.shape
    ppy_t_flat = ppy_t.reshape(-1, n_obs)               # (C*D, n_obs)

    # Apply inverse transform row-wise (each row is one posterior sample)
    # inv_transform is typically a vectorised numpy function, but guard for edge cases
    try:
        ppy_orig = inv_transform(ppy_t_flat)            # (C*D, n_obs)
    except Exception:
        ppy_orig = np.stack(
            [inv_transform(row) for row in ppy_t_flat], axis=0
        )

    slug = metric.replace("_", "-")
    out_path = out_dir / f"ppc_{slug}.npz"

    np.savez_compressed(
        out_path,
        # Prior predictive group means (original scale)
        prior_mu_draws=np.asarray(prior_mu_draws, dtype=np.float32),
        # Posterior predictive individual observations (original scale)
        post_pred_y=np.asarray(ppy_orig, dtype=np.float32),
        # Observed data
        y_obs=np.asarray(y_obs_raw, dtype=np.float32),
        obs_groups=np.array(obs_groups, dtype=object),
        groups=np.array(list(groups), dtype=object),
    )
    print(f"    [PPC export] {out_path.name}  "
          f"prior_mu_draws={prior_mu_draws.shape}  "
          f"post_pred_y={ppy_orig.shape}  "
          f"y_obs={y_obs_raw.shape}")