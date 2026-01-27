"""
Plotting utilities for Bayesian benchmark analysis with enhanced styling.
"""

from pathlib import Path
from typing import Callable, Optional, Sequence
import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from config import PLOT_STYLE, MODEL_COLORS, DATASET_COLORS


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    """
    Save figure to file and close.
    
    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure
    out_path : Path
        Output path
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def apply_reference_styling(ax: plt.Axes) -> None:
    """
    Apply consistent styling to axes matching reference figures.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object
    """
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('black')
    ax.grid(True, axis="x", alpha=0.25, linestyle="--", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=6.5, pad=2)
    ax.tick_params(axis='y', labelsize=7, pad=2)


def format_parameter_label(param: str) -> str:
    """
    Convert parameter names to readable labels with subscripts.
    
    Examples:
        beta_dataset[AE] -> Dataset effect: AE
        beta_arch[U-Net] -> Architecture effect: U-Net
        beta_interaction[AE×U-Net] -> Interaction: AE × U-Net
    
    Parameters
    ----------
    param : str
        Parameter name from model
    
    Returns
    -------
    str
        Formatted label
    """
    # Remove beta_ prefix
    param = param.replace("beta_", "")
    
    # Handle dataset effects
    if param.startswith("dataset["):
        match = re.search(r'dataset\[(.+?)\]', param)
        if match:
            return f"Dataset: {match.group(1)}"
    
    # Handle architecture effects
    if param.startswith("arch["):
        match = re.search(r'arch\[(.+?)\]', param)
        if match:
            return f"Architecture: {match.group(1)}"
    
    # Handle interactions
    if param.startswith("interaction["):
        match = re.search(r'interaction\[(.+?)\]', param)
        if match:
            parts = match.group(1).split("×")
            if len(parts) == 2:
                return f"Interaction: {parts[0]} × {parts[1]}"
            return f"Interaction: {match.group(1)}"
    
    return param


def get_color_for_group(group_name: str) -> str:
    """
    Get color for a group based on architecture or dataset.
    
    Parameters
    ----------
    group_name : str
        Group name
    
    Returns
    -------
    str
        Hex color code
    """
    # Check for architecture
    for arch, color in MODEL_COLORS.items():
        if arch in group_name:
            return color
    
    # Check for dataset
    for dataset, color in DATASET_COLORS.items():
        if dataset in group_name:
            return color
    
    return "#999999"  # Default gray


def plot_group_forest_boxplot(
    perf_table: pd.DataFrame,
    metric: str,
    out_path: Path,
    hdi_prob: float = 0.95,
) -> None:
    """
    Enhanced forest plot with boxplot-style visualization.
    
    Parameters
    ----------
    perf_table : pd.DataFrame
        Performance table with mean and HDI columns
    metric : str
        Metric name
    out_path : Path
        Output file path
    hdi_prob : float
        HDI probability
    """
    df = perf_table.copy()

    hdi_cols = [c for c in df.columns if c.startswith("hdi_") and c.endswith("%")]
    if len(hdi_cols) >= 2:
        hdi_lo = sorted(
            hdi_cols, key=lambda c: float(c.replace("hdi_", "").replace("%", ""))
        )[0]
        hdi_hi = sorted(
            hdi_cols, key=lambda c: float(c.replace("hdi_", "").replace("%", ""))
        )[-1]
    else:
        hdi_lo = "hdi_3%"
        hdi_hi = "hdi_97%"

    df = df.sort_values("E[rank]").reset_index(drop=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.5 * len(df))))
    
    y_positions = np.arange(len(df))
    
    # Get colors for each group
    colors = [get_color_for_group(group) for group in df["group"]]
    
    # Draw boxes (IQR-like visualization using HDI)
    for i, (idx, row) in enumerate(df.iterrows()):
        y = y_positions[i]
        mean_val = row["mean"]
        lo_val = row[hdi_lo]
        hi_val = row[hdi_hi]
        
        # Box width (representing uncertainty)
        box_height = 0.6
        
        # Draw box from lo to hi
        box = mpatches.Rectangle(
            (lo_val, y - box_height/2),
            hi_val - lo_val,
            box_height,
            facecolor=colors[i],
            edgecolor='black',
            linewidth=0.8,
            alpha=0.75,
            zorder=2
        )
        ax.add_patch(box)
        
        # Draw median line
        ax.plot([mean_val, mean_val], 
                [y - box_height/2, y + box_height/2],
                color='black', linewidth=1.5, zorder=3)
        
        # Add value label
        ax.text(mean_val, y, f' {mean_val:.3f}', 
               va='center', ha='left', fontsize=7,
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', 
                        edgecolor='none', 
                        alpha=0.8))
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["group"].tolist(), fontsize=7)
    ax.set_xlabel(f"{metric} (Original Scale)", fontsize=8, fontweight='bold')
    ax.set_title(f"Performance Comparison: {metric}", 
                fontsize=10, fontweight='bold', pad=12)
    
    # Create legend
    unique_models = []
    unique_colors = []
    for group in df["group"]:
        for arch in MODEL_COLORS.keys():
            if arch in group and arch not in unique_models:
                unique_models.append(arch)
                unique_colors.append(MODEL_COLORS[arch])
    
    if unique_models:
        legend_elements = [
            mpatches.Patch(facecolor=color, edgecolor='black', 
                          label=model, alpha=0.75, linewidth=0.8)
            for model, color in zip(unique_models, unique_colors)
        ]
        ax.legend(handles=legend_elements, loc='best', 
                 frameon=True, fontsize=7, 
                 title='Architecture', title_fontsize=8)
    
    apply_reference_styling(ax)
    ax.invert_yaxis()

    _save_fig(fig, out_path)


def plot_pr_best(
    perf_table: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    """
    Enhanced Pr(best) plot with color-coded bars.
    
    Parameters
    ----------
    perf_table : pd.DataFrame
        Performance table with Pr(best) column
    metric : str
        Metric name
    out_path : Path
        Output file path
    """
    df = perf_table.copy().sort_values("Pr(best)", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.5 * len(df))))
    
    # Get colors for each group
    colors = [get_color_for_group(group) for group in df["group"]]
    
    bars = ax.barh(df["group"], df["Pr(best)"], 
                   color=colors, alpha=0.75, 
                   edgecolor='black', linewidth=0.8)
    
    # Add value labels
    for i, (group, prob) in enumerate(zip(df["group"], df["Pr(best)"])):
        ax.text(prob + 0.02, i, f"{prob:.3f}", 
               va='center', fontsize=7,
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', 
                        edgecolor='none', 
                        alpha=0.8))
    
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Probability of Being Best", fontsize=8, fontweight='bold')
    ax.set_title(f"Ranking Probabilities: {metric}", 
                fontsize=10, fontweight='bold', pad=12)
    
    # Create legend
    unique_models = []
    unique_colors = []
    for group in df["group"]:
        for arch in MODEL_COLORS.keys():
            if arch in group and arch not in unique_models:
                unique_models.append(arch)
                unique_colors.append(MODEL_COLORS[arch])
    
    if unique_models:
        legend_elements = [
            mpatches.Patch(facecolor=color, edgecolor='black', 
                          label=model, alpha=0.75, linewidth=0.8)
            for model, color in zip(unique_models, unique_colors)
        ]
        ax.legend(handles=legend_elements, loc='best', 
                 frameon=True, fontsize=7,
                 title='Architecture', title_fontsize=8)
    
    apply_reference_styling(ax)

    _save_fig(fig, out_path)


def plot_factorial_heatmap(
    pred_table: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    """
    Enhanced heatmap with explicit cell values.
    
    Parameters
    ----------
    pred_table : pd.DataFrame
        Prediction table with dataset, arch, mean_original columns
    metric : str
        Metric name
    out_path : Path
        Output file path
    """
    pivot = pred_table.pivot(index="dataset", columns="arch", values="mean_original")

    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Use a better colormap
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", 
                  vmin=pivot.values.min()*0.95, vmax=pivot.values.max()*1.05)
    
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), fontsize=8)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=8)
    ax.set_title(f"Predicted Performance: {metric}", 
                fontsize=10, fontweight='bold', pad=12)
    ax.set_xlabel("Architecture", fontsize=8, fontweight='bold')
    ax.set_ylabel("Dataset", fontsize=8, fontweight='bold')

    # Add values to cells with better contrast
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            # Choose text color based on background
            text_color = 'white' if val < pivot.values.mean() else 'black'
            ax.text(j, i, f"{val:.4f}",
                   ha="center", va="center", 
                   color=text_color, fontsize=9, fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric, fontsize=8, fontweight='bold')
    cbar.ax.tick_params(labelsize=7)
    
    ax.tick_params(axis='x', labelsize=7, pad=2)
    ax.tick_params(axis='y', labelsize=7, pad=2)

    _save_fig(fig, out_path)


def plot_factorial_effects_boxplot(
    effects_report: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    """
    Enhanced factorial effects plot with boxplot-style visualization and formatted labels.
    
    Parameters
    ----------
    effects_report : pd.DataFrame
        Effects report with parameters and effect sizes
    metric : str
        Metric name
    out_path : Path
        Output file path
    """
    df = effects_report.copy()
    
    # Filter out zero or near-zero effects
    if df["scale"].iloc[0] == "log":
        df = df[np.abs(df["Abs_Δ_mean"]) > 1e-6].copy()
    else:
        df = df[np.abs(df["Δp_mean"]) > 1e-6].copy()
    
    if len(df) == 0:
        print(f"Warning: No non-zero effects for {metric}")
        # Create empty plot with message
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No significant effects detected", 
               ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        _save_fig(fig, out_path)
        return
    
    bf_col = "BF10_bw_scott" if "BF10_bw_scott" in df.columns else [c for c in df.columns if c.startswith("BF10_")][0]
    df = df.sort_values(bf_col, ascending=False).reset_index(drop=True)

    # Get effect sizes and intervals
    if df["scale"].iloc[0] == "log":
        x = df["Abs_Δ_mean"].to_numpy()
        lo = df["Abs_Δ_lo"].to_numpy()
        hi = df["Abs_Δ_hi"].to_numpy()
        xlabel = f"Effect Size (Absolute Change in {metric})"
    else:
        x = df["Δp_mean"].to_numpy()
        lo = df["Δp_lo"].to_numpy()
        hi = df["Δp_hi"].to_numpy()
        xlabel = f"Effect Size (Change in {metric})"

    # Format labels
    labels = [format_parameter_label(p) for p in df["param"]]

    y_positions = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(df))))
    
    # Determine colors based on parameter type
    colors = []
    for param in df["param"]:
        if "dataset" in param.lower():
            colors.append(DATASET_COLORS.get("AE", "#7986CB"))
        elif "arch" in param.lower():
            colors.append(MODEL_COLORS.get("U-Net", "#c27ba0"))
        else:  # interaction
            colors.append("#9575CD")  # Purple for interactions
    
    # Draw boxes
    for i, (xi, loi, hii) in enumerate(zip(x, lo, hi)):
        y = y_positions[i]
        box_height = 0.6
        
        # Box from lo to hi
        box = mpatches.Rectangle(
            (loi, y - box_height/2),
            hii - loi,
            box_height,
            facecolor=colors[i],
            edgecolor='black',
            linewidth=0.8,
            alpha=0.75,
            zorder=2
        )
        ax.add_patch(box)
        
        # Median line
        ax.plot([xi, xi], 
                [y - box_height/2, y + box_height/2],
                color='black', linewidth=1.5, zorder=3)
        
        # Add value label with Pr(>0)
        pr_pos = df.iloc[i]["Pr(>0)"]
        label_text = f'{xi:.3f}\n(p={pr_pos:.2f})'
        ax.text(xi, y, label_text, 
               va='center', ha='left' if xi > 0 else 'right', 
               fontsize=6,
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', 
                        edgecolor='none', 
                        alpha=0.8))
    
    # Zero line
    ax.axvline(0, linestyle='--', linewidth=1, color='black', alpha=0.5, zorder=1)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=8, fontweight='bold')
    ax.set_title(f"Factorial Effects: {metric}", 
                fontsize=10, fontweight='bold', pad=12)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=DATASET_COLORS.get("AE", "#7986CB"), 
                      edgecolor='black', label='Dataset Effect', 
                      alpha=0.75, linewidth=0.8),
        mpatches.Patch(facecolor=MODEL_COLORS.get("U-Net", "#c27ba0"), 
                      edgecolor='black', label='Architecture Effect', 
                      alpha=0.75, linewidth=0.8),
        mpatches.Patch(facecolor='#9575CD', 
                      edgecolor='black', label='Interaction Effect', 
                      alpha=0.75, linewidth=0.8),
    ]
    ax.legend(handles=legend_elements, loc='best', 
             frameon=True, fontsize=7,
             title='Effect Type', title_fontsize=8)
    
    apply_reference_styling(ax)

    _save_fig(fig, out_path)


def plot_contrasts(
    contrast_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title_suffix: str = "",
) -> None:
    """
    Enhanced contrast plot with boxplot-style visualization.
    
    Parameters
    ----------
    contrast_df : pd.DataFrame
        Contrast results
    metric : str
        Metric name
    out_path : Path
        Output file path
    title_suffix : str
        Additional title text
    """
    if contrast_df.empty:
        return

    df = contrast_df.copy()
    
    # Sort for clarity
    if "dataset" in df.columns:
        df = df.sort_values(["dataset", "mean_original"], ascending=[True, False]).reset_index(drop=True)
        labels = [f"{r.dataset}: {r.contrast}" for r in df.itertuples(index=False)]
        colors = [DATASET_COLORS.get(r.dataset, "#999999") for r in df.itertuples(index=False)]
    elif "arch" in df.columns:
        df = df.sort_values(["arch", "mean_original"], ascending=[True, False]).reset_index(drop=True)
        labels = [f"{r.arch}: {r.contrast}" for r in df.itertuples(index=False)]
        colors = [MODEL_COLORS.get(r.arch.split()[0], "#999999") for r in df.itertuples(index=False)]
    else:
        labels = [f"{r.contrast}" for r in df.itertuples(index=False)]
        colors = ["#999999"] * len(df)

    y_positions = np.arange(len(df))
    x = df["mean_original"].to_numpy()
    lo = df["hdi_lo_original"].to_numpy()
    hi = df["hdi_hi_original"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.4 * len(df))))
    
    # Draw boxes
    for i, (xi, loi, hii) in enumerate(zip(x, lo, hi)):
        y = y_positions[i]
        box_height = 0.6
        
        box = mpatches.Rectangle(
            (loi, y - box_height/2),
            hii - loi,
            box_height,
            facecolor=colors[i],
            edgecolor='black',
            linewidth=0.8,
            alpha=0.75,
            zorder=2
        )
        ax.add_patch(box)
        
        # Median line
        ax.plot([xi, xi], 
                [y - box_height/2, y + box_height/2],
                color='black', linewidth=1.5, zorder=3)
        
        # Add value label
        pr_pos = df.iloc[i]["Pr(>0)"]
        label_text = f'{xi:.3f}\n(p={pr_pos:.2f})'
        ax.text(xi, y, label_text, 
               va='center', ha='left' if xi > 0 else 'right', 
               fontsize=6,
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', 
                        edgecolor='none', 
                        alpha=0.8))
    
    # Zero line
    ax.axvline(0, linestyle='--', linewidth=1, color='black', alpha=0.5, zorder=1)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"Contrast (Change in {metric})", fontsize=8, fontweight='bold')
    
    title = f"Contrasts: {metric}"
    if title_suffix:
        title += f" - {title_suffix}"
    ax.set_title(title, fontsize=10, fontweight='bold', pad=12)
    
    apply_reference_styling(ax)

    _save_fig(fig, out_path)


def plot_overall_pr_best(
    overall_rank_table: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Enhanced overall Pr(best) plot.
    
    Parameters
    ----------
    overall_rank_table : pd.DataFrame
        Overall ranking table
    out_path : Path
        Output file path
    """
    df = overall_rank_table.copy().sort_values("Pr(best)", ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.5 * len(df))))
    
    colors = [get_color_for_group(group) for group in df["group"]]
    
    bars = ax.barh(df["group"], df["Pr(best)"], 
                   color=colors, alpha=0.75,
                   edgecolor='black', linewidth=0.8)
    
    # Add value labels
    for i, (group, prob) in enumerate(zip(df["group"], df["Pr(best)"])):
        ax.text(prob + 0.02, i, f"{prob:.3f}", 
               va='center', fontsize=7,
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', 
                        edgecolor='none', 
                        alpha=0.8))
    
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Probability of Being Best Overall", fontsize=8, fontweight='bold')
    ax.set_title("Overall Multi-Metric Ranking", fontsize=10, fontweight='bold', pad=12)
    
    # Legend
    unique_models = []
    unique_colors = []
    for group in df["group"]:
        for arch in MODEL_COLORS.keys():
            if arch in group and arch not in unique_models:
                unique_models.append(arch)
                unique_colors.append(MODEL_COLORS[arch])
    
    if unique_models:
        legend_elements = [
            mpatches.Patch(facecolor=color, edgecolor='black', 
                          label=model, alpha=0.75, linewidth=0.8)
            for model, color in zip(unique_models, unique_colors)
        ]
        ax.legend(handles=legend_elements, loc='best', 
                 frameon=True, fontsize=7,
                 title='Architecture', title_fontsize=8)
    
    apply_reference_styling(ax)
    
    _save_fig(fig, out_path)


def plot_posterior_predictive_check(
    idata,
    post_pred,
    y_obs: np.ndarray,
    metric: str,
    out_path: Path,
) -> None:
    """
    Posterior predictive check plot.
    
    Parameters
    ----------
    idata : az.InferenceData
        Inference data
    post_pred : az.InferenceData
        Posterior predictive samples
    y_obs : np.ndarray
        Observed data
    metric : str
        Metric name
    out_path : Path
        Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    y_pred = post_pred.posterior_predictive["y"].values

    ax.hist(y_obs, bins=30, alpha=0.6, label="Observed", density=True, 
           color='#667eea', edgecolor='black', linewidth=0.8)

    for i in range(min(100, y_pred.shape[1])):
        ax.hist(y_pred[0, i, :], bins=30, alpha=0.01, density=True, color='#f56565')

    ax.set_xlabel(f"{metric} (Transformed Scale)", fontsize=8, fontweight='bold')
    ax.set_ylabel("Density", fontsize=8, fontweight='bold')
    ax.set_title(f"Posterior Predictive Check: {metric}", fontsize=10, fontweight='bold', pad=12)
    ax.legend(fontsize=8, frameon=True)
    
    apply_reference_styling(ax)

    _save_fig(fig, out_path)


def plot_prior_predictive_check(
    metric: str,
    scale: str,
    inv_transform: Callable,
    out_path: Path,
    prior_sd_logit: float = 0.5,
    prior_sd_log: float = 0.2,
    n_samples: int = 1000,
) -> None:
    """
    Prior predictive check plot.
    
    Parameters
    ----------
    metric : str
        Metric name
    scale : str
        Scale name ('logit' or 'log')
    inv_transform : Callable
        Inverse transform function
    out_path : Path
        Output file path
    prior_sd_logit : float
        Prior SD for logit scale
    prior_sd_log : float
        Prior SD for log scale
    n_samples : int
        Number of samples
    """
    effect_sd = prior_sd_logit if scale == "logit" else prior_sd_log

    intercept_prior = np.random.normal(0, 1.5, n_samples)
    effect_prior = np.random.normal(0, effect_sd, n_samples)

    baseline = inv_transform(intercept_prior)
    with_effect = inv_transform(intercept_prior + effect_prior)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(baseline, bins=50, alpha=0.7, density=True, 
            color='#667eea', edgecolor='black', linewidth=0.8)
    ax1.set_xlabel(f"{metric} (Original Scale)", fontsize=8, fontweight='bold')
    ax1.set_ylabel("Density", fontsize=8, fontweight='bold')
    ax1.set_title("Prior: Baseline Performance", fontsize=10, fontweight='bold', pad=12)
    
    apply_reference_styling(ax1)

    ax2.hist(with_effect - baseline, bins=50, alpha=0.7, density=True, 
            color='#764ba2', edgecolor='black', linewidth=0.8)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel(f"Effect Size on {metric}", fontsize=8, fontweight='bold')
    ax2.set_ylabel("Density", fontsize=8, fontweight='bold')
    ax2.set_title("Prior: Effect Sizes", fontsize=10, fontweight='bold', pad=12)
    
    apply_reference_styling(ax2)

    _save_fig(fig, out_path)


def plot_consecutive_bf(
    mu_draws: np.ndarray,
    perf_table: pd.DataFrame,
    group_names: Sequence[str],
    metric: str,
    higher_better: bool,
    out_path: Path,
    sd_kde_bandwidths: list,
) -> pd.DataFrame:
    """
    Plot consecutive ranking Bayes factors.
    
    Parameters
    ----------
    mu_draws : np.ndarray
        Posterior draws, shape (samples, groups)
    perf_table : pd.DataFrame
        Performance table with rankings
    group_names : Sequence[str]
        Group names
    metric : str
        Metric name
    higher_better : bool
        Whether higher is better
    out_path : Path
        Output file path
    sd_kde_bandwidths : list
        KDE bandwidths for BF
    
    Returns
    -------
    pd.DataFrame
        BF results table
    """
    from statistical_utils import _bf10_from_prior_posterior, interpret_bf10
    
    sorted_ranks = perf_table.sort_values("E[rank]").reset_index(drop=True)
    
    bf_results = []
    for bw in sd_kde_bandwidths:
        rows = []
        for i in range(len(sorted_ranks) - 1):
            group_i = sorted_ranks.loc[i, "group"]
            group_j = sorted_ranks.loc[i + 1, "group"]
            rank_i = i + 1
            
            idx_i = list(group_names).index(group_i)
            idx_j = list(group_names).index(group_j)
            
            if higher_better:
                delta = mu_draws[:, idx_i] - mu_draws[:, idx_j]
            else:
                delta = mu_draws[:, idx_j] - mu_draws[:, idx_i]
            
            prior_delta = np.random.normal(0, mu_draws.std(), len(delta))
            bf10 = _bf10_from_prior_posterior(prior_delta, delta, bw=bw)
            pr_better = float((delta > 0).mean())
            
            rows.append({
                "rank": rank_i,
                "group": group_i,
                "next_rank_group": group_j,
                "Pr(better)": pr_better,
                "BF10_better": bf10,
            })
        
        bf_df = pd.DataFrame(rows)
        bw_str = bw if isinstance(bw, str) else f"{bw:.1f}"
        bf_df[f"BF10_bw_{bw_str}"] = bf_df["BF10_better"]
        bf_results.append(bf_df)
    
    main_df = bf_results[0][["rank", "group", "next_rank_group", "Pr(better)"]].copy()
    for bf_df in bf_results:
        bw_col = [c for c in bf_df.columns if c.startswith("BF10_bw_")][0]
        main_df[bw_col] = bf_df[bw_col]
    
    main_df["BF10_interpretation"] = main_df["BF10_bw_scott"].apply(interpret_bf10)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(main_df))))
    
    y = np.arange(len(main_df))
    x = main_df["Pr(better)"].to_numpy()
    
    # Color based on probability
    colors = []
    for p in x:
        if p > 0.95:
            colors.append('#48bb78')  # green
        elif p > 0.80:
            colors.append('#ed8936')  # orange
        else:
            colors.append('#f56565')  # red
    
    ax.barh(y, x, color=colors, alpha=0.75, edgecolor='black', linewidth=0.8)
    ax.axvline(0.5, linestyle="--", color="gray", linewidth=1, alpha=0.5)
    ax.axvline(0.95, linestyle="--", color="#48bb78", linewidth=1, alpha=0.5, label="95% threshold")
    
    # Add value labels
    for i, (yi, xi) in enumerate(zip(y, x)):
        ax.text(xi + 0.02, yi, f"{xi:.2f}", va='center', fontsize=7)
    
    labels = [f"{r.group} vs {r.next_rank_group}" for r in main_df.itertuples()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Probability of Being Better", fontsize=8, fontweight='bold')
    ax.set_title(f"Consecutive Rankings: {metric}", fontsize=10, fontweight='bold', pad=12)
    ax.legend(fontsize=7, frameon=True)
    
    apply_reference_styling(ax)
    
    _save_fig(fig, out_path)
    
    return main_df


def plot_distribution_kde(
    data: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    """
    KDE plot showing distribution of metric across groups.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data with 'group' and metric columns
    metric : str
        Metric name
    out_path : Path
        Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Get unique groups and assign colors
    groups = data["group"].unique()
    colors = [get_color_for_group(g) for g in groups]
    
    for group, color in zip(groups, colors):
        group_data = data[data["group"] == group][metric].dropna()
        if len(group_data) > 0:
            ax.hist(group_data, bins=20, alpha=0.5, label=group, 
                   color=color, edgecolor='black', linewidth=0.5, density=True)
    
    ax.set_xlabel(metric, fontsize=8, fontweight='bold')
    ax.set_ylabel("Density", fontsize=8, fontweight='bold')
    ax.set_title(f"Distribution: {metric}", fontsize=10, fontweight='bold', pad=12)
    ax.legend(fontsize=7, frameon=True, ncol=2)
    
    apply_reference_styling(ax)
    
    _save_fig(fig, out_path)