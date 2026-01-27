"""
Main report building logic with enhanced visualizations.
"""

from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

from config import (
    METRICS,
    BOUNDED_01_METRICS,
    MAXIMIZE_METRICS,
    GROUP_ORDER,
    DATASET_ORDER,
    ARCH_ORDER,
    BASELINE_ARCH,
    BASELINE_DATASET,
    ROPE_BY_METRIC_TRANSFORMED,
    METRIC_WEIGHTS,
    HDI_PROB,
    TEST,
    PLOT_STYLE,
    SD_KDE_BANDWIDTHS,
    get_output_paths,
)
from column_guides import *
from html_report import HtmlReport, add_table_with_column_guide, _safe_slug
from data_utils import (
    load_checkpoint_metrics,
    higher_is_better,
    generate_test_data,
)
from models import (
    fit_group_model_with_ranking,
    fit_factorial_model_with_bf,
    overall_ranking_across_metrics,
)
from contrasts import (
    contrasts_within_each_dataset,
    contrasts_within_each_architecture,
    contrasts_arch_overall,
    contrasts_dataset_overall,
)
from plotting import (
    plot_group_forest_boxplot,
    plot_pr_best,
    plot_factorial_heatmap,
    plot_factorial_effects_boxplot,
    plot_contrasts,
    plot_overall_pr_best,
    plot_posterior_predictive_check,
    plot_prior_predictive_check,
    plot_consecutive_bf,
    plot_distribution_kde,
)

# Apply plot styling
matplotlib.use("Agg")
matplotlib.rcParams.update(PLOT_STYLE)
sns.set_style("white")


def add_metric_analysis(
    report: HtmlReport,
    metric: str,
    data: pd.DataFrame,
    assets_dir: Path,
    mu_draws_by_metric: Dict,
    hib_by_metric: Dict,
    groups_by_metric: Dict,
) -> None:
    """
    Add complete analysis for a single metric to the report.
    
    Parameters
    ----------
    report : HtmlReport
        Report instance
    metric : str
        Metric name
    data : pd.DataFrame
        Data for this metric
    assets_dir : Path
        Assets directory
    mu_draws_by_metric : Dict
        Dictionary to store draws for overall ranking
    hib_by_metric : Dict
        Dictionary to store higher_is_better flags
    groups_by_metric : Dict
        Dictionary to store group names
    """
    tab_id = f"tab-{_safe_slug(metric)}"
    report.open_tab_content(tab_id)

    report.add_card_start()
    report.add_h2(f"Analysis: {metric}")
    if TEST:
        report.add_html('<span class="badge warning">TEST MODE - Synthetic Data</span>')
    report.add_card_end()

    if data.empty:
        report.add_card_start()
        report.add_html('<div class="alert alert-warning">')
        report.add_paragraph("No data available for this metric after filtering.")
        report.add_html('</div>')
        report.add_card_end()
        report.close_tab_content()
        return

    # Distribution plot
    kde_name = f"kde_{_safe_slug(metric)}.png"
    plot_distribution_kde(data, metric, assets_dir / kde_name)

    # Prior predictive check
    from data_utils import transform_y
    y_raw = data[metric].to_numpy()
    y_t, inv, tname = transform_y(y_raw, metric, BOUNDED_01_METRICS)
    
    prior_pred_name = f"prior_pred_{_safe_slug(metric)}.png"
    plot_prior_predictive_check(metric, tname, inv, assets_dir / prior_pred_name)

    report.add_card_start()
    report.add_h3("Data Distribution")
    report.add_paragraph(
        f"Empirical distribution of {metric} across all experimental runs. "
        "Different colors represent different architecture-dataset combinations."
    )
    report.add_image(f"{assets_dir.name}/{kde_name}", alt=f"Distribution for {metric}")
    report.add_card_end()

    # Fit hierarchical group model
    idata, perf_table, _, mu_draws, scale_name, groups_this, post_pred = (
        fit_group_model_with_ranking(
            data, 
            group_col="group", 
            metric_col=metric,
            bounded_metrics=BOUNDED_01_METRICS,
            maximize_metrics=MAXIMIZE_METRICS,
        )
    )

    # Store for overall ranking
    mu_draws_by_metric[metric] = mu_draws
    hib_by_metric[metric] = higher_is_better(metric, MAXIMIZE_METRICS)
    groups_by_metric[metric] = list(groups_this)

    # Posterior predictive check
    post_pred_name = f"post_pred_{_safe_slug(metric)}.png"
    plot_posterior_predictive_check(idata, post_pred, y_t, metric, assets_dir / post_pred_name)

    # Forest and Pr(best) plots - UPDATED TO USE BOXPLOT STYLE
    forest_name = f"group_forest_{_safe_slug(metric)}.png"
    plot_group_forest_boxplot(perf_table, metric, assets_dir / forest_name)

    prbest_name = f"prbest_{_safe_slug(metric)}.png"
    plot_pr_best(perf_table, metric, assets_dir / prbest_name)

    # Add to report
    report.add_card_start()
    report.add_h3("Hierarchical Model Results")
    report.add_paragraph(
        f"<strong>Modeling approach:</strong> Data transformed to {scale_name} scale for robust estimation. "
        "Results presented on original scale with hierarchical variance structure for improved regularization."
    )
    
    report.add_html('<div class="alert alert-info">')
    report.add_paragraph(
        f"<strong>Sample size:</strong> {len(data)} observations across {len(groups_this)} groups. "
        f"<strong>Transform:</strong> {scale_name} scale."
    )
    report.add_html('</div>')
    
    add_table_with_column_guide(report, perf_table, COLGUIDE_PERF)
    report.add_card_end()

    # Model validation
    report.add_card_start()
    report.add_h3("Model Validation")
    report.add_html('<div class="grid-2">')
    
    report.add_html('<div class="card" style="margin:0">')
    report.add_h3("Prior Predictive Check")
    report.add_image(
        f"{assets_dir.name}/{prior_pred_name}",
        alt=f"Prior predictive check for {metric}"
    )
    report.add_paragraph(
        "Shows whether prior distributions produce reasonable values on the original scale."
    )
    report.add_html("</div>")
    
    report.add_html('<div class="card" style="margin:0">')
    report.add_h3("Posterior Predictive Check")
    report.add_image(
        f"{assets_dir.name}/{post_pred_name}",
        alt=f"Posterior predictive check for {metric}"
    )
    report.add_paragraph(
        "Red distributions show model predictions; blue histogram shows observed data. "
        "Good fit indicates overlap between predicted and observed."
    )
    report.add_html("</div>")
    
    report.add_html("</div>")
    report.add_card_end()

    # Performance comparisons
    report.add_card_start()
    report.add_h3("Performance Comparisons")
    report.add_html('<div class="grid-2">')
    
    report.add_html('<div class="card" style="margin:0">')
    report.add_h3("Estimated Performance")
    report.add_image(
        f"{assets_dir.name}/{forest_name}", 
        alt=f"Performance comparison for {metric}"
    )
    report.add_paragraph(
        "Box width represents uncertainty (95% HDI). "
        "Vertical line shows point estimate. "
        "Groups sorted by expected rank."
    )
    report.add_html("</div>")
    
    report.add_html('<div class="card" style="margin:0">')
    report.add_h3("Ranking Probabilities")
    report.add_image(
        f"{assets_dir.name}/{prbest_name}", 
        alt=f"Ranking probabilities for {metric}"
    )
    report.add_paragraph(
        "Probability that each configuration is the best performer. "
        "Values close to 1.0 indicate strong evidence."
    )
    report.add_html("</div>")
    
    report.add_html("</div>")
    report.add_card_end()

    # Consecutive ranking BF analysis
    consecutive_bf_name = f"consecutive_bf_{_safe_slug(metric)}.png"
    bf_table = plot_consecutive_bf(
        mu_draws=mu_draws,
        perf_table=perf_table,
        group_names=groups_this,
        metric=metric,
        higher_better=higher_is_better(metric, MAXIMIZE_METRICS),
        out_path=assets_dir / consecutive_bf_name,
        sd_kde_bandwidths=SD_KDE_BANDWIDTHS,
    )

    report.add_card_start()
    report.add_h3("Pairwise Ranking Comparisons")
    report.add_paragraph(
        "Evidence for ranking order by comparing each configuration to the next-ranked configuration. "
        "Green bars (>0.95) indicate strong confidence in ranking. "
        "Red bars (<0.80) suggest uncertain rankings."
    )
    add_table_with_column_guide(report, bf_table, COLGUIDE_CONSECUTIVE_BF)
    report.add_image(
        f"{assets_dir.name}/{consecutive_bf_name}",
        alt=f"Consecutive ranking comparison for {metric}"
    )
    report.add_card_end()

    # Factorial model
    (
        idata_fact,
        factorial_report,
        pred_table,
        pred_draws_transformed,
        pred_draws_original,
        (datasets, archs),
        post_pred_fact,
    ) = fit_factorial_model_with_bf(
        data,
        dataset_col="dataset",
        arch_col="arch",
        metric_col=metric,
        bounded_metrics=BOUNDED_01_METRICS,
        sd_kde_bandwidths=SD_KDE_BANDWIDTHS,
    )

    # Generate factorial plots - UPDATED TO USE BOXPLOT STYLE
    heat_name = f"heat_{_safe_slug(metric)}.png"
    plot_factorial_heatmap(pred_table, metric, assets_dir / heat_name)

    eff_name = f"effects_{_safe_slug(metric)}.png"
    plot_factorial_effects_boxplot(factorial_report, metric, assets_dir / eff_name)

    # Compute all contrasts
    rope_transformed = ROPE_BY_METRIC_TRANSFORMED.get(metric, 0.05)
    
    contrast_df = contrasts_within_each_dataset(
        pred_draws_transformed, pred_draws_original,
        datasets, archs, BASELINE_ARCH,
        rope_transformed, HDI_PROB
    )
    
    dataset_contrast_df = contrasts_within_each_architecture(
        pred_draws_transformed, pred_draws_original,
        datasets, archs, BASELINE_DATASET,
        rope_transformed, HDI_PROB
    )
    
    arch_overall_df = contrasts_arch_overall(
        pred_draws_transformed, pred_draws_original,
        archs, BASELINE_ARCH,
        rope_transformed, HDI_PROB
    )
    
    dataset_overall_df = contrasts_dataset_overall(
        pred_draws_transformed, pred_draws_original,
        datasets, BASELINE_DATASET,
        rope_transformed, HDI_PROB
    )

    # Generate all contrast plots
    contrast_plot_name = f"contrasts_arch_{_safe_slug(metric)}.png"
    plot_contrasts(contrast_df, metric, assets_dir / contrast_plot_name,
                  title_suffix="Architecture within Dataset")

    dataset_contrast_plot_name = f"contrasts_dataset_{_safe_slug(metric)}.png"
    plot_contrasts(dataset_contrast_df, metric, assets_dir / dataset_contrast_plot_name,
                  title_suffix="Dataset within Architecture")

    arch_overall_plot_name = f"arch_overall_{_safe_slug(metric)}.png"
    plot_contrasts(arch_overall_df, metric, assets_dir / arch_overall_plot_name,
                  title_suffix="Overall Architecture Effects")

    dataset_overall_plot_name = f"dataset_overall_{_safe_slug(metric)}.png"
    plot_contrasts(dataset_overall_df, metric, assets_dir / dataset_overall_plot_name,
                  title_suffix="Overall Dataset Effects")

    # Add factorial results to report
    report.add_card_start()
    report.add_h3("Factorial Analysis")
    report.add_paragraph(
        "Decomposition of performance into main effects (dataset, architecture) "
        "and interaction effects. Helps identify whether certain dataset-architecture "
        "combinations perform better or worse than expected."
    )
    
    if factorial_report["scale"].iloc[0] == "log":
        eff_view = factorial_report[[
            "param", "Pr(>0)", "BF10_interpretation",
            "BF10_bw_scott", "BF10_bw_0.1", "BF10_bw_0.3", "BF10_bw_0.5", "BF10_bw_silverman",
            "Abs_Δ_mean", "Abs_Δ_lo", "Abs_Δ_hi", "%_change", "baseline"
        ]].copy()
        add_table_with_column_guide(report, eff_view, COLGUIDE_EFFECTS_LOG)
    else:
        eff_view = factorial_report[[
            "param", "Pr(>0)", "BF10_interpretation",
            "BF10_bw_scott", "BF10_bw_0.1", "BF10_bw_0.3", "BF10_bw_0.5", "BF10_bw_silverman",
            "OR", "Δp_mean", "Δp_lo", "Δp_hi", "baseline"
        ]].copy()
        add_table_with_column_guide(report, eff_view, COLGUIDE_EFFECTS_LOGIT)

    report.add_h3("Effect Sizes")
    report.add_image(f"{assets_dir.name}/{eff_name}", alt=f"Factorial effects for {metric}")
    report.add_paragraph(
        "Box width represents uncertainty in effect size. "
        "Effects crossing zero are uncertain. "
        "Probability values (p) indicate confidence in direction."
    )
    report.add_card_end()

    # Predicted performance
    report.add_card_start()
    report.add_h3("Predicted Performance by Configuration")
    
    pred_view = pred_table[["dataset", "arch", "mean_original", "hdi_lo_original", "hdi_hi_original"]].copy()
    pred_view = pred_view.rename(columns={
        "mean_original": "mean",
        "hdi_lo_original": "hdi_lo",
        "hdi_hi_original": "hdi_hi",
    })
    add_table_with_column_guide(report, pred_view, COLGUIDE_PRED)

    report.add_h3("Performance Heatmap")
    report.add_image(f"{assets_dir.name}/{heat_name}", alt=f"Heatmap for {metric}")
    report.add_paragraph(
        "Color intensity indicates performance level. "
        "Warmer colors (green) indicate better performance. "
        "Numbers show exact predicted values."
    )
    report.add_card_end()

    # Architecture contrasts within datasets
    report.add_card_start()
    report.add_h3(f"Architecture Comparisons (Baseline: {BASELINE_ARCH})")
    report.add_paragraph(
        f"How do different architectures compare within each dataset? "
        f"All comparisons are relative to {BASELINE_ARCH}."
    )
    
    contrast_view = contrast_df[[
        "dataset", "contrast", "mean_original", "hdi_lo_original", "hdi_hi_original", "Pr(>0)", "Pr(|Δ|<ROPE)"
    ]].copy().rename(columns={
        "mean_original": "Δ_mean",
        "hdi_lo_original": "Δ_lo",
        "hdi_hi_original": "Δ_hi",
    })
    add_table_with_column_guide(report, contrast_view, COLGUIDE_CONTRASTS_DATASET)
    
    report.add_image(f"{assets_dir.name}/{contrast_plot_name}", 
                    alt=f"Architecture contrasts for {metric}")
    report.add_card_end()

    # Dataset contrasts within architectures
    report.add_card_start()
    report.add_h3(f"Dataset Comparisons (Baseline: {BASELINE_DATASET})")
    report.add_paragraph(
        f"How do different datasets compare within each architecture? "
        f"All comparisons are relative to {BASELINE_DATASET}."
    )
    
    dataset_contrast_view = dataset_contrast_df[[
        "arch", "contrast", "mean_original", "hdi_lo_original", "hdi_hi_original", "Pr(>0)", "Pr(|Δ|<ROPE)"
    ]].copy().rename(columns={
        "mean_original": "Δ_mean",
        "hdi_lo_original": "Δ_lo",
        "hdi_hi_original": "Δ_hi",
    })
    add_table_with_column_guide(report, dataset_contrast_view, COLGUIDE_CONTRASTS_ARCH)
    
    report.add_image(f"{assets_dir.name}/{dataset_contrast_plot_name}", 
                    alt=f"Dataset contrasts for {metric}")
    report.add_card_end()

    # Overall marginal contrasts
    report.add_card_start()
    report.add_h3("Marginal Effects")
    report.add_paragraph(
        "Overall effects averaged across all conditions. "
        "Shows which architectures and datasets perform best on average."
    )
    
    report.add_html('<div class="grid-2">')
    
    report.add_html('<div>')
    report.add_h3(f"Architecture Effects (averaged over datasets)")
    arch_overall_view = arch_overall_df[[
        "scope", "contrast", "mean_original", "hdi_lo_original", "hdi_hi_original", "Pr(>0)", "Pr(|Δ|<ROPE)"
    ]].copy()
    add_table_with_column_guide(report, arch_overall_view, COLGUIDE_CONTRASTS_OVERALL)
    report.add_image(f"{assets_dir.name}/{arch_overall_plot_name}", 
                    alt=f"Overall architecture effects for {metric}")
    report.add_html('</div>')
    
    report.add_html('<div>')
    report.add_h3(f"Dataset Effects (averaged over architectures)")
    dataset_overall_view = dataset_overall_df[[
        "scope", "contrast", "mean_original", "hdi_lo_original", "hdi_hi_original", "Pr(>0)", "Pr(|Δ|<ROPE)"
    ]].copy()
    add_table_with_column_guide(report, dataset_overall_view, COLGUIDE_CONTRASTS_OVERALL)
    report.add_image(f"{assets_dir.name}/{dataset_overall_plot_name}", 
                    alt=f"Overall dataset effects for {metric}")
    report.add_html('</div>')
    
    report.add_html('</div>')
    report.add_card_end()

    # Add interpretation guide
    report.add_footnote(
        "Interpreting Results",
        """
<ul>
  <li><strong>Pr(>0):</strong> Probability that the effect is positive. Values >0.95 indicate strong evidence.</li>
  <li><strong>Pr(|Δ|<ROPE):</strong> Probability that the effect is negligibly small. High values suggest no practical difference.</li>
  <li><strong>BF10:</strong> Bayes factor comparing effect vs no-effect. >10 = strong evidence for effect, <0.1 = strong evidence for no effect.</li>
  <li><strong>HDI (Highest Density Interval):</strong> 95% credible interval for the parameter. Narrower intervals indicate more certainty.</li>
  <li><strong>Box visualization:</strong> Width represents uncertainty. Boxes not crossing zero indicate clear directional effects.</li>
</ul>
"""
    )

    report.close_tab_content()