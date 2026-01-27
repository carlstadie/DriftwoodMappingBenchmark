"""
Main script for running Bayesian benchmark analysis.

Usage:
    python main.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

from config import (
    CSV_SPECS,
    METRICS,
    BOUNDED_01_METRICS,
    MAXIMIZE_METRICS,
    GROUP_ORDER,
    DATASET_ORDER,
    ARCH_ORDER,
    BASELINE_ARCH,
    BASELINE_DATASET,
    METRIC_WEIGHTS,
    TEST,
    get_output_paths,
    HDI_PROB,
    SD_KDE_BANDWIDTHS,
)
from column_guides import *
from html_report import HtmlReport, add_table_with_column_guide, _safe_slug
from data_utils import load_checkpoint_metrics, generate_test_data
from models import (
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
    plot_factorial_heatmap,
    plot_factorial_effects,
    plot_contrasts,
    plot_overall_pr_best,
)
from report_builder import add_metric_analysis


def add_overview_tab(report: HtmlReport, all_data: pd.DataFrame) -> None:
    """
    Add overview tab with data summary and guidelines.
    
    Parameters
    ----------
    report : HtmlReport
        Report instance
    all_data : pd.DataFrame
        Combined data
    """
    report.open_tab_content("tab-overview")
    
    report.add_card_start()
    report.add_h2("Data summary")
    counts = (
        all_data.groupby(["dataset", "arch", "group"], observed=True)
        .size()
        .reset_index(name="n_rows")
        .sort_values(["dataset", "arch", "group"])
        .reset_index(drop=True)
    )
    report.add_h3("Rows per dataset / arch / group")
    add_table_with_column_guide(report, counts, COLGUIDE_COUNTS)
    report.add_card_end()

    report.add_card_start()
    report.add_h2("Statistical Enhancements")
    report.add_paragraph(
        "<strong>1. Complete Factorial Analysis for Combined Posteriors:</strong> The Overall tab now includes "
        "full factorial modeling, contrasts, and interaction analysis—not just ranking."
    )
    report.add_paragraph(
        "<strong>2. Comprehensive Plot Coverage:</strong> Every contrast table now has a corresponding visualization. "
        "No missing plots for dataset-within-architecture or architecture-within-dataset comparisons."
    )
    report.add_paragraph(
        "<strong>3. Explicit Quantification:</strong> All plots now display actual numerical values, "
        "probabilities, and sample sizes directly on the visualizations."
    )
    report.add_paragraph(
        "<strong>4. Publication-Ready Styling:</strong> Updated visual aesthetic matching reference code "
        "with consistent colors, fonts, grid styling, and proper axis formatting."
    )
    report.add_card_end()

    report.add_card_start()
    report.add_h2("Interpretation Guidelines")
    report.add_footnote(
        "Bayes Factors (BF10)",
        """
<ul>
  <li>BF10 &gt; 1 supports H1 (effect exists); BF10 &lt; 1 supports H0 (no effect).</li>
  <li>Thresholds: 1–3 anecdotal, 3–10 moderate, 10–30 strong, 30–100 very strong, &gt;100 extreme.</li>
  <li><strong>Sensitivity:</strong> Multiple bandwidths show robustness. Consistent BF across bandwidths = stronger evidence.</li>
  <li>BF is NOT a p-value; it's an evidence ratio comparing hypotheses.</li>
</ul>
""",
    )
    report.add_footnote(
        "ROPE (Region of Practical Equivalence)",
        """
<ul>
  <li><strong>ROPE is on transformed scale</strong> for consistency with model parameter space.</li>
  <li><b>Pr(|Δ|&lt;ROPE)</b> = probability the effect is practically negligible.</li>
  <li>High Pr(>0) with low Pr(|Δ|&lt;ROPE) → meaningful improvement.</li>
  <li>High Pr(|Δ|&lt;ROPE) → differences are negligible.</li>
</ul>
""",
    )
    report.add_footnote(
        "Interpretable Effect Sizes",
        """
<ul>
  <li><strong>For log-scale metrics:</strong> 
    <ul>
      <li><b>% Change:</b> derived from exp(Δ_log) - 1</li>
      <li><b>Absolute Change:</b> computed via contrasts on predicted means (original scale)</li>
    </ul>
  </li>
  <li><strong>For logit-scale metrics:</strong>
    <ul>
      <li><b>Odds Ratio:</b> exp(Δ_logit)</li>
      <li><b>Δp:</b> difference in predicted metric (original scale) via contrasts</li>
    </ul>
  </li>
</ul>
""",
    )
    report.add_card_end()
    
    report.close_tab_content()


def add_overall_tab(
    report: HtmlReport,
    mu_draws_by_metric: dict,
    hib_by_metric: dict,
    groups_by_metric: dict,
    assets_dir: Path,
) -> None:
    """
    Add overall tab with multi-metric ranking and factorial analysis.
    
    Parameters
    ----------
    report : HtmlReport
        Report instance
    mu_draws_by_metric : dict
        Posterior draws by metric
    hib_by_metric : dict
        Higher-is-better flags by metric
    groups_by_metric : dict
        Group names by metric
    assets_dir : Path
        Assets directory
    """
    report.open_tab_content("tab-overall")

    if not mu_draws_by_metric:
        report.add_card_start()
        report.add_h2("Overall multi-metric ranking")
        report.add_paragraph("No metrics available for overall analysis.")
        report.add_card_end()
        report.close_tab_content()
        return

    # Find common groups
    common_groups = None
    for _, g_list in groups_by_metric.items():
        g_set = set(g_list)
        common_groups = g_set if common_groups is None else (common_groups & g_set)

    if not common_groups:
        report.add_card_start()
        report.add_h2("Overall multi-metric ranking")
        report.add_paragraph(
            "Could not compute an overall ranking because there are no common groups "
            "across all metrics."
        )
        report.add_card_end()
        report.close_tab_content()
        return

    group_names = [g for g in GROUP_ORDER if g in common_groups]

    # Align draws
    aligned_draws = {}
    for m in METRICS:
        if m not in mu_draws_by_metric:
            continue
        cols = list(groups_by_metric[m])
        idx = [cols.index(g) for g in group_names]
        aligned_draws[m] = mu_draws_by_metric[m][:, idx]

    # Compute overall ranking
    composite_draws, overall_rank_table, used_weights = overall_ranking_across_metrics(
        mu_draws_by_metric=aligned_draws,
        higher_is_better_by_metric=hib_by_metric,
        group_names=group_names,
        weights=METRIC_WEIGHTS,
    )

    # Generate plot
    overall_plot_name = "overall_prbest.png"
    plot_overall_pr_best(overall_rank_table, assets_dir / overall_plot_name)

    # Add to report
    report.add_card_start()
    report.add_h2("Overall multi-metric ranking")
    report.add_paragraph(
        "Composite ranking across all metrics using z-score standardization. "
        "This depends on your utility function (weights). Default: equal weights."
    )
    report.add_h3("Ranking table")
    add_table_with_column_guide(report, overall_rank_table, COLGUIDE_OVERALL_RANK)

    report.add_h3("Pr(best) plot")
    report.add_image(
        f"{assets_dir.name}/{overall_plot_name}",
        alt="Overall Pr(best) plot",
    )

    report.add_h3("Weights used")
    weights_df = (
        pd.DataFrame(
            [{"metric": k, "weight": float(v)} for k, v in used_weights.items()]
        )
        .sort_values("metric")
        .reset_index(drop=True)
    )
    add_table_with_column_guide(
        report,
        weights_df,
        {"metric": "Metric name.", "weight": "Weight used in composite scoring."},
    )
    report.add_card_end()

    # Factorial analysis of composite score
    report.add_card_start()
    report.add_h2("Factorial Analysis of Combined Posteriors")
    report.add_paragraph(
        "Beyond ranking, we now analyze the factorial structure of the composite score "
        "to understand dataset, architecture, and interaction effects on overall performance."
    )
    report.add_card_end()

    # Create synthetic dataset from composite draws
    composite_data_rows = []
    for g_idx, group in enumerate(group_names):
        parts = group.split(" | ")
        if len(parts) == 2:
            arch_part = parts[0].strip()
            dataset = parts[1].strip()
            
            if "Swin" in arch_part:
                arch = "Swin"
            elif "Terramind" in arch_part:
                arch = "Terramind"
            else:
                arch = "U-Net"
            
            for sample_idx in range(composite_draws.shape[0]):
                composite_data_rows.append({
                    "group": group,
                    "dataset": dataset,
                    "arch": arch,
                    "composite_score": composite_draws[sample_idx, g_idx]
                })
    
    composite_df = pd.DataFrame(composite_data_rows)
    
    # Fit factorial model on composite
    (
        idata_comp,
        factorial_report_comp,
        pred_table_comp,
        pred_draws_transformed_comp,
        pred_draws_original_comp,
        (datasets_comp, archs_comp),
        post_pred_comp,
    ) = fit_factorial_model_with_bf(
        composite_df,
        dataset_col="dataset",
        arch_col="arch",
        metric_col="composite_score",
        bounded_metrics=set(),  # Composite is not bounded
        sd_kde_bandwidths=SD_KDE_BANDWIDTHS,
    )

    # Generate plots
    heat_name_comp = "heat_composite.png"
    plot_factorial_heatmap(pred_table_comp, "Composite Score", assets_dir / heat_name_comp)

    eff_name_comp = "effects_composite.png"
    plot_factorial_effects(factorial_report_comp, "Composite Score", assets_dir / eff_name_comp)

    # Contrasts
    rope_comp = 0.1
    
    contrast_df_comp = contrasts_within_each_dataset(
        pred_draws_transformed_comp, pred_draws_original_comp,
        datasets_comp, archs_comp, BASELINE_ARCH,
        rope_comp, HDI_PROB
    )

    dataset_contrast_df_comp = contrasts_within_each_architecture(
        pred_draws_transformed_comp, pred_draws_original_comp,
        datasets_comp, archs_comp, BASELINE_DATASET,
        rope_comp, HDI_PROB
    )

    arch_overall_df_comp = contrasts_arch_overall(
        pred_draws_transformed_comp, pred_draws_original_comp,
        archs_comp, BASELINE_ARCH, rope_comp, HDI_PROB
    )
    
    dataset_overall_df_comp = contrasts_dataset_overall(
        pred_draws_transformed_comp, pred_draws_original_comp,
        datasets_comp, BASELINE_DATASET, rope_comp, HDI_PROB
    )

    # Generate contrast plots
    contrast_plot_comp = "contrasts_arch_composite.png"
    plot_contrasts(contrast_df_comp, "Composite Score", assets_dir / contrast_plot_comp,
                  title_suffix="(Architecture within Dataset)")

    dataset_contrast_plot_comp = "contrasts_dataset_composite.png"
    plot_contrasts(dataset_contrast_df_comp, "Composite Score", 
                  assets_dir / dataset_contrast_plot_comp,
                  title_suffix="(Dataset within Architecture)")

    arch_overall_plot_comp = "arch_overall_composite.png"
    plot_contrasts(arch_overall_df_comp, "Composite Score", 
                  assets_dir / arch_overall_plot_comp,
                  title_suffix="(Overall Architecture Effects)")

    dataset_overall_plot_comp = "dataset_overall_composite.png"
    plot_contrasts(dataset_overall_df_comp, "Composite Score", 
                  assets_dir / dataset_overall_plot_comp,
                  title_suffix="(Overall Dataset Effects)")

    # Add to report
    if factorial_report_comp["scale"].iloc[0] == "log":
        eff_view_comp = factorial_report_comp[[
            "param", "Pr(>0)", "BF10_interpretation",
            "BF10_bw_scott", "BF10_bw_0.1", "BF10_bw_0.3", "BF10_bw_0.5", "BF10_bw_silverman",
            "Abs_Δ_mean", "Abs_Δ_lo", "Abs_Δ_hi", "%_change", "baseline"
        ]].copy()
    else:
        eff_view_comp = factorial_report_comp[[
            "param", "Pr(>0)", "BF10_interpretation",
            "BF10_bw_scott", "BF10_bw_0.1", "BF10_bw_0.3", "BF10_bw_0.5", "BF10_bw_silverman",
            "OR", "Δp_mean", "Δp_lo", "Δp_hi", "baseline"
        ]].copy()

    report.add_card_start()
    report.add_h3("Factorial Effects (Composite Score)")
    report.add_paragraph(
        "Main effects and interactions for dataset, architecture, and their combination "
        "on the overall composite performance score."
    )
    
    if factorial_report_comp["scale"].iloc[0] == "log":
        add_table_with_column_guide(report, eff_view_comp, COLGUIDE_EFFECTS_LOG)
    else:
        add_table_with_column_guide(report, eff_view_comp, COLGUIDE_EFFECTS_LOGIT)

    report.add_h3("Factorial effects plot (Composite)")
    report.add_image(f"{assets_dir.name}/{eff_name_comp}", alt="Factorial effects plot for composite score")

    report.add_h3("Predicted means per dataset×arch (Composite)")
    pred_view_comp = pred_table_comp[
        ["dataset", "arch", "mean_original", "hdi_lo_original", "hdi_hi_original"]
    ].copy()
    pred_view_comp = pred_view_comp.rename(columns={
        "mean_original": "mean",
        "hdi_lo_original": "hdi_lo",
        "hdi_hi_original": "hdi_hi",
    })
    add_table_with_column_guide(report, pred_view_comp, COLGUIDE_PRED)

    report.add_h3("Heatmap of composite score")
    report.add_image(f"{assets_dir.name}/{heat_name_comp}", alt="Heatmap for composite score")

    report.add_h3(f"Architecture contrasts (Composite, baseline={BASELINE_ARCH})")
    contrast_view_comp = contrast_df_comp[
        ["dataset", "contrast", "mean_original", "hdi_lo_original", 
         "hdi_hi_original", "Pr(>0)", "Pr(|Δ|<ROPE)"]
    ].copy().rename(columns={
        "mean_original": "Δ_mean",
        "hdi_lo_original": "Δ_lo",
        "hdi_hi_original": "Δ_hi",
    })
    add_table_with_column_guide(report, contrast_view_comp, COLGUIDE_CONTRASTS_DATASET)

    report.add_h3("Architecture contrast plot (Composite)")
    report.add_image(f"{assets_dir.name}/{contrast_plot_comp}", alt="Architecture contrasts for composite score")

    report.add_h3(f"Dataset contrasts (Composite, baseline={BASELINE_DATASET})")
    dataset_contrast_view_comp = dataset_contrast_df_comp[
        ["arch", "contrast", "mean_original", "hdi_lo_original", 
         "hdi_hi_original", "Pr(>0)", "Pr(|Δ|<ROPE)"]
    ].copy().rename(columns={
        "mean_original": "Δ_mean",
        "hdi_lo_original": "Δ_lo",
        "hdi_hi_original": "Δ_hi",
    })
    add_table_with_column_guide(report, dataset_contrast_view_comp, COLGUIDE_CONTRASTS_ARCH)

    report.add_h3("Dataset contrast plot (Composite)")
    report.add_image(f"{assets_dir.name}/{dataset_contrast_plot_comp}", alt="Dataset contrasts for composite score")

    report.add_h3("Overall architecture contrasts (Composite)")
    arch_overall_view_comp = arch_overall_df_comp[
        ["scope", "contrast", "mean_original", "hdi_lo_original", 
         "hdi_hi_original", "Pr(>0)", "Pr(|Δ|<ROPE)"]
    ].copy()
    add_table_with_column_guide(report, arch_overall_view_comp, COLGUIDE_CONTRASTS_OVERALL)

    report.add_h3("Overall architecture plot (Composite)")
    report.add_image(f"{assets_dir.name}/{arch_overall_plot_comp}", alt="Overall architecture contrasts for composite")

    report.add_h3("Overall dataset contrasts (Composite)")
    dataset_overall_view_comp = dataset_overall_df_comp[
        ["scope", "contrast", "mean_original", "hdi_lo_original", 
         "hdi_hi_original", "Pr(>0)", "Pr(|Δ|<ROPE)"]
    ].copy()
    add_table_with_column_guide(report, dataset_overall_view_comp, COLGUIDE_CONTRASTS_OVERALL)

    report.add_h3("Overall dataset plot (Composite)")
    report.add_image(f"{assets_dir.name}/{dataset_overall_plot_comp}", alt="Overall dataset contrasts for composite")

    report.add_footnote(
        "Understanding composite factorial analysis",
        """
<ul>
  <li><b>Composite score</b> combines all metrics using z-score standardization and weighting.</li>
  <li><b>Main effects</b> show how dataset/architecture choices affect overall performance.</li>
  <li><b>Interactions</b> reveal whether certain dataset-architecture combinations perform 
      better/worse than expected from their individual effects.</li>
  <li><b>All visualizations include explicit values</b> for easy interpretation.</li>
  <li>This analysis complements the ranking by showing <i>why</i> certain groups rank higher.</li>
</ul>
""",
    )
    report.add_card_end()

    report.close_tab_content()


def main():
    """Main execution function."""
    print("="*80)
    print("Bayesian Benchmark Statistics Analysis")
    print("="*80)
    print(f"Running with PyMC version: {pm.__version__}")
    
    # Get output paths
    paths = get_output_paths()
    base_dir = paths["base_dir"]
    assets_dir = paths["assets_dir"]
    report_path = paths["report_path"]
    
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Initialize report
    report = HtmlReport(report_path, title="Bayesian Benchmark Statistics Report (Enhanced)")

    # Header
    report.add_html('<div class="topbar">')
    report.add_html(f'<span class="badge">PyMC {pm.__version__}</span>')
    report.add_html(
        f'<span class="badge">Hierarchical Variance | Multiple BF Bandwidths | Publication-Ready Plots</span>'
    )
    if TEST:
        report.add_html('<span class="badge" style="color:var(--accent2)">TESTMODE</span>')
    report.add_html("</div>")

    report.add_card_start()
    report.add_paragraph(
        "This ENHANCED report includes: (1) Complete factorial analysis for combined posteriors, "
        "(2) Full plot coverage for all contrasts, (3) Explicit quantification in all visualizations, "
        "(4) Publication-ready styling matching reference figures."
    )
    report.add_card_end()

    # Load data
    print("\nLoading data...")
    all_data = load_checkpoint_metrics(CSV_SPECS, METRICS)
    all_data["group"] = pd.Categorical(all_data["group"], categories=GROUP_ORDER, ordered=True)

    # Generate test data if enabled
    if TEST:
        print("TEST MODE: Generating synthetic data...")
        all_data = generate_test_data(
            dataset_order=DATASET_ORDER,
            arch_order=ARCH_ORDER,
            metrics=METRICS,
            group_order=GROUP_ORDER,
        )
        all_data["group"] = pd.Categorical(
            all_data["group"], categories=GROUP_ORDER, ordered=True
        )

    if all_data.empty:
        report.add_card_start()
        report.add_h2("No data loaded")
        report.add_paragraph("No CSVs were found/read successfully. Check your CSV paths.")
        report.add_card_end()
        report.close()
        print(f"Wrote report: {report_path}")
        return

    print(f"Loaded {len(all_data)} rows")

    # Setup tabs
    tab_list = [("tab-overview", "Overview")]
    for m in METRICS:
        tab_list.append((f"tab-{_safe_slug(m)}", m))
    tab_list.append(("tab-overall", "Overall"))

    report.start_tabs(tab_list, default_tab_id="tab-overview")

    # Overview tab
    print("\nGenerating overview tab...")
    add_overview_tab(report, all_data)

    # Store for overall ranking
    mu_draws_by_metric = {}
    hib_by_metric = {}
    groups_by_metric = {}

    # Process each metric
    for metric in METRICS:
        print(f"\nProcessing metric: {metric}")
        
        data = all_data[["group", "dataset", "arch", metric]].dropna().copy()

        if pd.api.types.is_categorical_dtype(data["group"]):
            data["group"] = data["group"].cat.remove_unused_categories()
        else:
            data["group"] = pd.Categorical(data["group"], categories=GROUP_ORDER, ordered=True)
            data["group"] = data["group"].remove_unused_categories()

        if data.empty:
            print(f"  Skipping {metric} - no data")
            continue

        add_metric_analysis(
            report=report,
            metric=metric,
            data=data,
            assets_dir=assets_dir,
            mu_draws_by_metric=mu_draws_by_metric,
            hib_by_metric=hib_by_metric,
            groups_by_metric=groups_by_metric,
        )
        
        print(f"  Completed {metric}")

    # Overall tab
    print("\nGenerating overall tab...")
    add_overall_tab(
        report=report,
        mu_draws_by_metric=mu_draws_by_metric,
        hib_by_metric=hib_by_metric,
        groups_by_metric=groups_by_metric,
        assets_dir=assets_dir,
    )

    # Finalize report
    report.close()
    
    print("\n" + "="*80)
    print(f"✓ Report written to: {report_path}")
    print(f"✓ Assets saved to: {assets_dir}")
    print("="*80)


if __name__ == "__main__":
    main()