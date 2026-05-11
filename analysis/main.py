# main.py
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Sequence

import matplotlib

# Use a non-interactive backend so the script can run headless (no GUI required).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymc as pm  # noqa: E402

from colguides import (
    COLGUIDE_COUNTS,
    COLGUIDE_PERF,
    COLGUIDE_EFFECTS_LOG,
    COLGUIDE_EFFECTS_LOGIT,
    COLGUIDE_PRED,
    COLGUIDE_CONTRASTS_DATASET,
    COLGUIDE_CONTRASTS_ARCH,
    COLGUIDE_CONTRASTS_OVERALL,
    COLGUIDE_OVERALL_RANK,
)
from config import (
    ARCH_ORDER,
    BASELINE_ARCH,
    BASELINE_DATASET,
    CHAINS,
    CSV_SPECS,
    DATASET_ORDER,
    DRAWS,
    GROUP_ORDER,
    HDI_PROB,
    METRICS,
    METRIC_WEIGHTS,
    OVERALL_SCORING_MODE,
    RANDOM_SEED,
    ROPE_BY_METRIC_TRANSFORMED,
    SD_KDE_BANDWIDTHS,
    TARGET_ACCEPT,
    TEST,
    TUNE,
)
from contrasts import (
    contrasts_arch_overall,
    contrasts_dataset_overall,
    contrasts_within_each_architecture,
    contrasts_within_each_dataset,
)
from data_io import load_checkpoint_metrics
from export_ppc_data import export_ppc_data
from models_group import (
    add_consecutive_rope_analysis,
    compute_pairwise_rope_vs_best,
    fit_group_model_with_ranking,
    rank_probabilities_from_draws,
)

from models_factorial import fit_factorial_model_with_bf
from overall_ranking import (
    add_overall_draw_diagnostics,
    overall_ranking_across_metrics,
)
from plots import (
    arch_from_group,
    dataset_from_group,
    plot_contrasts,
    plot_factorial_effects,
    plot_factorial_heatmap,
    plot_observed_boxplot,
    plot_overall_draw_top2,
    plot_overall_pr_best,
    plot_posterior_group_boxplot,
    plot_posterior_predictive_check,
    plot_pr_best,
    plot_prior_predictive_check,
)
from reporting import HtmlReport, add_table_with_column_guide, safe_slug
from transforms import higher_is_better, transform_y

# Suppress noisy threadpoolctl OpenMP runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*OpenMP.*")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    assets_dir = base_dir / "stats_report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    report_path = base_dir / "stats_report.html"
    report = HtmlReport(report_path, title="Bayesian Benchmark Statistics Report")

    # Header badges
    report.add_html('<div class="topbar">')
    report.add_html(f'<span class="badge">PyMC {pm.__version__}</span>')
    report.add_html(
        f'<span class="badge">Draws {DRAWS} | Tune {TUNE} | Chains {CHAINS}</span>'
    )
    report.add_html(
        f'<span class="badge">Target accept {TARGET_ACCEPT} | HDI {HDI_PROB:.0%}</span>'
    )
    report.add_html('<span class="badge">Hierarchical Variance | Multiple BF Bandwidths | ROPE Practical Significance</span>')
    if TEST:
        report.add_html('<span class="badge" style="color:var(--accent2)">TESTMODE</span>')
    report.add_html("</div>")

    report.add_card_start()
    report.add_paragraph(
        "This report includes: (1) Hierarchical variance structure for better regularization, "
        "(2) All outputs transformed to natural, interpretable scales, (3) Multiple bandwidth sensitivity "
        "analysis for Bayes factors, (4) Posterior and prior predictive checks, "
        "(5) ROPE probabilities reported alongside effects/contrasts for practical significance, "
        "(6) ROPE applied on transformed scale for consistency with the model, (7) Visualizations with actual values."
    )
    report.add_card_end()

    all_data = load_checkpoint_metrics(CSV_SPECS, METRICS)
    all_data["group"] = pd.Categorical(all_data["group"], categories=GROUP_ORDER, ordered=True)

    # Optional TEST mode synthetic data
    if TEST and not all_data.empty:
        rng = np.random.default_rng(RANDOM_SEED)

        def make_group(dataset: str, arch: str) -> str:
            if arch == "U-Net":
                return f"U-Net | {dataset}"
            if arch == "Swin":
                return f"Swin U-Net | {dataset}"
            if arch == "Terramind":
                return f"Terramind | {dataset}"
            return f"{arch} | {dataset}"

        N_ROWS = 10
        KAPPA_BY_METRIC = {m: 120.0 for m in METRICS}

        BASE_MEAN = {
            "IoU": 0.45,
            "dice_coef": 0.55,
            "normalized_surface_distance": 0.35,
            "mean_epistemic_uncertainty": 0.25,
            "mean_aleatoric_uncertainty": 0.25,
        }

        ARCH_SHIFT = {
            "U-Net": {m: 0.0 for m in METRICS},
            "Swin": {
                "IoU": 0.15,
                "dice_coef": 0.12,
                "normalized_surface_distance": 0.05,
                "mean_epistemic_uncertainty": -0.03,
                "mean_aleatoric_uncertainty": -0.02,
            },
            "Terramind": {
                "IoU": -0.10,
                "dice_coef": -0.10,
                "normalized_surface_distance": -0.05,
                "mean_epistemic_uncertainty": 0.08,
                "mean_aleatoric_uncertainty": 0.06,
            },
        }

        DATASET_SHIFT = {
            "AE": {
                "IoU": 0.10,
                "dice_coef": 0.08,
                "normalized_surface_distance": 0.03,
                "mean_epistemic_uncertainty": -0.02,
                "mean_aleatoric_uncertainty": -0.01,
            },
            "PS": {m: 0.0 for m in METRICS},
            "S2": {
                "IoU": -0.12,
                "dice_coef": -0.10,
                "normalized_surface_distance": -0.05,
                "mean_epistemic_uncertainty": 0.05,
                "mean_aleatoric_uncertainty": 0.04,
            },
        }

        MEAN_JITTER_SD = 0.015

        def beta_sample(mu: float, kappa: float, n: int) -> np.ndarray:
            mu = float(np.clip(mu, 1e-4, 1 - 1e-4))
            a = mu * kappa
            b = (1.0 - mu) * kappa
            return rng.beta(a, b, size=n)

        test_frames = []
        for dataset in DATASET_ORDER:
            for arch in ARCH_ORDER:
                group = make_group(dataset, arch)

                row = {
                    "dataset": [dataset] * N_ROWS,
                    "arch": [arch] * N_ROWS,
                    "group": [group] * N_ROWS,
                }
                for metric in METRICS:
                    mu = (
                        BASE_MEAN[metric]
                        + ARCH_SHIFT[arch][metric]
                        + DATASET_SHIFT[dataset][metric]
                        + rng.normal(0.0, MEAN_JITTER_SD)
                    )
                    mu = float(np.clip(mu, 1e-3, 1 - 1e-3))
                    row[metric] = beta_sample(mu, KAPPA_BY_METRIC[metric], N_ROWS)

                test_frames.append(pd.DataFrame(row))

        all_data = pd.concat(test_frames, ignore_index=True)
        all_data["group"] = pd.Categorical(all_data["group"], categories=GROUP_ORDER, ordered=True)

    if all_data.empty:
        report.add_card_start()
        report.add_h2("No data loaded")
        report.add_paragraph("No CSVs were found/read successfully. Check your CSV paths.")
        report.add_card_end()
        report.close()
        print(f"Wrote report: {report_path}")
        return

    # Tabs
    tab_list = [("tab-overview", "Overview")]
    for m in METRICS:
        tab_list.append((f"tab-{safe_slug(m)}", m))
    tab_list.append(("tab-overall", "Overall"))
    report.start_tabs(tab_list, default_tab_id="tab-overview")

    # =============================================================================
    # OVERVIEW TAB
    # =============================================================================
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
    report.add_h2("Statistical Improvements in This Report")
    report.add_paragraph(
        "<strong>1. Hierarchical Variance:</strong> All models now use hierarchical priors for sigma, "
        "improving regularization even with equal sample sizes (n=10)."
    )
    report.add_paragraph(
        "<strong>2. Natural Scale Outputs:</strong> All effects are reported on both transformed "
        "and natural/interpretable scales (e.g., absolute changes, percentage changes, probability changes)."
    )
    report.add_paragraph(
        "<strong>3. BF Sensitivity Analysis:</strong> Bayes factors computed with multiple KDE bandwidths "
        f"({', '.join([str(b) for b in SD_KDE_BANDWIDTHS])}) to assess robustness."
    )
    report.add_paragraph(
        "<strong>4. ROPE Consistency:</strong> Region of Practical Equivalence (ROPE) is now applied "
        "on the transformed scale where the model lives, ensuring statistical consistency."
    )
    report.add_paragraph(
        "<strong>ROPE probabilities:</strong> In addition to Pr(>0), we report "
        "Pr(Δ>ROPE), Pr(|Δ|<ROPE), and Pr(Δ<-ROPE) as a coherent, practical-significance measure."
    )
    report.add_paragraph(
        "<strong>5. Predictive Checks:</strong> Prior and posterior predictive checks included to "
        "validate model assumptions and fit."
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
  <li><b>Pr(Δ&gt;ROPE)</b> = probability the effect is meaningfully positive (exceeds ROPE).</li>
  <li><b>Pr(|Δ|&lt;ROPE)</b> = probability the effect is practically negligible (a practical tie).</li>
  <li><b>Pr(Δ&lt;-ROPE)</b> = probability the effect is meaningfully negative (below -ROPE).</li>
  <li><b>Pr(&gt;0)</b> can be high even for tiny effects; prefer ROPE probabilities for practical significance.</li>
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

    # Store for overall ranking
    mu_draws_by_metric: Dict[str, np.ndarray] = {}
    hib_by_metric: Dict[str, bool] = {}
    groups_by_metric: Dict[str, Sequence[str]] = {}

    # =============================================================================
    # METRIC TABS
    # =============================================================================
    for metric in METRICS:
        tab_id = f"tab-{safe_slug(metric)}"
        report.open_tab_content(tab_id)

        data = all_data[["group", "dataset", "arch", metric]].dropna().copy()

        if isinstance(data["group"].dtype, pd.CategoricalDtype):
            data["group"] = data["group"].cat.remove_unused_categories()
        else:
            data["group"] = pd.Categorical(data["group"], categories=GROUP_ORDER, ordered=True)
            data["group"] = data["group"].remove_unused_categories()

        report.add_card_start()
        report.add_h2(f"Metric: {metric}")
        if TEST:
            report.add_html('<span class="badge" style="color:var(--accent2)">TESTMODE</span>')
        report.add_card_end()

        if data.empty:
            report.add_card_start()
            report.add_paragraph("No rows available after dropping NaNs for this metric.")
            report.add_card_end()
            report.close_tab_content()
            continue

        metric_std = data[metric].dropna().std()
        if metric_std < 1e-12:
            report.add_card_start()
            report.add_paragraph(
                f"The metric <strong>{metric}</strong> is nearly constant across all observations "
                f"(std = {metric_std:.2e}). This provides no information for model comparison. "
                "Skipping Bayesian analysis for this metric."
            )
            report.add_card_end()
            report.close_tab_content()
            continue

        y_raw = data[metric].to_numpy()
        y_t, inv, tname = transform_y(y_raw, metric)

        if not np.all(np.isfinite(y_t)):
            report.add_card_start()
            report.add_paragraph(
                f"The metric <strong>{metric}</strong> produced non-finite transformed values. "
                "Skipping analysis."
            )
            report.add_card_end()
            report.close_tab_content()
            continue

        if np.std(y_t) < 1e-12:
            report.add_card_start()
            report.add_paragraph(
                f"The metric <strong>{metric}</strong> has near-zero transformed variance (std={np.std(y_t):.2e}). "
                "Skipping analysis."
            )
            report.add_card_end()
            report.close_tab_content()
            continue

        if tname == "log":
            clip_frac = np.mean(y_raw <= 1e-12)
            if clip_frac >= 0.98:
                report.add_card_start()
                report.add_paragraph(
                    f"The metric <strong>{metric}</strong> has {clip_frac*100:.1f}% of values below "
                    "the log-scale clipping threshold (1e-12). This metric is essentially all zeros "
                    "and provides no information. Skipping analysis."
                )
                report.add_card_end()
                report.close_tab_content()
                continue

        # Observed distribution plot
        obs_name = f"observed_box_{safe_slug(metric)}.png"
        plot_observed_boxplot(data, metric, assets_dir / obs_name, group_order=GROUP_ORDER)

        report.add_card_start()
        report.add_h3("Observed results across runs")
        report.add_image(f"{assets_dir.name}/{obs_name}", alt=f"Observed boxplot for {metric}")
        report.add_card_end()

        # Group model
        (
            idata,
            perf_table,
            _post_rank_table,
            mu_draws,
            scale_name,
            groups_this,
            post_pred,
            prior_mu_draws,
        ) = fit_group_model_with_ranking(data, group_col="group", metric_col=metric)

        # Observations aligned to the fitted model's y indexing
        obs_df = data.loc[:, ["group", metric]].dropna().copy()
        obs_groups = obs_df["group"].astype(str).to_numpy()
        y_obs_raw = obs_df[metric].to_numpy()
        y_obs_t, _, _ = transform_y(y_obs_raw, metric)

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

        # Prior predictive check plot (model-driven, original scale)
        prior_pred_name = f"prior_pred_{safe_slug(metric)}.png"
        plot_prior_predictive_check(
            prior_mu_draws=prior_mu_draws,
            y_obs_original=y_obs_raw,
            obs_groups=obs_groups,
            groups=groups_this,
            out_path=assets_dir / prior_pred_name,
            metric=metric,
        )

        # Posterior predictive check plot
        post_pred_name = f"post_pred_{safe_slug(metric)}.png"
        plot_posterior_predictive_check(
            post_pred=post_pred,
            y_obs_transformed=y_obs_t,
            inv_transform=inv,
            obs_groups=obs_groups,
            group_order=GROUP_ORDER,
            metric=metric,
            out_path=assets_dir / post_pred_name,
        )

        # Group plots
        forest_name = f"posterior_box_{safe_slug(metric)}.png"
        plot_posterior_group_boxplot(
            mu_draws=mu_draws,
            groups=groups_this,
            metric=metric,
            out_path=assets_dir / forest_name,
            group_order=GROUP_ORDER,
        )

        prbest_name = f"prbest_{safe_slug(metric)}.png"
        plot_pr_best(perf_table, metric, assets_dir / prbest_name)

        report.add_card_start()
        report.add_h3("Hierarchical group model (with hierarchical variance)")
        report.add_paragraph(
            f"Modelled on <span class='badge'>{scale_name}</span> scale; "
            "reporting mu_orig on original scale. Hierarchical variance improves regularization."
        )
        
        # Compute ROPE vs best for this metric
        rope_transformed = float(ROPE_BY_METRIC_TRANSFORMED.get(metric, 0.05))
        rope_df = compute_pairwise_rope_vs_best(
            idata=idata,
            mu_draws_original=mu_draws,
            perf_table=perf_table,
            group_names=groups_this,
            higher_better=higher_is_better(metric),
            rope_transformed=rope_transformed,
        )
        
        # Merge ROPE columns into perf_table (skip the vs_group column for best group, keep it for others)
        perf_table_with_rope = perf_table.copy()
        rope_cols_to_add = ["Pr(Δ>ROPE)", "Pr(|Δ|<ROPE)", "Pr(Δ<-ROPE)"]
        for col in rope_cols_to_add:
            perf_table_with_rope[col] = rope_df.set_index("group").loc[perf_table_with_rope["group"], col].values
        
        add_table_with_column_guide(report, perf_table_with_rope, COLGUIDE_PERF)

        report.add_h3("Model Validation Plots")
        report.add_html('<div class="grid-2">')
        report.add_html('<div class="card" style="margin:0">')
        report.add_h3("Prior Predictive Check")
        report.add_image(f"{assets_dir.name}/{prior_pred_name}", alt=f"Prior predictive check for {metric}")
        report.add_paragraph("Check if priors are reasonable on original scale.")
        report.add_html("</div>")
        report.add_html('<div class="card" style="margin:0">')
        report.add_h3("Posterior Predictive Check")
        report.add_image(f"{assets_dir.name}/{post_pred_name}", alt=f"Posterior predictive check for {metric}")
        report.add_paragraph("Red overlay shows model predictions; blue shows observed data.")
        report.add_html("</div>")
        report.add_html("</div>")

        report.add_h3("Group Comparison Plots")
        report.add_html('<div class="grid-2">')
        report.add_html('<div class="card" style="margin:0">')
        report.add_h3("Group means (with HDI)")
        report.add_image(f"{assets_dir.name}/{forest_name}", alt=f"Group means forest plot for {metric}")
        report.add_html("</div>")
        report.add_html('<div class="card" style="margin:0">')
        report.add_h3("Pr(best)")
        report.add_image(f"{assets_dir.name}/{prbest_name}", alt=f"Pr(best) plot for {metric}")
        report.add_html("</div>")
        report.add_html("</div>")
        report.add_card_end()

        # Store for overall
        mu_draws_by_metric[metric] = mu_draws
        hib_by_metric[metric] = higher_is_better(metric)
        groups_by_metric[metric] = list(groups_this)

        # Metric-specific ROPE (on transformed scale)
        rope_transformed = float(ROPE_BY_METRIC_TRANSFORMED.get(metric, 0.05))

        # Consecutive ranking BF analysis
        add_consecutive_rope_analysis(
            report=report,
            idata=idata,
            mu_draws_original=mu_draws,
            perf_table=perf_table,
            group_names=groups_this,
            metric=metric,
            higher_better=higher_is_better(metric),
            rope_transformed=rope_transformed,
            assets_dir=assets_dir,
        )
        if add_consecutive_rope_analysis is not None:
            add_consecutive_rope_analysis(
                report=report,
                idata=idata,
                mu_draws_original=mu_draws,
                perf_table=perf_table,
                group_names=groups_this,
                metric=metric,
                higher_better=higher_is_better(metric),
                rope_transformed=rope_transformed,
                assets_dir=assets_dir,
            )

        # Factorial model
        try:
            (
                _idata_fact,
                factorial_report,
                pred_table,
                pred_draws_transformed,
                pred_draws_original,
                (datasets, archs),
                _post_pred_fact,
            ) = fit_factorial_model_with_bf(
                data,
                dataset_col="dataset",
                arch_col="arch",
                metric_col=metric,
                rope_transformed=rope_transformed,
            )
        except TypeError:
            # Backwards compatibility: older version without ROPE argument
            (
                _idata_fact,
                factorial_report,
                pred_table,
                pred_draws_transformed,
                pred_draws_original,
                (datasets, archs),
                _post_pred_fact,
            ) = fit_factorial_model_with_bf(
                data,
                dataset_col="dataset",
                arch_col="arch",
                metric_col=metric,
            )

        # Factorial plots
        heat_name = f"heat_{safe_slug(metric)}.png"
        plot_factorial_heatmap(pred_table, metric, assets_dir / heat_name)

        eff_name = f"effects_{safe_slug(metric)}.png"
        plot_factorial_effects(factorial_report, metric, assets_dir / eff_name)

        # Contrasts
        # (ROPE already defined above on transformed scale)

        contrast_df = contrasts_within_each_dataset(
            pred_draws_transformed=pred_draws_transformed,
            pred_draws_original=pred_draws_original,
            datasets=datasets,
            archs=archs,
            baseline_arch=BASELINE_ARCH,
            rope_transformed=rope_transformed,
            hdi_prob=HDI_PROB,
        )

        dataset_contrast_df = contrasts_within_each_architecture(
            pred_draws_transformed=pred_draws_transformed,
            pred_draws_original=pred_draws_original,
            datasets=datasets,
            archs=archs,
            baseline_dataset=BASELINE_DATASET,
            rope_transformed=rope_transformed,
            hdi_prob=HDI_PROB,
        )

        arch_overall_df = contrasts_arch_overall(
            pred_draws_transformed=pred_draws_transformed,
            pred_draws_original=pred_draws_original,
            archs=archs,
            baseline_arch=BASELINE_ARCH,
            rope_transformed=rope_transformed,
            hdi_prob=HDI_PROB,
        )

        dataset_overall_df = contrasts_dataset_overall(
            pred_draws_transformed=pred_draws_transformed,
            pred_draws_original=pred_draws_original,
            datasets=datasets,
            baseline_dataset=BASELINE_DATASET,
            rope_transformed=rope_transformed,
            hdi_prob=HDI_PROB,
        )

        contrast_plot_name = f"architecture_contrasts_{safe_slug(metric)}.png"
        plot_contrasts(contrast_df, metric, assets_dir / contrast_plot_name)

        dataset_contrast_plot_name = f"dataset_contrasts_{safe_slug(metric)}.png"
        plot_contrasts(dataset_contrast_df, metric, assets_dir / dataset_contrast_plot_name)

        arch_overall_plot_name = f"arch_overall_{safe_slug(metric)}.png"
        plot_contrasts(arch_overall_df, metric, assets_dir / arch_overall_plot_name)

        dataset_overall_plot_name = f"dataset_overall_{safe_slug(metric)}.png"
        plot_contrasts(dataset_overall_df, metric, assets_dir / dataset_overall_plot_name)

        # Interpretable effects view (include ROPE columns if available)
        if factorial_report["scale"].iloc[0] == "log":
            cols = [
                "param",
                "Pr(>0)",
                "Pr(Δ>ROPE)",
                "Pr(|Δ|<ROPE)",
                "Pr(Δ<-ROPE)",
                "BF10_interpretation",
                "BF10_bw_scott",
                "BF10_bw_0.1",
                "BF10_bw_0.3",
                "BF10_bw_0.5",
                "BF10_bw_silverman",
                "Abs_Δ_mean",
                "Abs_Δ_lo",
                "Abs_Δ_hi",
                "%_change",
                "baseline",
            ]
        else:
            cols = [
                "param",
                "Pr(>0)",
                "Pr(Δ>ROPE)",
                "Pr(|Δ|<ROPE)",
                "Pr(Δ<-ROPE)",
                "BF10_interpretation",
                "BF10_bw_scott",
                "BF10_bw_0.1",
                "BF10_bw_0.3",
                "BF10_bw_0.5",
                "BF10_bw_silverman",
                "OR",
                "Δp_mean",
                "Δp_lo",
                "Δp_hi",
                "baseline",
            ]

        eff_view = factorial_report[[c for c in cols if c in factorial_report.columns]].copy()

        report.add_card_start()
        report.add_h3("Factorial effects (Natural Scale with BF Sensitivity + ROPE)")
        report.add_paragraph(
            "All effects shown on natural, interpretable scales. Multiple BF bandwidths show robustness of evidence. "
            "ROPE probabilities add a practical-significance view (meaningful vs negligible effects)."
        )
        if factorial_report["scale"].iloc[0] == "log":
            add_table_with_column_guide(report, eff_view, COLGUIDE_EFFECTS_LOG)
        else:
            add_table_with_column_guide(report, eff_view, COLGUIDE_EFFECTS_LOGIT)

        report.add_h3("Factorial effects plot")
        report.add_image(f"{assets_dir.name}/{eff_name}", alt=f"Factorial effects plot for {metric}")

        report.add_h3("Predicted means per dataset×arch")
        pred_view = pred_table[["dataset", "arch", "mean_original", "hdi_lo_original", "hdi_hi_original"]].copy()
        pred_view = pred_view.rename(
            columns={"mean_original": "mean", "hdi_lo_original": "hdi_lo", "hdi_hi_original": "hdi_hi"}
        )
        add_table_with_column_guide(report, pred_view, COLGUIDE_PRED)

        report.add_h3("Heatmap of posterior mean (Original Scale)")
        report.add_image(f"{assets_dir.name}/{heat_name}", alt=f"Heatmap for {metric}")

        report.add_h3(
            f"Key architecture contrasts (within each dataset, baseline arch={BASELINE_ARCH}; ROPE={rope_transformed} on transformed scale)"
        )
        cols = [
            "dataset",
            "contrast",
            "mean_original",
            "hdi_lo_original",
            "hdi_hi_original",
            "Pr(Δ>ROPE)",
            "Pr(|Δ|<ROPE)",
            "Pr(Δ<-ROPE)",
            "Pr(>0)",
        ]
        contrast_view = contrast_df[[c for c in cols if c in contrast_df.columns]].copy().rename(
            columns={"mean_original": "Δ_mean", "hdi_lo_original": "Δ_lo", "hdi_hi_original": "Δ_hi"}
        )
        add_table_with_column_guide(report, contrast_view, COLGUIDE_CONTRASTS_DATASET)
        if not contrast_df.empty:
            report.add_h3("Architecture Contrast plot")
            report.add_image(f"{assets_dir.name}/{contrast_plot_name}", alt=f"Contrast plot for {metric}")

        report.add_h3(
            f"Key dataset contrasts (within each architecture, baseline dataset={BASELINE_DATASET}; ROPE={rope_transformed} on transformed scale)"
        )
        if not dataset_contrast_df.empty:
            report.add_h3("Dataset contrast plot")
            report.add_image(f"{assets_dir.name}/{dataset_contrast_plot_name}", alt=f"Dataset contrast plot for {metric}")

        dataset_contrast_view = dataset_contrast_df[
            [c for c in [
                "arch",
                "contrast",
                "mean_original",
                "hdi_lo_original",
                "hdi_hi_original",
                "Pr(Δ>ROPE)",
                "Pr(|Δ|<ROPE)",
                "Pr(Δ<-ROPE)",
                "Pr(>0)",
            ] if c in dataset_contrast_df.columns]
        ].copy().rename(
            columns={"mean_original": "Δ_mean", "hdi_lo_original": "Δ_lo", "hdi_hi_original": "Δ_hi"}
        )
        add_table_with_column_guide(report, dataset_contrast_view, COLGUIDE_CONTRASTS_ARCH)

        report.add_h3(f"Overall architecture contrasts (marginal over datasets, baseline arch={BASELINE_ARCH})")
        cols = [
            "scope",
            "contrast",
            "mean_original",
            "hdi_lo_original",
            "hdi_hi_original",
            "Pr(Δ>ROPE)",
            "Pr(|Δ|<ROPE)",
            "Pr(Δ<-ROPE)",
            "Pr(>0)",
        ]
        arch_overall_view = arch_overall_df[[c for c in cols if c in arch_overall_df.columns]].copy()
        add_table_with_column_guide(report, arch_overall_view, COLGUIDE_CONTRASTS_OVERALL)
        if not arch_overall_df.empty:
            report.add_image(f"{assets_dir.name}/{arch_overall_plot_name}", alt=f"Overall architecture contrasts for {metric}")

        report.add_h3(f"Overall dataset contrasts (marginal over architectures, baseline dataset={BASELINE_DATASET})")
        cols = [
            "scope",
            "contrast",
            "mean_original",
            "hdi_lo_original",
            "hdi_hi_original",
            "Pr(Δ>ROPE)",
            "Pr(|Δ|<ROPE)",
            "Pr(Δ<-ROPE)",
            "Pr(>0)",
        ]
        dataset_overall_view = dataset_overall_df[[c for c in cols if c in dataset_overall_df.columns]].copy()
        add_table_with_column_guide(report, dataset_overall_view, COLGUIDE_CONTRASTS_OVERALL)
        if not dataset_overall_df.empty:
            report.add_image(f"{assets_dir.name}/{dataset_overall_plot_name}", alt=f"Overall dataset contrasts for {metric}")

        report.add_footnote(
            "Understanding the factorial section",
            """
<ul>
  <li><b>Effects coding:</b> Coefficients are not simple “add to intercept” baselines.</li>
  <li><b>Interpretable effects:</b> We compute OR/Δp (or absolute change) from <b>contrasts on predicted means</b>, which is correct under effects coding.</li>
  <li><b>BF sensitivity:</b> If BF values are similar across bandwidths, the evidence is robust.</li>
  <li><b>ROPE on transformed scale:</b> For consistency with the model parameter space where effects are estimated.</li>  <li><b>Practical significance:</b> Prefer Pr(Δ&gt;ROPE)/Pr(|Δ|&lt;ROPE)/Pr(Δ&lt;-ROPE) over Pr(&gt;0) when judging meaningful differences.</li></ul>
""",
        )
        report.add_card_end()

        report.close_tab_content()

    # =============================================================================
    # OVERALL TAB
    # =============================================================================
    report.open_tab_content("tab-overall")

    if mu_draws_by_metric:
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
        else:
            group_names = [g for g in GROUP_ORDER if g in common_groups]

            print("\n=== METRIC DIRECTIONS ===")
            for m in METRICS:
                if m in hib_by_metric:
                    direction = "↑ higher is better" if hib_by_metric[m] else "↓ lower is better"
                    print(f"  {m}: {direction}")
            print()

            aligned_draws: Dict[str, np.ndarray] = {}
            for m in METRICS:
                if m not in mu_draws_by_metric:
                    continue
                cols = list(groups_by_metric[m])
                missing = set(group_names) - set(cols)
                if missing:
                    raise RuntimeError(
                        f"Metric '{m}' is missing groups {missing}. Expected {group_names}, got {cols}"
                    )
                idx = [cols.index(g) for g in group_names]
                aligned_draws[m] = mu_draws_by_metric[m][:, idx]

            print(f"Using scoring mode: {OVERALL_SCORING_MODE}")
            composite, overall_rank_table, used_weights = overall_ranking_across_metrics(
                mu_draws_by_metric=aligned_draws,
                higher_is_better_by_metric=hib_by_metric,
                group_names=group_names,
                weights=METRIC_WEIGHTS,
                mode=OVERALL_SCORING_MODE,
            )

            overall_rank_table = add_overall_draw_diagnostics(
                overall_rank_table=overall_rank_table,
                composite=composite,
                group_names=group_names,
                tie_eps=0.1,
                ci=(0.05, 0.95),
                topk=2,
                rope_composite=0.2,  # ROPE threshold on composite scale
            )

            # Compute marginal rankings for architectures and datasets
            arch_labels = [arch_from_group(g) for g in group_names]
            ds_labels = [dataset_from_group(g) for g in group_names]

            archs = [a for a in ARCH_ORDER]
            datasets = [d for d in DATASET_ORDER]

            # Build arch_draws: shape (samples, len(archs))
            arch_draws = np.zeros((composite.shape[0], len(archs)))
            for i, arch in enumerate(archs):
                cols = [j for j, label in enumerate(arch_labels) if label == arch]
                if cols:
                    arch_draws[:, i] = composite[:, cols].mean(axis=1)

            # Build dataset_draws: shape (samples, len(datasets))
            dataset_draws = np.zeros((composite.shape[0], len(datasets)))
            for i, dataset in enumerate(datasets):
                cols = [j for j, label in enumerate(ds_labels) if label == dataset]
                if cols:
                    dataset_draws[:, i] = composite[:, cols].mean(axis=1)

            # Compute ranking tables
            arch_rank_table = rank_probabilities_from_draws(arch_draws, archs, higher_better=True)
            dataset_rank_table = rank_probabilities_from_draws(dataset_draws, datasets, higher_better=True)
            
            # Add ROPE diagnostics to marginal rankings (rope_composite=0.2 on composite scale)
            best_arch_idx = arch_rank_table.iloc[0:1].index[0]
            best_arch = arch_rank_table.iloc[0]["group"]
            best_arch_draws_idx = archs.index(best_arch)
            
            for i, arch in enumerate(archs):
                delta_arch = arch_draws[:, i] - arch_draws[:, best_arch_draws_idx]
                arch_rank_table.loc[arch_rank_table["group"] == arch, "Pr(Δ>ROPE)"] = (delta_arch > 0.2).mean()
                arch_rank_table.loc[arch_rank_table["group"] == arch, "Pr(|Δ|<ROPE)"] = (np.abs(delta_arch) < 0.2).mean()
                arch_rank_table.loc[arch_rank_table["group"] == arch, "Pr(Δ<-ROPE)"] = (delta_arch < -0.2).mean()
            
            best_ds = dataset_rank_table.iloc[0]["group"]
            best_ds_draws_idx = datasets.index(best_ds)
            
            for i, dataset in enumerate(datasets):
                delta_ds = dataset_draws[:, i] - dataset_draws[:, best_ds_draws_idx]
                dataset_rank_table.loc[dataset_rank_table["group"] == dataset, "Pr(Δ>ROPE)"] = (delta_ds > 0.2).mean()
                dataset_rank_table.loc[dataset_rank_table["group"] == dataset, "Pr(|Δ|<ROPE)"] = (np.abs(delta_ds) < 0.2).mean()
                dataset_rank_table.loc[dataset_rank_table["group"] == dataset, "Pr(Δ<-ROPE)"] = (delta_ds < -0.2).mean()

            overall_plot_name = "overall_prbest.png"
            plot_overall_pr_best(overall_rank_table, assets_dir / overall_plot_name)

            overall_draw_plot_name = "overall_draw_top2.png"
            plot_overall_draw_top2(overall_rank_table, assets_dir / overall_draw_plot_name)

            report.add_card_start()
            report.add_h2("Overall multi-metric ranking")

            mode_explanation = {
                "rank": (
                    "Rank-based utility scoring (ROBUST): For each posterior draw, converts metric values "
                    "to ranks (1=best), then to utility scores where best=G, worst=1. Aggregates utilities "
                    "across metrics. This mode is highly robust to outliers and scale differences, and ensures "
                    "the overall winner reflects consistent high rankings across metrics."
                ),
                "robust_z": (
                    "Robust z-score standardization: Uses median and MAD (median absolute deviation) for "
                    "centering and scaling per draw. More robust to outliers than standard z-scores. "
                    "Extreme z-values are clipped to [-8, 8]."
                ),
                "z": (
                    "Standard z-score standardization: Uses mean and standard deviation per draw. "
                    "Extreme z-values are clipped to [-8, 8] for stability."
                ),
            }

            report.add_paragraph(
                f"<b>Scoring mode:</b> {OVERALL_SCORING_MODE}. "
                f"{mode_explanation.get(OVERALL_SCORING_MODE, 'Unknown mode.')}"
            )
            report.add_paragraph(
                "<b>Metric directions:</b> Uncertainties (epistemic, aleatoric) are treated as lower-is-better "
                "(sign flipped before aggregation). Performance metrics (IoU, Dice, NSD) are higher-is-better. "
                "This ensures all metrics contribute consistently to the overall ranking."
            )
            report.add_paragraph(
                "<b>Weights:</b> Default is equal weighting across metrics. You can adjust via METRIC_WEIGHTS."
            )

            report.add_h3("Ranking table")
            add_table_with_column_guide(report, overall_rank_table, COLGUIDE_OVERALL_RANK)

            # Add marginal ranking tables
            report.add_h3("Overall architecture ranking (marginal over datasets)")
            report.add_paragraph(
                "This table shows the ranking of architectures averaged across all datasets. "
                "Each architecture's composite score is the average of its performance across datasets."
            )
            arch_colguide = {
                "group": "Architecture name.",
                "E[rank]": "Expected rank (lower is better).",
                "Pr(best)": "Probability of being the best architecture overall.",
            }
            for k in range(1, len(archs) + 1):
                arch_colguide[f"Pr(rank={k})"] = f"Probability of ranking #{k}."
            add_table_with_column_guide(report, arch_rank_table, arch_colguide)

            report.add_h3("Overall dataset ranking (marginal over architectures)")
            report.add_paragraph(
                "This table shows the ranking of datasets averaged across all architectures. "
                "Each dataset's composite score is the average of its performance across architectures."
            )
            ds_colguide = {
                "group": "Dataset name.",
                "E[rank]": "Expected rank (lower is better).",
                "Pr(best)": "Probability of being the best dataset overall.",
            }
            for k in range(1, len(datasets) + 1):
                ds_colguide[f"Pr(rank={k})"] = f"Probability of ranking #{k}."
            add_table_with_column_guide(report, dataset_rank_table, ds_colguide)

            report.add_h3("Pr(best) plot")
            report.add_image(f"{assets_dir.name}/{overall_plot_name}", alt="Overall Pr(best) plot")

            report.add_h3("Top-2 draw diagnostics")
            report.add_paragraph(
                "Draw diagnostics compare the top-ranked group against the runner-up. "
                "<b>Pr(win)</b> is the probability the top group has higher composite score. "
                "<b>Pr(tie)</b> is the probability of a near-draw (|difference| < 0.1). "
                "<b>Draw index</b> measures decisiveness: 1.0 = clear winner, 0.0 = coin flip."
            )
            report.add_image(f"{assets_dir.name}/{overall_draw_plot_name}", alt="Top-2 draw diagnostics")

            report.add_h3("Weights used")
            weights_df = (
                pd.DataFrame([{"metric": k, "weight": float(v)} for k, v in used_weights.items()])
                .sort_values("metric")
                .reset_index(drop=True)
            )
            add_table_with_column_guide(
                report,
                weights_df,
                {"metric": "Metric name.", "weight": "Weight used in composite scoring."},
            )

            report.add_footnote(
                "Overall ranking interpretation",
                """
<ul>
  <li><b>Pr(rank=1)</b> = probability a group is best overall under this utility.</li>
  <li><b>E[rank]</b> = expected rank (lower = better).</li>
  <li>If probabilities are spread, the winner is uncertain—consider individual metrics.</li>
  <li><b>Important:</b> This is a composite score. Always check individual metrics for full picture.</li>
</ul>
""",
            )
            report.add_card_end()

    report.close_tab_content()

    report.close()
    print(f"Wrote report: {report_path}")
    print(f"Assets saved to: {assets_dir}")


if __name__ == "__main__":
    main()
