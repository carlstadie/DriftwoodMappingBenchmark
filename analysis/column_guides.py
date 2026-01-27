"""
Column guide documentation for all tables in the report.
"""

# =============================================================================
# DATA SUMMARY TABLES
# =============================================================================

COLGUIDE_COUNTS = {
    "dataset": "Dataset label (AE/PS/S2).",
    "arch": "Architecture label (U-Net/Swin/Terramind).",
    "group": "Combined group name used in the group model.",
    "n_rows": "Number of runs/rows in this cell.",
}

# =============================================================================
# PERFORMANCE TABLES
# =============================================================================

COLGUIDE_PERF = {
    "group": "Group name (dataset×arch).",
    "mean": "Posterior mean of group mean on the ORIGINAL metric scale.",
    "sd": "Posterior SD of the group mean.",
    "hdi_2.5%": "Lower bound of the 95% HDI for the group mean.",
    "hdi_97.5%": "Upper bound of the 95% HDI for the group mean.",
    "E[rank]": "Expected rank (1=best).",
    "Pr(best)": "Posterior probability the group is the best.",
    "Pr(best)_prior": "Prior probability the group is best (before seeing data).",
    "BF_best": "Bayes factor for 'this group is best' vs 'not best' (via odds update).",
}

# Add dynamic rank columns
for k in range(1, 10):
    COLGUIDE_PERF[f"Pr(rank={k})"] = f"Posterior probability this group is rank {k} (1=best)."

# =============================================================================
# FACTORIAL EFFECTS TABLES
# =============================================================================

COLGUIDE_EFFECTS_LOGIT = {
    "param": "Model term (dataset/arch/interaction coefficient).",
    "Pr(>0)": "Posterior probability the term's effect is positive (on transformed scale).",
    "BF10_interpretation": "Qualitative interpretation of BF10 (H1: effect ≠ 0 vs H0: effect = 0).",
    "BF10_bw_scott": "Savage–Dickey BF10 using KDE bandwidth='scott'.",
    "BF10_bw_0.1": "Savage–Dickey BF10 using KDE bandwidth=0.1.",
    "BF10_bw_0.3": "Savage–Dickey BF10 using KDE bandwidth=0.3.",
    "BF10_bw_0.5": "Savage–Dickey BF10 using KDE bandwidth=0.5.",
    "BF10_bw_silverman": "Savage–Dickey BF10 using KDE bandwidth='silverman'.",
    "OR": "Odds ratio implied by the contrast on logit scale (exp(Δ_logit)).",
    "Δp_mean": "Mean difference in predicted metric on ORIGINAL scale for the contrast.",
    "Δp_lo": "Lower 95% HDI for Δp (original scale).",
    "Δp_hi": "Upper 95% HDI for Δp (original scale).",
    "baseline": "Baseline level used for the contrast (original scale; depends on term type).",
}

COLGUIDE_EFFECTS_LOG = {
    "param": "Model term (dataset/arch/interaction coefficient).",
    "Pr(>0)": "Posterior probability the term's effect is positive (on transformed scale).",
    "BF10_interpretation": "Qualitative interpretation of BF10 (H1: effect ≠ 0 vs H0: effect = 0).",
    "BF10_bw_scott": "Savage–Dickey BF10 using KDE bandwidth='scott'.",
    "BF10_bw_0.1": "Savage–Dickey BF10 using KDE bandwidth=0.1.",
    "BF10_bw_0.3": "Savage–Dickey BF10 using KDE bandwidth=0.3.",
    "BF10_bw_0.5": "Savage–Dickey BF10 using KDE bandwidth=0.5.",
    "BF10_bw_silverman": "Savage–Dickey BF10 using KDE bandwidth='silverman'.",
    "Abs_Δ_mean": "Mean absolute change on original scale for the contrast.",
    "Abs_Δ_lo": "Lower 95% HDI of absolute change.",
    "Abs_Δ_hi": "Upper 95% HDI of absolute change.",
    "%_change": "Mean percent change implied by the effect (computed on transformed scale).",
    "baseline": "Baseline level used for the contrast (original scale; depends on term type).",
}

# =============================================================================
# PREDICTION TABLES
# =============================================================================

COLGUIDE_PRED = {
    "dataset": "Dataset for this cell.",
    "arch": "Architecture for this cell.",
    "mean": "Posterior mean predicted performance (original scale).",
    "hdi_lo": "Lower 95% HDI for the predicted mean.",
    "hdi_hi": "Upper 95% HDI for the predicted mean.",
}

# =============================================================================
# CONTRAST TABLES
# =============================================================================

COLGUIDE_CONTRASTS_DATASET = {
    "dataset": "Dataset scope for this set of architecture comparisons.",
    "contrast": "Which architecture is compared against the baseline architecture.",
    "Δ_mean": "Mean difference on original scale.",
    "Δ_lo": "Lower 95% HDI for the difference.",
    "Δ_hi": "Upper 95% HDI for the difference.",
    "Pr(>0)": "Probability the difference is > 0 on transformed scale.",
    "Pr(|Δ|<ROPE)": "Probability the difference is practically negligible (ROPE) on transformed scale.",
}

COLGUIDE_CONTRASTS_ARCH = {
    "arch": "Architecture scope for this set of dataset comparisons.",
    "contrast": "Which dataset is compared against the baseline dataset.",
    "Δ_mean": "Mean difference on original scale.",
    "Δ_lo": "Lower 95% HDI for the difference.",
    "Δ_hi": "Upper 95% HDI for the difference.",
    "Pr(>0)": "Probability the difference is > 0 on transformed scale.",
    "Pr(|Δ|<ROPE)": "Probability the difference is practically negligible (ROPE) on transformed scale.",
}

COLGUIDE_CONTRASTS_OVERALL = {
    "scope": "What was averaged over to form the marginal comparison.",
    "contrast": "Comparison (A − baseline).",
    "mean_original": "Mean difference on original scale.",
    "hdi_lo_original": "Lower 95% HDI of difference.",
    "hdi_hi_original": "Upper 95% HDI of difference.",
    "Pr(>0)": "Probability the contrast > 0 (transformed scale).",
    "Pr(|Δ|<ROPE)": "Probability the effect is practically negligible (transformed scale ROPE).",
}

# =============================================================================
# RANKING TABLES
# =============================================================================

COLGUIDE_OVERALL_RANK = {
    "group": "Group name.",
    "E[rank]": "Expected rank under composite score (lower is better).",
    "Pr(best)": "Probability the group is best under composite score.",
}

for k in range(1, 10):
    COLGUIDE_OVERALL_RANK[f"Pr(rank={k})"] = f"Probability this group is rank {k} under composite score."

COLGUIDE_CONSECUTIVE_BF = {
    "rank": "Expected rank of this group.",
    "group": "Group name.",
    "next_rank_group": "Next lower-ranked group.",
    "Pr(better)": "Posterior probability this group is better than the next-ranked group.",
    "BF10_better": "Bayes factor for 'this group is better' vs 'no difference'.",
    "BF10_interpretation": "Qualitative interpretation of BF10.",
}