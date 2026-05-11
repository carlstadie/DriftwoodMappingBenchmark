# colguides.py
from __future__ import annotations

from typing import Dict
import numpy as np

# =============================================================================
# COLUMN GUIDES (for table explanations)
# =============================================================================

COLGUIDE_COUNTS = {
    "dataset": "Dataset label (AE/PS/S2).",
    "arch": "Architecture label (U-Net/Swin/Terramind).",
    "group": "Combined group name used in the group model.",
    "n_rows": "Number of runs/rows in this cell.",
}

COLGUIDE_PERF = {
    "group": "Group name (dataset×arch).",
    "mean": "Posterior mean of group mean on the ORIGINAL metric scale.",
    "sd": "Posterior SD of the group mean.",
    "hdi_2.5%": "Lower bound of the 95% HDI for the group mean.",
    "hdi_97.5%": "Upper bound of the 95% HDI for the group mean.",
    "sigma_mean": "Posterior mean of residual SD on the TRANSFORMED scale (group-specific). Higher = more variable runs.",
    "sigma_sd": "Posterior SD of that group-specific residual SD.",
    "sigma_hdi_lo": "Lower HDI bound of group residual SD (transformed scale).",
    "sigma_hdi_hi": "Upper HDI bound of group residual SD (transformed scale).",
    "E[rank]": "Expected rank (1=best).",
    "Pr(best)": "Posterior probability the group is the best.",
    "Pr(best)_prior": "Prior probability the group is best (before seeing data).",
    "Pr(Δ>ROPE)": "Probability this group is meaningfully better than best group (on transformed scale).",
    "Pr(|Δ|<ROPE)": "Probability this group is practically tied with best group (on transformed scale).",
    "Pr(Δ<-ROPE)": "Probability this group is meaningfully worse than best group (on transformed scale).",
}
for k in range(1, 10):
    COLGUIDE_PERF[f"Pr(rank={k})"] = f"Posterior probability this group is rank {k} (1=best)."

COLGUIDE_EFFECTS_LOGIT = {
    "param": "Model term (dataset/arch/interaction coefficient).",
    "Pr(Δ>ROPE)": "Probability the effect is meaningfully positive (Δ on transformed scale exceeds ROPE).",
    "Pr(|Δ|<ROPE)": "Probability the effect is practically negligible (inside ROPE; transformed scale).",
    "Pr(Δ<-ROPE)": "Probability the effect is meaningfully negative (Δ on transformed scale below -ROPE).",
    "Pr(>0)": "Probability the effect is positive (may include tiny, non-meaningful effects).",
    "OR": "Odds ratio implied by the effect on logit scale (exp(Δ_logit)).",
    "Δp_mean": "Mean difference on ORIGINAL scale implied by the effect.",
    "Δp_lo": "Lower 95% HDI for Δp (original scale).",
    "Δp_hi": "Upper 95% HDI for Δp (original scale).",
    "baseline": "Baseline level used for the contrast (original scale; depends on term type).",
}

COLGUIDE_EFFECTS_LOG = {
    "param": "Model term (dataset/arch/interaction coefficient).",
    "Pr(Δ>ROPE)": "Probability the effect is meaningfully positive (Δ on log scale exceeds ROPE).",
    "Pr(|Δ|<ROPE)": "Probability the effect is practically negligible (inside ROPE; log scale).",
    "Pr(Δ<-ROPE)": "Probability the effect is meaningfully negative (Δ below -ROPE on log scale).",
    "Pr(>0)": "Probability the effect is positive (may include tiny, non-meaningful effects).",
    "Abs_Δ_mean": "Mean absolute change on original scale implied by the effect.",
    "Abs_Δ_lo": "Lower 95% HDI of absolute change.",
    "Abs_Δ_hi": "Upper 95% HDI of absolute change.",
    "%_change": "Mean percent change implied by the effect (computed on transformed/log scale).",
    "baseline": "Baseline level used for the contrast (original scale; depends on term type).",
}

COLGUIDE_PRED = {
    "dataset": "Dataset for this cell.",
    "arch": "Architecture for this cell.",
    "mean": "Posterior mean predicted performance (original scale).",
    "hdi_lo": "Lower 95% HDI for the predicted mean.",
    "hdi_hi": "Upper 95% HDI for the predicted mean.",
}

COLGUIDE_CONTRASTS_DATASET = {
    "dataset": "Dataset scope for this set of architecture comparisons.",
    "contrast": "Which architecture is compared against the baseline architecture.",
    "Δ_mean": "Mean difference on original scale.",
    "Δ_lo": "Lower 95% HDI for the difference.",
    "Δ_hi": "Upper 95% HDI for the difference.",
    "Pr(Δ>ROPE)": "Probability the contrast is meaningfully positive (transformed scale ROPE).",
    "Pr(|Δ|<ROPE)": "Probability the contrast is practically negligible (inside ROPE; transformed scale).",
    "Pr(Δ<-ROPE)": "Probability the contrast is meaningfully negative (transformed scale ROPE).",
    "Pr(>0)": "Probability the contrast is positive (may include tiny effects).",
}

COLGUIDE_CONTRASTS_ARCH = {
    "arch": "Architecture scope for this set of dataset comparisons.",
    "contrast": "Which dataset is compared against the baseline dataset.",
    "Δ_mean": "Mean difference on original scale.",
    "Δ_lo": "Lower 95% HDI for the difference.",
    "Δ_hi": "Upper 95% HDI for the difference.",
    "Pr(Δ>ROPE)": "Probability the contrast is meaningfully positive (transformed scale ROPE).",
    "Pr(|Δ|<ROPE)": "Probability the contrast is practically negligible (inside ROPE; transformed scale).",
    "Pr(Δ<-ROPE)": "Probability the contrast is meaningfully negative (transformed scale ROPE).",
    "Pr(>0)": "Probability the contrast is positive (may include tiny effects).",
}

COLGUIDE_CONTRASTS_OVERALL = {
    "scope": "What was averaged over to form the marginal comparison.",
    "contrast": "Comparison (A − baseline).",
    "mean_original": "Mean difference on original scale.",
    "hdi_lo_original": "Lower 95% HDI of difference.",
    "hdi_hi_original": "Upper 95% HDI of difference.",
    "Pr(Δ>ROPE)": "Probability the contrast is meaningfully positive (transformed scale ROPE).",
    "Pr(|Δ|<ROPE)": "Probability the contrast is practically negligible (inside ROPE; transformed scale).",
    "Pr(Δ<-ROPE)": "Probability the contrast is meaningfully negative (transformed scale ROPE).",
    "Pr(>0)": "Probability the contrast is positive (may include tiny effects).",
}

COLGUIDE_OVERALL_RANK = {
    "group": "Group name (dataset×arch).",
    "E[rank]": "Expected rank (1=best) across all metrics (weighted).",
    "Pr(best)": "Posterior probability the group is the overall best.",
    "Pr(top2)": "Posterior probability the group is in top 2.",
    "rank_ci90": "90% credible interval for rank [q05, q95].",
    "entropy_rank": "Entropy of rank distribution (higher = more uncertain).",
    "N_eff_winners": "Effective number of winners (1/sum(Pr(best)^2)); higher = more competitive field.",
    "Pr(Δ>ROPE)": "Probability this group is meaningfully better than best group (on composite scale).",
    "Pr(|Δ|<ROPE)": "Probability this group is practically tied with best group (on composite scale).",
    "Pr(Δ<-ROPE)": "Probability this group is meaningfully worse than best group (on composite scale).",
    "runner_up": "Name of the runner-up group (only shown for top-ranked group).",
    "Pr(win_vs_runnerup)": "Probability top group beats runner-up (only for top group).",
    "Pr(tie_vs_runnerup)": "Probability of tie/draw vs runner-up (|delta| < tie_eps; only for top group).",
    "draw_index": "Draw index = 1 - 2*|Pr(win)-0.5|, range [0,1]; 1=decisive, 0=coin flip (only for top group).",
}
for k in range(1, 10):
    COLGUIDE_OVERALL_RANK[f"Pr(rank={k})"] = f"Probability this group is rank {k} under composite score."

# For consecutive ranking ROPE table (used by models_group.py)
COLGUIDE_CONSECUTIVE_ROPE = {
    "rank": "Expected rank of this group.",
    "group": "Group name.",
    "next_rank_group": "Next lower-ranked group.",
    "Δ_mean": "Mean difference (original scale), oriented so positive means the higher-ranked group is better.",
    "Δ_lo": "Lower 95% HDI of the difference (original scale).",
    "Δ_hi": "Upper 95% HDI of the difference (original scale).",
    "Pr(better)": "Probability the higher-ranked group is better (Δ > 0 on transformed scale).",
    "Pr(Δ>ROPE)": "Probability the higher-ranked group is meaningfully better (Δ > ROPE on transformed scale).",
    "Pr(|Δ|<ROPE)": "Probability the two groups are practically tied (|Δ| < ROPE on transformed scale).",
    "Pr(Δ<-ROPE)": "Probability the ranking is meaningfully reversed (Δ < -ROPE on transformed scale).",
}