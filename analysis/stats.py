import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, List

import arviz as az
import matplotlib

# Use a non-interactive backend so the script can run headless (no GUI required).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pymc as pm  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

print("Running with PyMC version:", pm.__version__)

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# paths
UNET_AE_METRICS = (
    r"N:\isipd\projects\p_planetdw\data\methods_test\results\UNET\AE"
    r"\evaluation_unet.csv"
)
SWIN_AE_METRICS = (
    r"N:\isipd\projects\p_planetdw\data\methods_test\results\SWIN\AE"
    r"\evaluation_swin.csv"
)
TERRAMIND_AE_METRICS = (
    r"N:\isipd\projects\p_planetdw\data\methods_test\results\Terramind\AE"
    r"\evaluation_tm.csv"
)

UNET_PS_METRICS = (
    r"N:\isipd\projects\p_planetdw\data\methods_test\results\UNET\PS"
    r"\evaluation_unet.csv"
)
SWIN_PS_METRICS = (
    r"N:\isipd\projects\p_planetdw\data\methods_test\results\SWIN\PS"
    r"\evaluation_swin.csv"
)
TERRAMIND_PS_METRICS = (
    r"N:\isipd\projects\p_planetdw\data\methods_test\results\Terramind\PS"
    r"\evaluation_tm.csv"
)

UNET_S2_METRICS = (
    r"N:\isipd\projects\p_planetdw\data\methods_test\results\UNET\S2"
    r"\evaluation_unet.csv"
)
SWIN_S2_METRICS = (
    r"N:\isipd\projects\p_planetdw\data\methods_test\results\SWIN\S2"
    r"\evaluation_swin.csv"
)
TERRAMIND_S2_METRICS = (
    r"N:\isipd\projects\p_planetdw\data\methods_test\results\Terramind\S2"
    r"\evaluation_tm.csv"
)

# metrics to analyse
METRICS = [
    "IoU",
    "dice_coef",
    "normalized_surface_distance",
    "mean_epistemic_uncertainty",
    "mean_aleatoric_uncertainty",
]

# which metrics are "higher is better"
MAXIMIZE_METRICS = {"IoU", "dice_coef"}

# metrics that are bounded in (0,1) and should be modelled on logit scale
BOUNDED_01_METRICS = {
    "IoU",
    "dice_coef",
    "normalized_surface_distance",
    "mean_epistemic_uncertainty",
    "mean_aleatoric_uncertainty",
}

# priors for factorial effects (on transformed scale)
PRIOR_SD_LOGIT = 0.5
PRIOR_SD_LOG = 0.2

TEST = True

# sampling config
DRAWS = 2000
TUNE = 2000
CHAINS = 4
TARGET_ACCEPT = 0.9
HDI_PROB = 0.95
RANDOM_SEED = 1701

# Savage–Dickey KDE bandwidths for sensitivity analysis
SD_KDE_BANDWIDTHS = [0.1, 0.3, 0.5, "scott", "silverman"]

# overall ranking weights across metrics
METRIC_WEIGHTS: Optional[Dict[str, float]] = None

# Stable ordering
DATASET_ORDER = ["AE", "PS", "S2"]
ARCH_ORDER = ["U-Net", "Swin", "Terramind"]

# Contrast baselines
BASELINE_ARCH = "U-Net"
BASELINE_DATASET = "AE"  # for dataset-within-architecture contrasts

# Baselines for interpreting factorial coefficients under EFFECTS CODING
EFFECTS_BASELINE_DATASET = DATASET_ORDER[0]  # e.g., "AE"
EFFECTS_BASELINE_ARCH = ARCH_ORDER[0]        # e.g., "U-Net"

# Practical effect thresholds (ROPE) on TRANSFORMED scale for consistency
ROPE_BY_METRIC_TRANSFORMED: Dict[str, float] = {
    "IoU": 0.05,  # on logit scale
    "dice_coef": 0.05,
    "normalized_surface_distance": 0.05,
    "mean_epistemic_uncertainty": 0.05,
    "mean_aleatoric_uncertainty": 0.05,
}

CSV_SPECS = [
    (UNET_AE_METRICS, "U-Net | AE", "AE", "U-Net"),
    (SWIN_AE_METRICS, "Swin U-Net | AE", "AE", "Swin"),
    (TERRAMIND_AE_METRICS, "Terramind | AE", "AE", "Terramind"),
    (UNET_PS_METRICS, "U-Net | PS", "PS", "U-Net"),
    (SWIN_PS_METRICS, "Swin U-Net | PS", "PS", "Swin"),
    (TERRAMIND_PS_METRICS, "Terramind | PS", "PS", "Terramind"),
    (UNET_S2_METRICS, "U-Net | S2", "S2", "U-Net"),
    (SWIN_S2_METRICS, "Swin U-Net | S2", "S2", "Swin"),
    (TERRAMIND_S2_METRICS, "Terramind | S2", "S2", "Terramind"),
]

GROUP_ORDER = [
    "U-Net | AE",
    "Swin U-Net | AE",
    "Terramind | AE",
    "U-Net | PS",
    "Swin U-Net | PS",
    "Terramind | PS",
    "U-Net | S2",
    "Swin U-Net | S2",
    "Terramind | S2",
]

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
    "E[rank]": "Expected rank (1=best).",
    "Pr(best)": "Posterior probability the group is the best.",
    "Pr(best)_prior": "Prior probability the group is best (before seeing data).",
    "BF_best": "Bayes factor for 'this group is best' vs 'not best' (via odds update).",
}
for k in range(1, 10):
    COLGUIDE_PERF[f"Pr(rank={k})"] = f"Posterior probability this group is rank {k} (1=best)."

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

COLGUIDE_OVERALL_RANK = {
    "group": "Group name.",
    "E[rank]": "Expected rank under composite score (lower is better).",
    "Pr(best)": "Probability the group is best under composite score.",
}
for k in range(1, 10):
    COLGUIDE_OVERALL_RANK[f"Pr(rank={k})"] = f"Probability this group is rank {k} under composite score."

# =============================================================================
# HTML REPORT HELPERS
# =============================================================================


def _safe_slug(text: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "item"


def _df_to_html_table(df: pd.DataFrame) -> str:
    fmt_df = df.copy()
    
    def format_value(x):
        if isinstance(x, float):
            if np.isinf(x):
                return "∞" if x > 0 else "-∞"
            return f"{x:.4f}"
        return str(x)
    
    return fmt_df.to_html(
        index=False,
        escape=True,
        classes="tbl",
        border=0,
        formatters={col: format_value for col in fmt_df.columns},
    )


class HtmlReport:
    def __init__(self, report_path: Path, title: str):
        self.report_path = report_path
        self._f = open(report_path, "w", encoding="utf-8")
        self._write_header(title)

    def _write_header(self, title: str) -> None:
        self._f.write(
            f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --bg: #0b1020;
  --card: #121a33;
  --text: #e7ecff;
  --muted: #aab5e6;
  --border: rgba(255,255,255,0.10);
  --accent: #88a6ff;
  --accent2: #b68cff;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  background: linear-gradient(180deg, var(--bg), #070a14);
  color: var(--text);
}}
.container {{
    max-width: max-content;
    min-width: 800px;
  margin: 0 auto;
  padding: 22px 16px 80px;
}}
h1 {{
  font-size: 26px;
  margin: 0 0 10px;
  letter-spacing: .2px;
}}
h2 {{ font-size: 20px; margin: 18px 0 10px; }}
h3 {{ font-size: 15px; margin: 16px 0 10px; color: var(--text); }}
p {{ line-height: 1.5; color: var(--muted); margin: 8px 0; }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

.topbar {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
  margin-bottom: 12px;
}}
.badge {{
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--border);
  color: var(--muted);
  font-size: 12px;
}}

.tabbar {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 12px 0 14px;
}}
.tabbtn {{
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 8px 10px;
  border-radius: 999px;
  cursor: pointer;
  font-size: 13px;
}}
.tabbtn:hover {{ border-color: rgba(255,255,255,0.18); }}
.tabbtn.active {{
  background: rgba(136,166,255,0.20);
  border-color: rgba(136,166,255,0.45);
}}
.tabcontent {{
  display: none;
}}
.tabcontent.active {{
  display: block;
}}

.card {{
  background: rgba(18, 26, 51, 0.92);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,.25);
  margin: 12px 0;
}}
.grid {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 12px;
}}
@media(min-width: 980px) {{
  .grid-2 {{
    display: grid;
    grid-template-columns: 1.1fr 0.9fr;
    gap: 12px;
    align-items: start;
  }}
}}

img.figure {{
  width: 100%;
  height: auto;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.02);
}}

table.tbl {{
  width: 100%;
  border-collapse: collapse;
  overflow: hidden;
  border-radius: 12px;
  border: 1px solid var(--border);
}}
.tbl th, .tbl td {{
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
  vertical-align: top;
}}
.tbl th {{
  text-align: left;
  font-weight: 600;
  color: var(--text);
  background: rgba(255,255,255,0.04);
}}
.tbl tr:nth-child(even) td {{
  background: rgba(255,255,255,0.02);
}}

details.footnote {{
  margin-top: 10px;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.03);
}}
details.footnote summary {{
  cursor: pointer;
  color: var(--text);
  font-size: 13px;
}}
.small {{
  font-size: 12px;
  color: var(--muted);
}}
.footer {{
  margin-top: 18px;
  font-size: 12px;
  color: var(--muted);
}}
</style>
</head>
<body>
<div class="container">
"""
        )

        self._f.write(f"<h1>{title}</h1>\n")

    def add_html(self, html: str) -> None:
        self._f.write(html + "\n")

    def add_paragraph(self, text: str) -> None:
        self._f.write(f"<p>{text}</p>\n")

    def add_h2(self, text: str) -> None:
        self._f.write(f"<h2>{text}</h2>\n")

    def add_h3(self, text: str) -> None:
        self._f.write(f"<h3>{text}</h3>\n")

    def add_card_start(self) -> None:
        self._f.write('<div class="card">\n')

    def add_card_end(self) -> None:
        self._f.write("</div>\n")

    def add_image(self, rel_path: str, alt: str) -> None:
        self._f.write(f'<img class="figure" src="{rel_path}" alt="{alt}">\n')

    def start_tabs(self, tabs: Sequence[Tuple[str, str]], default_tab_id: str) -> None:
        btns = []
        for tab_id, label in tabs:
            btns.append(f'<button class="tabbtn" data-tab="{tab_id}">{label}</button>')
        self._f.write('<div class="tabbar">\n' + "\n".join(btns) + "\n</div>\n")
        self._f.write(
            f"""
<script>
function openTab(tabId) {{
  const tabs = document.querySelectorAll('.tabcontent');
  const btns = document.querySelectorAll('.tabbtn');
  tabs.forEach(t => t.classList.remove('active'));
  btns.forEach(b => b.classList.remove('active'));
  const t = document.getElementById(tabId);
  if (t) t.classList.add('active');
  const b = document.querySelector(`.tabbtn[data-tab="${{tabId}}"]`);
  if (b) b.classList.add('active');
}}
document.addEventListener('click', (e) => {{
  const btn = e.target.closest('.tabbtn');
  if (!btn) return;
  openTab(btn.getAttribute('data-tab'));
}});
document.addEventListener('DOMContentLoaded', () => {{
    // Adjust container width based on widest table
    const tables = document.querySelectorAll('table.tbl');
    let maxWidth = 800;
    tables.forEach(tbl => {{
        const rect = tbl.getBoundingClientRect();
        if (rect.width > maxWidth) {{
            maxWidth = rect.width;
        }}
    }});
    const container = document.querySelector('.container');
    if (container && maxWidth > 800) {{
        container.style.maxWidth = Math.min(maxWidth + 64, window.innerWidth - 32) + 'px';
    }}
    // Open default tab
  let tab = "{default_tab_id}";
  openTab(tab);
}});
</script>
"""
        )

    def open_tab_content(self, tab_id: str) -> None:
        self._f.write(f'<div class="tabcontent" id="{tab_id}">\n')

    def close_tab_content(self) -> None:
        self._f.write("</div>\n")

    def add_footnote(self, title: str, html_body: str) -> None:
        self._f.write(
            f"""
<details class="footnote">
  <summary>{title}</summary>
  <div class="small">{html_body}</div>
</details>
"""
        )

    def close(self) -> None:
        self._f.write(
            f"""
<div class="footer">
Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</div>
</div>
</body>
</html>
"""
        )
        self._f.close()


def add_table_with_column_guide(
    report: HtmlReport,
    df: pd.DataFrame,
    col_guide: Dict[str, str],
    title: str = "Column guide",
) -> None:
    """
    Writes an HTML table and then a small column-name explanation beneath it.
    Only columns that exist in df are shown.
    """
    report.add_html(_df_to_html_table(df))

    items = []
    for col, desc in col_guide.items():
        if col in df.columns:
            items.append(f"<li><b>{col}</b>: {desc}</li>")

    if not items:
        return

    report.add_html(
        f"""
<div class="small" style="margin-top:8px">
  <b>{title}:</b>
  <ul style="margin:6px 0 0 18px">
    {''.join(items)}
  </ul>
</div>
"""
    )


# =============================================================================
# CORE HELPERS
# =============================================================================


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _summarize_draws(draws: np.ndarray, hdi_prob: float) -> Dict[str, float]:
    draws = np.asarray(draws).ravel()
    lo_q = (1.0 - hdi_prob) / 2.0
    hi_q = 1.0 - lo_q
    return {
        "mean": float(np.mean(draws)),
        "hdi_lo": float(np.quantile(draws, lo_q)),
        "hdi_hi": float(np.quantile(draws, hi_q)),
    }


def _density_at_zero(draws_1d: np.ndarray, bw) -> float:
    draws_1d = np.asarray(draws_1d).ravel()
    kde = gaussian_kde(draws_1d, bw_method=bw)
    return float(kde.evaluate(0.0)[0])


def _bf10_from_prior_posterior(
    prior_draws: np.ndarray,
    post_draws: np.ndarray,
    bw,
) -> float:
    prior0 = _density_at_zero(prior_draws, bw=bw)
    post0 = _density_at_zero(post_draws, bw=bw)
    if post0 == 0 or post0 < 1e-10:
        return np.inf
    bf10 = float(prior0 / post0)
    # Cap overly large BFs to inf for display
    if bf10 > 1e6:
        return np.inf
    return bf10


def interpret_bf10(bf10: float) -> str:
    if bf10 < 1 / 10:
        return "strong evidence for H0 (no effect)"
    if bf10 < 1 / 3:
        return "moderate evidence for H0"
    if bf10 < 1:
        return "anecdotal evidence for H0"
    if bf10 < 3:
        return "anecdotal evidence for H1 (effect)"
    if bf10 < 10:
        return "moderate evidence for H1"
    if bf10 < 30:
        return "strong evidence for H1"
    if bf10 < 100:
        return "very strong evidence for H1"
    return "extreme evidence for H1"


def _odds(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return p / (1.0 - p)


def _pick_hdi_columns(summary_df: pd.DataFrame) -> Tuple[str, str]:
    hdi_cols = [
        c for c in summary_df.columns if c.startswith("hdi_") and c.endswith("%")
    ]
    if len(hdi_cols) < 2:
        raise ValueError(f"Could not find HDI columns in: {list(summary_df.columns)}")

    def _as_float(col: str) -> float:
        return float(col.replace("hdi_", "").replace("%", ""))

    hdi_cols_sorted = sorted(hdi_cols, key=_as_float)
    return hdi_cols_sorted[0], hdi_cols_sorted[-1]


def _as_dataarray(extracted, var_name: str):
    if hasattr(extracted, "data_vars") and var_name in extracted.data_vars:
        return extracted[var_name]
    return extracted


def _extract_draws_1d(
    idata: az.InferenceData,
    group: str,
    var_name: str,
    sel: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """
    Robust draw extractor: stacks (chain, draw) -> sample and preserves coord order.
    Works for posterior and prior groups.
    """
    grp = getattr(idata, group)
    da = grp[var_name]
    if sel:
        da = da.sel(sel)
    if "chain" in da.dims and "draw" in da.dims:
        da = da.stack(sample=("chain", "draw"))
        return da.to_numpy().ravel()
    return da.to_numpy().ravel()


# =============================================================================
# CONTRASTS
# =============================================================================


def summarize_contrast(
    delta_transformed: np.ndarray,
    delta_original: np.ndarray,
    rope_transformed: float,
    hdi_prob: float,
) -> Dict[str, float]:
    """
    Summarize contrasts on both transformed and original scales.
    ROPE is applied on transformed scale for consistency with model.
    """
    s_t = _summarize_draws(delta_transformed, hdi_prob=hdi_prob)
    s_o = _summarize_draws(delta_original, hdi_prob=hdi_prob)

    return {
        "mean_transformed": s_t["mean"],
        "hdi_lo_transformed": s_t["hdi_lo"],
        "hdi_hi_transformed": s_t["hdi_hi"],
        "mean_original": s_o["mean"],
        "hdi_lo_original": s_o["hdi_lo"],
        "hdi_hi_original": s_o["hdi_hi"],
        "Pr(>0)": float((delta_transformed > 0).mean()),
        "Pr(<0)": float((delta_transformed < 0).mean()),
        "Pr(|Δ|<ROPE)": float((np.abs(delta_transformed) < rope_transformed).mean()),
    }


def contrasts_within_each_dataset(
    pred_draws_transformed: np.ndarray,
    pred_draws_original: np.ndarray,
    datasets: Sequence[str],
    archs: Sequence[str],
    baseline_arch: str,
    rope_transformed: float,
    hdi_prob: float,
) -> pd.DataFrame:
    if baseline_arch not in archs:
        baseline_arch = archs[0]

    a0 = archs.index(baseline_arch)
    rows = []
    for di, d in enumerate(datasets):
        base_t = pred_draws_transformed[:, di, a0]
        base_o = pred_draws_original[:, di, a0]
        for ai, a in enumerate(archs):
            if a == baseline_arch:
                continue
            delta_t = pred_draws_transformed[:, di, ai] - base_t
            delta_o = pred_draws_original[:, di, ai] - base_o
            stats = summarize_contrast(delta_t, delta_o, rope_transformed, hdi_prob)
            rows.append({"dataset": d, "contrast": f"{a} − {baseline_arch}", **stats})
    return pd.DataFrame(rows)


def contrasts_within_each_architecture(
    pred_draws_transformed: np.ndarray,
    pred_draws_original: np.ndarray,
    datasets: Sequence[str],
    archs: Sequence[str],
    baseline_dataset: str,
    rope_transformed: float,
    hdi_prob: float,
) -> pd.DataFrame:
    """
    Dataset contrasts within each architecture (mirror of contrasts_within_each_dataset).
    """
    if baseline_dataset not in datasets:
        baseline_dataset = datasets[0]

    d0 = datasets.index(baseline_dataset)
    rows = []
    for ai, a in enumerate(archs):
        base_t = pred_draws_transformed[:, d0, ai]
        base_o = pred_draws_original[:, d0, ai]
        for di, d in enumerate(datasets):
            if d == baseline_dataset:
                continue
            delta_t = pred_draws_transformed[:, di, ai] - base_t
            delta_o = pred_draws_original[:, di, ai] - base_o
            stats = summarize_contrast(delta_t, delta_o, rope_transformed, hdi_prob)
            rows.append({"arch": a, "contrast": f"{d} − {baseline_dataset}", **stats})
    return pd.DataFrame(rows)


def contrasts_arch_overall(
    pred_draws_transformed: np.ndarray,
    pred_draws_original: np.ndarray,
    archs: Sequence[str],
    baseline_arch: str,
    rope_transformed: float,
    hdi_prob: float,
) -> pd.DataFrame:
    """
    Overall architecture contrasts (marginalized over datasets).
    """
    if baseline_arch not in archs:
        baseline_arch = archs[0]
    a0 = archs.index(baseline_arch)

    marg_t = pred_draws_transformed.mean(axis=1)  # (samples, arch)
    marg_o = pred_draws_original.mean(axis=1)

    base_t = marg_t[:, a0]
    base_o = marg_o[:, a0]

    rows = []
    for ai, a in enumerate(archs):
        if a == baseline_arch:
            continue
        delta_t = marg_t[:, ai] - base_t
        delta_o = marg_o[:, ai] - base_o
        stats = summarize_contrast(delta_t, delta_o, rope_transformed, hdi_prob)
        rows.append({"scope": "Overall (avg over datasets)", "contrast": f"{a} − {baseline_arch}", **stats})
    return pd.DataFrame(rows)


def contrasts_dataset_overall(
    pred_draws_transformed: np.ndarray,
    pred_draws_original: np.ndarray,
    datasets: Sequence[str],
    baseline_dataset: str,
    rope_transformed: float,
    hdi_prob: float,
) -> pd.DataFrame:
    """
    Overall dataset contrasts (marginalized over architectures).
    """
    if baseline_dataset not in datasets:
        baseline_dataset = datasets[0]
    d0 = datasets.index(baseline_dataset)

    marg_t = pred_draws_transformed.mean(axis=2)  # (samples, dataset)
    marg_o = pred_draws_original.mean(axis=2)

    base_t = marg_t[:, d0]
    base_o = marg_o[:, d0]

    rows = []
    for di, d in enumerate(datasets):
        if d == baseline_dataset:
            continue
        delta_t = marg_t[:, di] - base_t
        delta_o = marg_o[:, di] - base_o
        stats = summarize_contrast(delta_t, delta_o, rope_transformed, hdi_prob)
        rows.append({"scope": "Overall (avg over arch)", "contrast": f"{d} − {baseline_dataset}", **stats})
    return pd.DataFrame(rows)


def plot_contrasts(
    contrast_df: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    if contrast_df.empty:
        return

    df = contrast_df.copy()
    df = df.sort_values(["dataset", "mean_original"], ascending=[True, False]).reset_index(drop=True) if "dataset" in df.columns else df

    labels = []
    if "dataset" in df.columns:
        labels = [f"{r.dataset}: {r.contrast}" for r in df.itertuples(index=False)]
    elif "arch" in df.columns:
        labels = [f"{r.arch}: {r.contrast}" for r in df.itertuples(index=False)]
    else:
        labels = [f"{r.contrast}" for r in df.itertuples(index=False)]

    y = np.arange(len(df))
    x = df["mean_original"].to_numpy()
    lo = df["hdi_lo_original"].to_numpy()
    hi = df["hdi_hi_original"].to_numpy()
    xerr_lo = x - lo
    xerr_hi = hi - x

    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.38 * len(df))))
    ax.errorbar(x, y, xerr=[xerr_lo, xerr_hi], fmt="o", capsize=3)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Δ({metric}) on original scale")
    ax.set_title(f"Contrasts: {metric}")
    ax.grid(True, axis="x", alpha=0.2)

    _save_fig(fig, out_path)


# =============================================================================
# DATA + TRANSFORMS
# =============================================================================


def load_checkpoint_metrics(csv_specs, metrics_):
    dfs = []
    for path, group, dataset, arch in csv_specs:
        if os.path.exists(path) is False:
            print(f"Warning: path {path} does not exist, skipping.")
            continue
        df = pd.read_csv(path)
        needed = [m for m in metrics_ if m in df.columns]
        subset = df[needed].copy()
        subset["group"] = group
        subset["dataset"] = dataset
        subset["arch"] = arch
        dfs.append(subset)
    return pd.concat(dfs, ignore_index=True)


def base_metric_name(metric_col: str) -> str:
    return metric_col[4:] if metric_col.startswith("val_") else metric_col


def is_bounded_01(metric_col: str) -> bool:
    return base_metric_name(metric_col) in BOUNDED_01_METRICS


def higher_is_better(metric_col: str) -> bool:
    m = base_metric_name(metric_col)
    return m in MAXIMIZE_METRICS


def _inv_logit(x):
    if hasattr(x, "type"):
        return pm.math.sigmoid(x)
    return 1.0 / (1.0 + np.exp(-x))


def _inv_exp(x):
    if hasattr(x, "type"):
        return pm.math.exp(x)
    return np.exp(x)


def transform_y(
    y: np.ndarray,
    metric_col: str,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Callable, str]:
    if is_bounded_01(metric_col):
        y_clip = np.clip(y, eps, 1 - eps)
        y_t = np.log(y_clip / (1 - y_clip))
        return y_t, _inv_logit, "logit"

    y_clip = np.clip(y, eps, None)
    y_t = np.log(y_clip)
    return y_t, _inv_exp, "log"


def rank_probabilities_from_draws(
    draws: np.ndarray,
    group_names,
    higher_better: bool,
) -> pd.DataFrame:
    s_count, g_count = draws.shape
    score = draws if higher_better else -draws

    order = np.argsort(-score, axis=1)
    ranks = np.empty_like(order)
    for s in range(s_count):
        ranks[s, order[s]] = np.arange(1, g_count + 1)

    out = {"group": list(group_names)}
    for k in range(1, g_count + 1):
        out[f"Pr(rank={k})"] = [(ranks[:, j] == k).mean() for j in range(g_count)]
    out["E[rank]"] = [ranks[:, j].mean() for j in range(g_count)]
    out["Pr(best)"] = out["Pr(rank=1)"]

    return pd.DataFrame(out).sort_values("E[rank]").reset_index(drop=True)


# =============================================================================
# MODELS WITH HIERARCHICAL VARIANCE
# =============================================================================


def fit_group_model_with_ranking(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    draws=DRAWS,
    tune=TUNE,
    chains=CHAINS,
    target_accept=TARGET_ACCEPT,
    seed=RANDOM_SEED,
):
    """
    Hierarchical group model WITH hierarchical variance structure.
    Even with equal n=10, this helps with regularization and is best practice.
    """
    data = df[[group_col, metric_col]].dropna().copy()

    if pd.api.types.is_categorical_dtype(data[group_col]):
        data[group_col] = data[group_col].cat.remove_unused_categories()

    gcat = pd.Categorical(data[group_col])
    g_idx = gcat.codes
    groups = list(gcat.categories)
    y_raw = data[metric_col].to_numpy()
    y_t, inv, tname = transform_y(y_raw, metric_col)
    coords = {"group": groups}

    with pm.Model(coords=coords) as model:
        g = pm.Data("g", g_idx)
        y_obs = pm.Data("y_obs", y_t)

        # Hierarchical mean structure
        mu0 = pm.Normal("mu0", 0.0, 1.5)
        tau = pm.HalfNormal("tau", 1.0)
        mu = pm.Normal("mu", mu0, tau, dims="group")

        # Hierarchical variance structure (even with n=10, this helps regularization)
        sigma_mu = pm.HalfNormal("sigma_mu", sigma=1.0)
        sigma_tau = pm.HalfNormal("sigma_tau", sigma=0.5)
        sigma_offset = pm.HalfNormal("sigma_offset", sigma=1.0, dims="group")
        sigma = pm.Deterministic("sigma", sigma_mu + sigma_tau * sigma_offset, dims="group")

        # Robust likelihood
        nu = pm.Exponential("nu", 1 / 30) + 1
        pm.StudentT("y", nu=nu, mu=mu[g], sigma=sigma[g], observed=y_obs)

        # Deterministic: group means on original scale
        mu_orig = pm.Deterministic("mu_orig", inv(mu), dims="group")

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            return_inferencedata=True,
        )

        # Prior predictive for BF
        prior = pm.sample_prior_predictive(var_names=["mu_orig"], random_seed=seed)

        # Posterior predictive check
        post_pred = pm.sample_posterior_predictive(idata, random_seed=seed)

    # Extract posterior draws
    post_mu_draws = (
        idata.posterior["mu_orig"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "group")
        .values
    )
    hib = higher_is_better(metric_col)
    post_rank_table = rank_probabilities_from_draws(
        post_mu_draws, groups, higher_better=hib
    )

    prior_mu_draws = (
        prior.prior["mu_orig"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "group")
        .values
    )
    prior_rank_table = rank_probabilities_from_draws(
        prior_mu_draws, groups, higher_better=hib
    )

    summ = az.summary(idata, var_names=["mu_orig"], hdi_prob=HDI_PROB).reset_index()
    summ = summ.rename(columns={"index": "param"})
    hdi_lo_col, hdi_hi_col = _pick_hdi_columns(summ)

    summ["group"] = summ["param"].str.replace(r"mu_orig\[|\]", "", regex=True)

    perf_table = (
        summ.loc[:, ["group", "mean", "sd", hdi_lo_col, hdi_hi_col]]
        .merge(post_rank_table, on="group", how="left")
        .merge(
            prior_rank_table.loc[:, ["group", "Pr(best)"]].rename(
                columns={"Pr(best)": "Pr(best)_prior"}
            ),
            on="group",
            how="left",
        )
        .sort_values("E[rank]")
        .reset_index(drop=True)
    )

    post_p = perf_table["Pr(best)"].to_numpy()
    prior_p = perf_table["Pr(best)_prior"].to_numpy()
    bf_best = _odds(post_p) / _odds(prior_p)
    perf_table["BF_best"] = bf_best

    return idata, perf_table, post_rank_table, post_mu_draws, tname, groups, post_pred


# =============================================================================
# CONSECUTIVE RANKING BF10 ANALYSIS
# =============================================================================

def compute_pairwise_bf10_consecutive(
    mu_draws: np.ndarray,
    rank_table: pd.DataFrame,
    group_names: Sequence[str],
    higher_better: bool,
    bw="scott",
) -> pd.DataFrame:
    """
    Compute BF10 for each group being better than the next lower-ranked group.
    
    Parameters:
    -----------
    mu_draws : np.ndarray
        Posterior draws of group means, shape (samples, groups)
    rank_table : pd.DataFrame
        Ranking table with 'group' and 'E[rank]' columns
    group_names : Sequence[str]
        Names of groups in same order as mu_draws columns
    higher_better : bool
        Whether higher values are better for this metric
    bw : str or float
        KDE bandwidth for Savage-Dickey BF
    
    Returns:
    --------
    pd.DataFrame with columns: rank, group, next_group, BF10_better, interpretation
    """
    # Sort by expected rank
    sorted_ranks = rank_table.sort_values("E[rank]").reset_index(drop=True)
    
    rows = []
    for i in range(len(sorted_ranks) - 1):
        group_i = sorted_ranks.loc[i, "group"]
        group_j = sorted_ranks.loc[i + 1, "group"]
        rank_i = i + 1  # assign next free rank based on sorted expected ranks
        
        # Get column indices
        idx_i = list(group_names).index(group_i)
        idx_j = list(group_names).index(group_j)
        
        # Compute difference (higher rank should be better)
        if higher_better:
            delta = mu_draws[:, idx_i] - mu_draws[:, idx_j]  # positive = i is better
        else:
            delta = mu_draws[:, idx_j] - mu_draws[:, idx_i]  # positive = i is better (lower values)
        
        # Prior: delta ~ Normal(0, something) centered at 0
        # For BF10 testing "i is better than j" vs "no difference"
        # We use Savage-Dickey: density at 0 under prior / density at 0 under posterior
        
        # Prior draws for delta (assuming independence, approximate as difference of priors)
        # This is a simplification - ideally we'd sample from the actual prior
        prior_delta = np.random.normal(0, mu_draws.std(), len(delta))
        
        bf10 = _bf10_from_prior_posterior(prior_delta, delta, bw=bw)
        
        # Also compute Pr(i > j)
        pr_better = float((delta > 0).mean())
        
        rows.append({
            "rank": rank_i,
            "group": group_i,
            "next_rank_group": group_j,
            "Pr(better)": pr_better,
            "BF10_better": bf10,
            "BF10_interpretation": interpret_bf10(bf10),
        })
    
    return pd.DataFrame(rows)


# Add this column guide for the new table
COLGUIDE_CONSECUTIVE_BF = {
    "rank": "Expected rank of this group.",
    "group": "Group name.",
    "next_rank_group": "Next lower-ranked group.",
    "Pr(better)": "Posterior probability this group is better than the next-ranked group.",
    "BF10_better": "Bayes factor for 'this group is better' vs 'no difference'.",
    "BF10_interpretation": "Qualitative interpretation of BF10.",
}


def add_consecutive_bf_analysis(
    report: HtmlReport,
    mu_draws: np.ndarray,
    perf_table: pd.DataFrame,
    group_names: Sequence[str],
    metric: str,
    higher_better: bool,
    assets_dir: Path,
):
    """Add consecutive ranking BF10 analysis to report."""
    
    # Compute BF10s with multiple bandwidths
    bf_results = []
    for bw in SD_KDE_BANDWIDTHS:
        bf_df = compute_pairwise_bf10_consecutive(
            mu_draws=mu_draws,
            rank_table=perf_table,
            group_names=group_names,
            higher_better=higher_better,
            bw=bw,
        )
        bw_str = bw if isinstance(bw, str) else f"{bw:.1f}"
        bf_df[f"BF10_bw_{bw_str}"] = bf_df["BF10_better"]
        bf_results.append(bf_df)
    
    # Merge all bandwidth results
    main_df = bf_results[0][["rank", "group", "next_rank_group", "Pr(better)"]].copy()
    for bf_df in bf_results:
        bw_col = [c for c in bf_df.columns if c.startswith("BF10_bw_")][0]
        main_df[bw_col] = bf_df[bw_col]
    
    # Use scott as default for interpretation
    main_df["BF10_interpretation"] = main_df["BF10_bw_scott"].apply(interpret_bf10)
    
    # Add to report
    report.add_h3("Consecutive ranking comparisons")
    report.add_paragraph(
        "Bayes factors for each group being better than the next lower-ranked group. "
        "This shows the strength of evidence for the ranking order."
    )
    
    add_table_with_column_guide(report, main_df, COLGUIDE_CONSECUTIVE_BF)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(main_df))))
    
    y = np.arange(len(main_df))
    x = main_df["Pr(better)"].to_numpy()
    
    colors = ["green" if p > 0.95 else "orange" if p > 0.80 else "red" for p in x]
    
    ax.barh(y, x, color=colors, alpha=0.6)
    ax.axvline(0.5, linestyle="--", color="gray", linewidth=1, alpha=0.5)
    ax.axvline(0.95, linestyle="--", color="green", linewidth=1, alpha=0.3, label="95% threshold")
    
    labels = [f"{r.group} vs {r.next_rank_group}" for r in main_df.itertuples()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pr(better than next-ranked)")
    ax.set_title(f"Consecutive ranking comparisons: {metric}")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.2)
    
    plot_name = f"consecutive_bf_{_safe_slug(metric)}.png"
    _save_fig(fig, assets_dir / plot_name)
    
    report.add_image(
        f"{assets_dir.name}/{plot_name}",
        alt=f"Consecutive BF plot for {metric}"
    )
    
    report.add_footnote(
        "Interpreting consecutive comparisons",
        """
<ul>
  <li><b>Pr(better)</b> close to 1.0 → strong confidence in ranking order</li>
  <li><b>Pr(better)</b> near 0.5 → ranking is uncertain, groups may be equivalent</li>
  <li><b>BF10 > 10</b> → strong evidence for ranking difference</li>
  <li><b>BF10 < 1</b> → evidence against ranking difference (groups may be equivalent)</li>
  <li>Multiple bandwidths show robustness of conclusions</li>
</ul>
"""
    )


# ============================
# HELPER: ADD IMPLICIT EFFECTS TO REPORT
# ============================
def add_implicit_effects_to_report(
    report: pd.DataFrame,
    idata: az.InferenceData,
    datasets: List[str],
    archs: List[str],
    tname: str,
    effect_sd: float,
    hdi_prob: float,
    hdi_lo_col: str,
    hdi_hi_col: str,
) -> pd.DataFrame:
    """
    Adds the *implicit* (sum-to-zero / effects-coded) levels to the effects report so that
    tables + plots show ALL levels:
      - beta_dataset[LAST]
      - beta_arch[LAST]
      - beta_interaction[ALL dataset×arch cells] (including LAST row/col and LAST×LAST)

    Works with your current effects-coding construction:
      - beta_dataset has K-1 params, LAST is -sum(others)
      - beta_arch has J-1 params, LAST is -sum(others)
      - beta_interaction has (K-1)(J-1) params, implied by row/col sum-to-zero.
    """

    def _summarize_1d(draws_1d: np.ndarray) -> Dict[str, float]:
        x = np.asarray(draws_1d).ravel()
        hdi = az.hdi(x, hdi_prob=hdi_prob)
        return {
            "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)),
            "hdi_lo": float(hdi[0]),
            "hdi_hi": float(hdi[1]),
        }

    def _make_row(param: str, post_draws: np.ndarray, prior_draws: np.ndarray) -> Dict[str, object]:
        s = _summarize_1d(post_draws)
        pr_gt0 = float((np.asarray(post_draws).ravel() > 0).mean())

        bf_results = {}
        for bw in SD_KDE_BANDWIDTHS:
            bf10 = _bf10_from_prior_posterior(prior_draws, post_draws, bw=bw)
            bw_str = bw if isinstance(bw, str) else f"{bw:.1f}"
            bf_results[f"BF10_bw_{bw_str}"] = bf10

        bf10_default = bf_results.get("BF10_bw_scott", bf_results[list(bf_results.keys())[0]])

        return {
            "param": param,
            "mean_transformed": s["mean"],
            "sd_transformed": s["sd"],
            f"{hdi_lo_col}_transformed": s["hdi_lo"],
            f"{hdi_hi_col}_transformed": s["hdi_hi"],
            "Pr(>0)": pr_gt0,
            "BF10_interpretation": interpret_bf10(bf10_default),
            **bf_results,
            "scale": tname,
            "prior": "Implied (sum-to-zero) from effects coding",
            # Filled later (same as your current code):
            "OR": np.nan,
            "Δp_mean": np.nan,
            "Δp_lo": np.nan,
            "Δp_hi": np.nan,
            "Abs_Δ_mean": np.nan,
            "Abs_Δ_lo": np.nan,
            "Abs_Δ_hi": np.nan,
            "%_change": np.nan,
            "baseline": np.nan,
        }

    existing = set(report["param"].astype(str).tolist())
    derived_rows = []

    # ---------- DATASET: add LAST level (e.g., S2) ----------
    if len(datasets) >= 2:
        d_exp = datasets[:-1]
        d_last = datasets[-1]

        # posterior: (sample, K-1)
        post_d = (
            idata.posterior["beta_dataset"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "dataset_effect")
            .to_numpy()
        )
        prior_d = (
            idata.prior["beta_dataset"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "dataset_effect")
            .to_numpy()
        )

        post_last = -post_d.sum(axis=1)
        prior_last = -prior_d.sum(axis=1)

        p = f"beta_dataset[{d_last}]"
        if p not in existing:
            derived_rows.append(_make_row(p, post_last, prior_last))
            existing.add(p)

    # ---------- ARCH: add LAST level (e.g., Terramind) ----------
    if len(archs) >= 2:
        a_exp = archs[:-1]
        a_last = archs[-1]

        post_a = (
            idata.posterior["beta_arch"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "arch_effect")
            .to_numpy()
        )
        prior_a = (
            idata.prior["beta_arch"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "arch_effect")
            .to_numpy()
        )

        post_last = -post_a.sum(axis=1)
        prior_last = -prior_a.sum(axis=1)

        p = f"beta_arch[{a_last}]"
        if p not in existing:
            derived_rows.append(_make_row(p, post_last, prior_last))
            existing.add(p)

    # ---------- INTERACTIONS: add ALL cells (including implicit row/col/last×last) ----------
    if len(datasets) >= 2 and len(archs) >= 2:
        K = len(datasets)
        J = len(archs)
        K1 = K - 1
        J1 = J - 1

        post_i = (
            idata.posterior["beta_interaction"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "interaction_effect")
            .to_numpy()
        )
        prior_i = (
            idata.prior["beta_interaction"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "interaction_effect")
            .to_numpy()
        )

        # reshape explicit grid: (sample, K-1, J-1)
        post_grid = post_i.reshape(post_i.shape[0], K1, J1)
        prior_grid = prior_i.reshape(prior_i.shape[0], K1, J1)

        # build full grid: (sample, K, J)
        post_full = np.zeros((post_i.shape[0], K, J))
        prior_full = np.zeros((prior_i.shape[0], K, J))

        # explicit block
        post_full[:, :K1, :J1] = post_grid
        prior_full[:, :K1, :J1] = prior_grid

        # last dataset row (d_last × a_exp): -sum over datasets (explicit)
        post_full[:, K1, :J1] = -post_full[:, :K1, :J1].sum(axis=1)
        prior_full[:, K1, :J1] = -prior_full[:, :K1, :J1].sum(axis=1)

        # last arch col (d_exp × a_last): -sum over archs (explicit)
        post_full[:, :K1, J1] = -post_full[:, :K1, :J1].sum(axis=2)
        prior_full[:, :K1, J1] = -prior_full[:, :K1, :J1].sum(axis=2)

        # last×last: implied (can derive either way; this is consistent)
        post_full[:, K1, J1] = post_full[:, :K1, :J1].sum(axis=(1, 2))
        prior_full[:, K1, J1] = prior_full[:, :K1, :J1].sum(axis=(1, 2))

        # append any missing cells as rows
        for di, d in enumerate(datasets):
            for ai, a in enumerate(archs):
                p = f"beta_interaction[{d}×{a}]"
                if p in existing:
                    continue
                derived_rows.append(_make_row(p, post_full[:, di, ai], prior_full[:, di, ai]))
                existing.add(p)

    if derived_rows:
        report = pd.concat([report, pd.DataFrame(derived_rows)], ignore_index=True)

    return report


def fit_factorial_model_with_bf(
    df: pd.DataFrame,
    dataset_col: str,
    arch_col: str,
    metric_col: str,
    draws=DRAWS,
    tune=TUNE,
    chains=CHAINS,
    target_accept=TARGET_ACCEPT,
    seed=RANDOM_SEED,
    hdi_prob=HDI_PROB,
):
    """
    Factorial model with multiple BF bandwidths.
    IMPORTANT: since we use effects coding, we compute interpretable effect sizes via
    contrasts on predicted means (not via intercept+coef on original scale).
    """
    data = df[[dataset_col, arch_col, metric_col]].dropna().copy()

    data[dataset_col] = pd.Categorical(
        data[dataset_col], categories=DATASET_ORDER, ordered=True
    )
    data[arch_col] = pd.Categorical(data[arch_col], categories=ARCH_ORDER, ordered=True)
    data = data.dropna(subset=[dataset_col, arch_col, metric_col]).copy()
    data[dataset_col] = data[dataset_col].cat.remove_unused_categories()
    data[arch_col] = data[arch_col].cat.remove_unused_categories()

    dcat = pd.Categorical(data[dataset_col])
    acat = pd.Categorical(data[arch_col])

    len_datasets = len(dcat.categories)
    len_archs = len(acat.categories)

    print(f"Fitting a {len_datasets}×{len_archs} factorial model for '{metric_col}'")
    print(f"Datasets: {list(dcat.categories)}")
    print(f"Architectures: {list(acat.categories)}")

    d_idx = dcat.codes
    a_idx = acat.codes
    n_obs = len(d_idx)

    # Effects coding matrices
    d_mat = np.zeros((n_obs, len_datasets - 1))
    a_mat = np.zeros((n_obs, len_archs - 1))

    for i in range(n_obs):
        if d_idx[i] < len_datasets - 1:
            d_mat[i, d_idx[i]] = 1
        else:
            d_mat[i, :] = -1

        if a_idx[i] < len_archs - 1:
            a_mat[i, a_idx[i]] = 1
        else:
            a_mat[i, :] = -1

    da_mat = np.zeros((n_obs, (len_datasets - 1) * (len_archs - 1)))
    col_idx = 0

    for d in range(len_datasets - 1):
        for a in range(len_archs - 1):
            da_mat[:, col_idx] = d_mat[:, d] * a_mat[:, a]
            col_idx += 1

    y_raw = data[metric_col].to_numpy()
    y_t, inv, tname = transform_y(y_raw, metric_col)
    effect_sd = PRIOR_SD_LOGIT if tname == "logit" else PRIOR_SD_LOG

    coords = {
        "dataset_effect": list(dcat.categories)[:-1],
        "arch_effect": list(acat.categories)[:-1],
        "interaction_effect": [
            f"{dcat.categories[d]}×{acat.categories[a]}"
            for d in range(len_datasets - 1)
            for a in range(len_archs - 1)
        ],
    }

    with pm.Model(coords=coords) as model:
        d = pm.Data("D", d_mat)
        a = pm.Data("A", a_mat)
        da = pm.Data("DA", da_mat)

        # Priors
        intercept = pm.Normal("intercept", 0.0, 1.5)
        beta_dataset = pm.Normal("beta_dataset", 0.0, effect_sd, dims="dataset_effect")
        beta_arch = pm.Normal("beta_arch", 0.0, effect_sd, dims="arch_effect")
        beta_interaction = pm.Normal(
            "beta_interaction", 0.0, effect_sd, dims="interaction_effect"
        )

        mu = intercept + pm.math.dot(d, beta_dataset) + pm.math.dot(
            a, beta_arch
        ) + pm.math.dot(da, beta_interaction)

        # Single sigma
        sigma = pm.HalfNormal("sigma", sigma=1.0)

        nu = pm.Exponential("nu", 1 / 30) + 1
        pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=y_t)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            return_inferencedata=True,
        )

        prior = pm.sample_prior_predictive(
            var_names=["beta_dataset", "beta_arch", "beta_interaction"],
            random_seed=seed,
        )
        idata.extend(prior)

        # Posterior predictive check
        post_pred = pm.sample_posterior_predictive(idata, random_seed=seed)

    summary = (
        az.summary(
            idata,
            var_names=["beta_dataset", "beta_arch", "beta_interaction"],
            hdi_prob=hdi_prob,
        )
        .reset_index()
        .rename(columns={"index": "param"})
    )
    hdi_lo_col, hdi_hi_col = _pick_hdi_columns(summary)

    # --- Calculate effects (transformed) + BF sensitivity ---
    rows = []

    def _add_param_rows(var_name: str, dim_name: str) -> None:
        extracted = az.extract(idata, group="posterior", var_names=[var_name])
        da_ex = _as_dataarray(extracted, var_name)

        coord_vals = list(da_ex.coords[dim_name].values)
        for dim_val in coord_vals:
            sel = {dim_name: dim_val}
            post_draws = _extract_draws_1d(idata, "posterior", var_name, sel=sel)
            prior_draws = _extract_draws_1d(idata, "prior", var_name, sel=sel)

            pr_gt0 = float((post_draws > 0).mean())

            bf_results = {}
            for bw in SD_KDE_BANDWIDTHS:
                bf10 = _bf10_from_prior_posterior(prior_draws, post_draws, bw=bw)
                bw_str = bw if isinstance(bw, str) else f"{bw:.1f}"
                bf_results[f"BF10_bw_{bw_str}"] = bf10

            bf10_default = bf_results.get("BF10_bw_scott", bf_results[list(bf_results.keys())[0]])

            param_str = f"{var_name}[{dim_val}]"
            base_row = summary.loc[summary["param"] == param_str].iloc[0].to_dict()

            row = {
                "param": param_str,
                "mean_transformed": base_row["mean"],
                "sd_transformed": base_row["sd"],
                f"{hdi_lo_col}_transformed": base_row[hdi_lo_col],
                f"{hdi_hi_col}_transformed": base_row[hdi_hi_col],
                "Pr(>0)": pr_gt0,
                "BF10_interpretation": interpret_bf10(bf10_default),
                **bf_results,
                "scale": tname,
                "prior": f"Normal(0, {effect_sd}) on {tname} scale",
                # Filled later from predicted-means contrasts:
                "OR": np.nan,
                "Δp_mean": np.nan,
                "Δp_lo": np.nan,
                "Δp_hi": np.nan,
                "Abs_Δ_mean": np.nan,
                "Abs_Δ_lo": np.nan,
                "Abs_Δ_hi": np.nan,
                "%_change": np.nan,
                "baseline": np.nan,
            }
            rows.append(row)

    _add_param_rows("beta_dataset", "dataset_effect")
    _add_param_rows("beta_arch", "arch_effect")
    _add_param_rows("beta_interaction", "interaction_effect")

    report = pd.DataFrame(rows)

    # --- Predicted means per dataset×arch (both scales), with robust stacking ---
    datasets = list(dcat.categories)
    archs = list(acat.categories)

    # Add implicit (sum-to-zero) effects to report
    report = add_implicit_effects_to_report(
        report=report,
        idata=idata,
        datasets=datasets,
        archs=archs,
        tname=tname,
        effect_sd=effect_sd,
        hdi_prob=hdi_prob,
        hdi_lo_col=hdi_lo_col,
        hdi_hi_col=hdi_hi_col,
    )

    # build effect-code rows (same logic as before)
    d_rows = []
    for d_i in range(len_datasets):
        if d_i < len_datasets - 1:
            row = np.zeros(len_datasets - 1)
            row[d_i] = 1.0
        else:
            row = -np.ones(len_datasets - 1)
        d_rows.append(row)

    a_rows = []
    for a_i in range(len_archs):
        if a_i < len_archs - 1:
            row = np.zeros(len_archs - 1)
            row[a_i] = 1.0
        else:
            row = -np.ones(len_archs - 1)
        a_rows.append(row)

    # Stack chain/draw properly (prevents accidental axis mishandling)
    intercept_s = idata.posterior["intercept"].stack(sample=("chain", "draw")).to_numpy().ravel()
    beta_d_s = (
        idata.posterior["beta_dataset"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "dataset_effect")
        .to_numpy()
    )
    beta_a_s = (
        idata.posterior["beta_arch"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "arch_effect")
        .to_numpy()
    )
    beta_i_s = (
        idata.posterior["beta_interaction"]
        .stack(sample=("chain", "draw"))
        .transpose("sample", "interaction_effect")
        .to_numpy()
    )

    pred_draws_transformed = np.zeros((intercept_s.shape[0], len(datasets), len(archs)))
    pred_draws_original = np.zeros((intercept_s.shape[0], len(datasets), len(archs)))
    pred_rows = []

    for di, d_name in enumerate(datasets):
        for ai, a_name in enumerate(archs):
            d_row = d_rows[di]
            a_row = a_rows[ai]

            da_row = np.zeros((len_datasets - 1) * (len_archs - 1))
            c = 0
            for d_j in range(len_datasets - 1):
                for a_j in range(len_archs - 1):
                    da_row[c] = d_row[d_j] * a_row[a_j]
                    c += 1

            mu_s = intercept_s + (beta_d_s @ d_row) + (beta_a_s @ a_row) + (beta_i_s @ da_row)
            mu_orig = inv(mu_s)

            mu_s = np.asarray(mu_s).ravel()
            mu_orig = np.asarray(mu_orig).ravel()

            pred_draws_transformed[:, di, ai] = mu_s
            pred_draws_original[:, di, ai] = mu_orig

            mu_stats_orig = _summarize_draws(mu_orig, hdi_prob=hdi_prob)
            pred_rows.append(
                {
                    "dataset": d_name,
                    "arch": a_name,
                    "mean_original": mu_stats_orig["mean"],
                    "hdi_lo_original": mu_stats_orig["hdi_lo"],
                    "hdi_hi_original": mu_stats_orig["hdi_hi"],
                }
            )

    pred_table = pd.DataFrame(pred_rows)

    # --- Fill INTERPRETABLE effect sizes via contrasts on predicted means (correct under effects coding) ---
    dataset_marg_t = pred_draws_transformed.mean(axis=2)  # (samples, dataset)
    dataset_marg_o = pred_draws_original.mean(axis=2)
    arch_marg_t = pred_draws_transformed.mean(axis=1)     # (samples, arch)
    arch_marg_o = pred_draws_original.mean(axis=1)

    # Grand means (for effects coding interpretation)
    grand_mean_dataset_t = dataset_marg_t.mean(axis=1)
    grand_mean_dataset_o = dataset_marg_o.mean(axis=1)
    grand_mean_arch_t = arch_marg_t.mean(axis=1)
    grand_mean_arch_o = arch_marg_o.mean(axis=1)
    grand_mean_cell_t = pred_draws_transformed.mean(axis=(1, 2))
    grand_mean_cell_o = pred_draws_original.mean(axis=(1, 2))

    def _write_effect(param: str, delta_t: np.ndarray, delta_o: np.ndarray, baseline_o: np.ndarray) -> None:
        s_o = _summarize_draws(delta_o, hdi_prob=hdi_prob)
        s_base = _summarize_draws(baseline_o, hdi_prob=hdi_prob)

        if tname == "log":
            ratio = np.exp(delta_t)
            pct = (ratio - 1.0) * 100.0
            s_pct = _summarize_draws(pct, hdi_prob=hdi_prob)

            report.loc[report["param"] == param, "Abs_Δ_mean"] = s_o["mean"]
            report.loc[report["param"] == param, "Abs_Δ_lo"] = s_o["hdi_lo"]
            report.loc[report["param"] == param, "Abs_Δ_hi"] = s_o["hdi_hi"]
            report.loc[report["param"] == param, "%_change"] = s_pct["mean"]
            report.loc[report["param"] == param, "baseline"] = s_base["mean"]
        else:
            OR = np.exp(delta_t)
            s_or = _summarize_draws(OR, hdi_prob=hdi_prob)

            report.loc[report["param"] == param, "OR"] = s_or["mean"]
            report.loc[report["param"] == param, "Δp_mean"] = s_o["mean"]
            report.loc[report["param"] == param, "Δp_lo"] = s_o["hdi_lo"]
            report.loc[report["param"] == param, "Δp_hi"] = s_o["hdi_hi"]
            report.loc[report["param"] == param, "baseline"] = s_base["mean"]

    # Dataset coefficients: ALL datasets vs GRAND MEAN (marginal over arch)
    for d_name in datasets:
        di = datasets.index(d_name)
        delta_t = dataset_marg_t[:, di] - grand_mean_dataset_t
        delta_o = dataset_marg_o[:, di] - grand_mean_dataset_o
        base_o = grand_mean_dataset_o
        _write_effect(f"beta_dataset[{d_name}]", delta_t, delta_o, base_o)

    # Arch coefficients: ALL archs vs GRAND MEAN (marginal over dataset)
    for a_name in archs:
        ai = archs.index(a_name)
        delta_t = arch_marg_t[:, ai] - grand_mean_arch_t
        delta_o = arch_marg_o[:, ai] - grand_mean_arch_o
        base_o = grand_mean_arch_o
        _write_effect(f"beta_arch[{a_name}]", delta_t, delta_o, base_o)

    # Interaction coefficients: ALL dataset×arch cells
    for d_name in datasets:
        for a_name in archs:
            di = datasets.index(d_name)
            ai = archs.index(a_name)

            # Interaction = cell - row_mean - col_mean + grand_mean
            delta_t = (
                pred_draws_transformed[:, di, ai]
                - dataset_marg_t[:, di]
                - arch_marg_t[:, ai]
                + grand_mean_cell_t
            )
            delta_o = (
                pred_draws_original[:, di, ai]
                - dataset_marg_o[:, di]
                - arch_marg_o[:, ai]
                + grand_mean_cell_o
            )

            base_o = grand_mean_cell_o
            _write_effect(f"beta_interaction[{d_name}×{a_name}]", delta_t, delta_o, base_o)

    return (
        idata,
        report,
        pred_table,
        pred_draws_transformed,
        pred_draws_original,
        (datasets, archs),
        post_pred,
    )


# =============================================================================
# OVERALL RANKING
# =============================================================================


def overall_ranking_across_metrics(
    mu_draws_by_metric: Dict[str, np.ndarray],
    higher_is_better_by_metric: Dict[str, bool],
    group_names: Sequence[str],
    weights: Optional[Dict[str, float]] = None,
):
    metric_list = list(mu_draws_by_metric.keys())

    if weights is None:
        weights = {m: 1.0 for m in metric_list}
    wsum = sum(weights.values())
    weights = {m: weights[m] / wsum for m in metric_list}

    s_count = min(mu_draws_by_metric[m].shape[0] for m in metric_list)
    g_count = len(group_names)

    composite = np.zeros((s_count, g_count))
    for m in metric_list:
        mu = mu_draws_by_metric[m][:s_count, :]
        score = mu if higher_is_better_by_metric[m] else -mu

        sd = score.std() + 1e-12
        z = score / sd

        composite += weights[m] * z

    overall_rank_table = rank_probabilities_from_draws(
        composite, group_names, higher_better=True
    )

    return composite, overall_rank_table, weights


# =============================================================================
# PLOTS
# =============================================================================


def plot_group_forest(
    perf_table: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
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

    y = np.arange(len(df))
    x = df["mean"].to_numpy()
    xerr_lo = x - df[hdi_lo].to_numpy()
    xerr_hi = df[hdi_hi].to_numpy() - x

    fig, ax = plt.subplots(figsize=(10, max(3.8, 0.45 * len(df))))
    ax.errorbar(x, y, xerr=[xerr_lo, xerr_hi], fmt="o", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["group"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel(f"Estimated mean ({metric}) - Original Scale")
    ax.set_title(f"Group means with {HDI_PROB:.0%} interval: {metric}")
    ax.grid(True, axis="x", alpha=0.2)

    _save_fig(fig, out_path)


def plot_pr_best(
    perf_table: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    df = perf_table.copy().sort_values("Pr(best)", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, max(3.8, 0.45 * len(df))))
    ax.barh(df["group"], df["Pr(best)"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pr(best)")
    ax.set_title(f"Posterior probability of being best: {metric}")
    ax.grid(True, axis="x", alpha=0.2)

    _save_fig(fig, out_path)


def plot_factorial_heatmap(
    pred_table: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    pivot = pred_table.pivot(index="dataset", columns="arch", values="mean_original")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist())
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title(f"Posterior mean by dataset×arch: {metric} (Original Scale)")

    # Add values to cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(
                j, i, f"{pivot.values[i, j]:.3f}",
                ha="center", va="center", color="white", fontsize=9
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    _save_fig(fig, out_path)


def plot_factorial_effects(
    effects_report: pd.DataFrame,
    metric: str,
    out_path: Path,
) -> None:
    df = effects_report.copy()
    bf_col = "BF10_bw_scott" if "BF10_bw_scott" in df.columns else [c for c in df.columns if c.startswith("BF10_")][0]
    df = df.sort_values(bf_col, ascending=False).reset_index(drop=True)

    if df["scale"].iloc[0] == "log":
        x = df["Abs_Δ_mean"].to_numpy()
        lo = df["Abs_Δ_lo"].to_numpy()
        hi = df["Abs_Δ_hi"].to_numpy()
        xlabel = f"Effect as absolute change in {metric} (from baseline)"
    else:
        x = df["Δp_mean"].to_numpy()
        lo = df["Δp_lo"].to_numpy()
        hi = df["Δp_hi"].to_numpy()
        xlabel = "Effect as Δ on original scale (from baseline)"

    y = np.arange(len(df))
    xerr_lo = x - lo
    xerr_hi = hi - x

    fig, ax = plt.subplots(figsize=(10, max(4.2, 0.32 * len(df))))
    ax.errorbar(x, y, xerr=[xerr_lo, xerr_hi], fmt="o", capsize=3)
    ax.axvline(0, linestyle="--", linewidth=1, color='red', alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["param"].tolist())
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(f"Factorial effects (interpretable scale): {metric}")
    ax.grid(True, axis="x", alpha=0.2)

    _save_fig(fig, out_path)


def plot_overall_pr_best(
    overall_rank_table: pd.DataFrame,
    out_path: Path,
) -> None:
    df = overall_rank_table.copy().sort_values("Pr(best)", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(3.8, 0.45 * len(df))))
    ax.barh(df["group"], df["Pr(best)"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pr(best) overall")
    ax.set_title("Overall multi-metric: Pr(best)")
    ax.grid(True, axis="x", alpha=0.2)
    _save_fig(fig, out_path)


def plot_posterior_predictive_check(
    idata,
    post_pred,
    y_obs,
    metric: str,
    out_path: Path,
) -> None:
    """Plot posterior predictive check to validate model fit."""
    fig, ax = plt.subplots(figsize=(10, 5))

    y_pred = post_pred.posterior_predictive["y"].values

    ax.hist(y_obs, bins=30, alpha=0.5, label="Observed", density=True, color='blue')

    for i in range(min(100, y_pred.shape[1])):
        ax.hist(y_pred[0, i, :], bins=30, alpha=0.01, density=True, color='red')

    ax.set_xlabel(f"{metric} (transformed scale)")
    ax.set_ylabel("Density")
    ax.set_title(f"Posterior Predictive Check: {metric}")
    ax.legend()
    ax.grid(True, alpha=0.2)

    _save_fig(fig, out_path)


def plot_prior_predictive_check(
    metric: str,
    scale: str,
    inv_transform: Callable,
    out_path: Path,
    n_samples: int = 1000,
) -> None:
    """Plot prior predictive distribution on original scale."""
    effect_sd = PRIOR_SD_LOGIT if scale == "logit" else PRIOR_SD_LOG

    intercept_prior = np.random.normal(0, 1.5, n_samples)
    effect_prior = np.random.normal(0, effect_sd, n_samples)

    baseline = inv_transform(intercept_prior)
    with_effect = inv_transform(intercept_prior + effect_prior)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(baseline, bins=50, alpha=0.7, density=True)
    ax1.set_xlabel(f"{metric} (original scale)")
    ax1.set_ylabel("Density")
    ax1.set_title("Prior: Baseline Performance")
    ax1.grid(True, alpha=0.2)

    ax2.hist(with_effect - baseline, bins=50, alpha=0.7, density=True)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel(f"Effect size on {metric}")
    ax2.set_ylabel("Density")
    ax2.set_title("Prior: Effect Sizes")
    ax2.grid(True, alpha=0.2)

    _save_fig(fig, out_path)


# =============================================================================
# MAIN
# =============================================================================


def main():
    base_dir = Path(__file__).resolve().parent
    assets_dir = base_dir / "stats_report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    report_path = base_dir / "stats_report.html"
    report = HtmlReport(report_path, title="Bayesian Benchmark Statistics Report (Improved)")

    # Header
    report.add_html('<div class="topbar">')
    report.add_html(f'<span class="badge">PyMC {pm.__version__}</span>')
    report.add_html(
        f'<span class="badge">Draws {DRAWS} | Tune {TUNE} | Chains {CHAINS}</span>'
    )
    report.add_html(
        f'<span class="badge">Target accept {TARGET_ACCEPT} | HDI {HDI_PROB:.0%}</span>'
    )
    report.add_html('<span class="badge">Hierarchical Variance | Multiple BF Bandwidths</span>')
    if TEST:
        report.add_html('<span class="badge" style="color:var(--accent2)">TESTMODE</span>')
    report.add_html("</div>")

    report.add_card_start()
    report.add_paragraph(
        "This IMPROVED report includes: (1) Hierarchical variance structure for better regularization, "
        "(2) All outputs transformed to natural, interpretable scales, (3) Multiple bandwidth sensitivity "
        "analysis for Bayes factors, (4) Posterior and prior predictive checks, (5) ROPE applied on "
        "transformed scale for consistency with the model, (6) Enhanced visualizations with actual values."
    )
    report.add_card_end()

    all_data = load_checkpoint_metrics(CSV_SPECS, METRICS)
    all_data["group"] = pd.Categorical(all_data["group"], categories=GROUP_ORDER, ordered=True)

    if TEST and not all_data.empty:
        rng = np.random.default_rng(RANDOM_SEED)

        # You want 3x3 = 9 combinations: dataset x arch.
        # Define these if you don't already have them:
        DATASET_ORDER = ["AE", "PS", "S2"]
        ARCH_ORDER = ["U-Net", "Swin", "Terramind"]

        # Helper: build the group label exactly like your real data
        def make_group(dataset: str, arch: str) -> str:
            if arch == "U-Net":
                return f"U-Net | {dataset}"
            if arch == "Swin":
                return f"Swin U-Net | {dataset}"
            if arch == "Terramind":
                return f"Terramind | {dataset}"
            return f"{arch} | {dataset}"

        # How many synthetic runs per combo
        N_ROWS = 10

        # Concentration controls "narrowness":
        # - 30: fairly wide
        # - 80: narrow
        # - 150+: very narrow
        KAPPA_BY_METRIC = {m: 120.0 for m in METRICS}

        # Base means per metric (in (0,1)). Pick something plausible.
        BASE_MEAN = {
            "IoU": 0.45,
            "dice_coef": 0.55,
            "normalized_surface_distance": 0.35,
            "mean_epistemic_uncertainty": 0.25,
            "mean_aleatoric_uncertainty": 0.25,
        }

        # Systematic shifts (still bounded after clipping).
        ARCH_SHIFT = {
            "U-Net": {
                "IoU": 0.00,
                "dice_coef": 0.00,
                "normalized_surface_distance": 0.00,
                "mean_epistemic_uncertainty": 0.00,
                "mean_aleatoric_uncertainty": 0.00,
            },
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
            "PS": {
                "IoU": 0.00,
                "dice_coef": 0.00,
                "normalized_surface_distance": 0.00,
                "mean_epistemic_uncertainty": 0.00,
                "mean_aleatoric_uncertainty": 0.00,
            },
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

    # Tabs
    tab_list = [("tab-overview", "Overview")]
    for m in METRICS:
        tab_list.append((f"tab-{_safe_slug(m)}", m))
    tab_list.append(("tab-overall", "Overall"))

    report.start_tabs(tab_list, default_tab_id="tab-overview")

    # Overview tab
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

    # Store for overall ranking
    mu_draws_by_metric: Dict[str, np.ndarray] = {}
    hib_by_metric: Dict[str, bool] = {}
    groups_by_metric: Dict[str, Sequence[str]] = {}

    # Metric tabs
    for metric in METRICS:
        tab_id = f"tab-{_safe_slug(metric)}"
        report.open_tab_content(tab_id)

        data = all_data[["group", "dataset", "arch", metric]].dropna().copy()

        if pd.api.types.is_categorical_dtype(data["group"]):
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

        # Get transformation info for prior predictive
        y_raw = data[metric].to_numpy()
        y_t, inv, tname = transform_y(y_raw, metric)

        # Prior predictive check
        prior_pred_name = f"prior_pred_{_safe_slug(metric)}.png"
        plot_prior_predictive_check(metric, tname, inv, assets_dir / prior_pred_name)

        # Distribution plot
        fig, ax = plt.subplots(figsize=(10, 4.8))
        sns.kdeplot(data=data, x=metric, hue="group", ax=ax)
        ax.set_title(f"Distribution across runs: {metric}")
        ax.grid(True, alpha=0.2)
        kde_name = f"kde_{_safe_slug(metric)}.png"
        _save_fig(fig, assets_dir / kde_name)

        report.add_card_start()
        report.add_h3("Distribution across runs")
        report.add_image(f"{assets_dir.name}/{kde_name}", alt=f"KDE plot for {metric}")
        report.add_card_end()

        # Hierarchical group model
        idata, perf_table, _, mu_draws, scale_name, groups_this, post_pred = (
            fit_group_model_with_ranking(
                data, group_col="group", metric_col=metric
            )
        )

        # Posterior predictive check
        post_pred_name = f"post_pred_{_safe_slug(metric)}.png"
        plot_posterior_predictive_check(idata, post_pred, y_t, metric, assets_dir / post_pred_name)

        # Forest and Pr(best) plots
        forest_name = f"group_forest_{_safe_slug(metric)}.png"
        plot_group_forest(perf_table, metric, assets_dir / forest_name)

        prbest_name = f"prbest_{_safe_slug(metric)}.png"
        plot_pr_best(perf_table, metric, assets_dir / prbest_name)

        report.add_card_start()
        report.add_h3("Hierarchical group model (with hierarchical variance)")
        report.add_paragraph(
            f"Modelled on <span class='badge'>{scale_name}</span> scale; "
            "reporting mu_orig on original scale. Hierarchical variance improves regularization."
        )
        add_table_with_column_guide(report, perf_table, COLGUIDE_PERF)

        report.add_h3("Model Validation Plots")
        report.add_html('<div class="grid-2">')
        report.add_html('<div class="card" style="margin:0">')
        report.add_h3("Prior Predictive Check")
        report.add_image(
            f"{assets_dir.name}/{prior_pred_name}",
            alt=f"Prior predictive check for {metric}"
        )
        report.add_paragraph("Check if priors are reasonable on original scale.")
        report.add_html("</div>")
        report.add_html('<div class="card" style="margin:0">')
        report.add_h3("Posterior Predictive Check")
        report.add_image(
            f"{assets_dir.name}/{post_pred_name}",
            alt=f"Posterior predictive check for {metric}"
        )
        report.add_paragraph("Red overlay shows model predictions; blue shows observed data.")
        report.add_html("</div>")
        report.add_html("</div>")

        report.add_h3("Group Comparison Plots")
        report.add_html('<div class="grid-2">')
        report.add_html('<div class="card" style="margin:0">')
        report.add_h3("Group means (with HDI)")
        report.add_image(
            f"{assets_dir.name}/{forest_name}", alt=f"Group means forest plot for {metric}"
        )
        report.add_html("</div>")
        report.add_html('<div class="card" style="margin:0">')
        report.add_h3("Pr(best)")
        report.add_image(
            f"{assets_dir.name}/{prbest_name}", alt=f"Pr(best) plot for {metric}"
        )
        report.add_html("</div>")
        report.add_html("</div>")
        report.add_card_end()

        # Store for overall
        mu_draws_by_metric[metric] = mu_draws
        hib_by_metric[metric] = higher_is_better(metric)
        groups_by_metric[metric] = list(groups_this)

        # Consecutive ranking BF10 analysis
        add_consecutive_bf_analysis(
            report=report,
            mu_draws=mu_draws,
            perf_table=perf_table,
            group_names=groups_this,
            metric=metric,
            higher_better=higher_is_better(metric),
            assets_dir=assets_dir,
        )

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
        )

        # Plots
        heat_name = f"heat_{_safe_slug(metric)}.png"
        plot_factorial_heatmap(pred_table, metric, assets_dir / heat_name)

        eff_name = f"effects_{_safe_slug(metric)}.png"
        plot_factorial_effects(factorial_report, metric, assets_dir / eff_name)

        # Contrasts (architecture within each dataset)
        rope_transformed = ROPE_BY_METRIC_TRANSFORMED.get(metric, 0.05)
        contrast_df = contrasts_within_each_dataset(
            pred_draws_transformed=pred_draws_transformed,
            pred_draws_original=pred_draws_original,
            datasets=datasets,
            archs=archs,
            baseline_arch=BASELINE_ARCH,
            rope_transformed=rope_transformed,
            hdi_prob=HDI_PROB,
        )

        # Dataset contrasts within each architecture (mirror)
        dataset_contrast_df = contrasts_within_each_architecture(
            pred_draws_transformed=pred_draws_transformed,
            pred_draws_original=pred_draws_original,
            datasets=datasets,
            archs=archs,
            baseline_dataset=BASELINE_DATASET,
            rope_transformed=rope_transformed,
            hdi_prob=HDI_PROB,
        )

        # Overall/marginal contrasts
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

        contrast_plot_name = f"contrasts_{_safe_slug(metric)}.png"
        plot_contrasts(contrast_df, metric, assets_dir / contrast_plot_name)

        arch_overall_plot_name = f"arch_overall_{_safe_slug(metric)}.png"
        plot_contrasts(arch_overall_df, metric, assets_dir / arch_overall_plot_name)

        dataset_overall_plot_name = f"dataset_overall_{_safe_slug(metric)}.png"
        plot_contrasts(dataset_overall_df, metric, assets_dir / dataset_overall_plot_name)

        # Interpretable effects view
        if factorial_report["scale"].iloc[0] == "log":
            eff_view = factorial_report.loc[
                :,
                [
                    "param",
                    "Pr(>0)",
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
                ],
            ].copy()
        else:
            eff_view = factorial_report.loc[
                :,
                [
                    "param",
                    "Pr(>0)",
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
                ],
            ].copy()

        report.add_card_start()
        report.add_h3("Factorial effects (Natural Scale with BF Sensitivity)")
        report.add_paragraph(
            "All effects shown on natural, interpretable scales. Multiple BF bandwidths "
            "show robustness of evidence. Consistent BF across bandwidths = stronger conclusion."
        )
        if factorial_report["scale"].iloc[0] == "log":
            add_table_with_column_guide(report, eff_view, COLGUIDE_EFFECTS_LOG)
        else:
            add_table_with_column_guide(report, eff_view, COLGUIDE_EFFECTS_LOGIT)

        report.add_h3("Factorial effects plot")
        report.add_image(
            f"{assets_dir.name}/{eff_name}", alt=f"Factorial effects plot for {metric}"
        )

        report.add_h3("Predicted means per dataset×arch")
        pred_view = pred_table[["dataset", "arch", "mean_original", "hdi_lo_original", "hdi_hi_original"]].copy()
        pred_view = pred_view.rename(columns={
            "mean_original": "mean",
            "hdi_lo_original": "hdi_lo",
            "hdi_hi_original": "hdi_hi",
        })
        add_table_with_column_guide(report, pred_view, COLGUIDE_PRED)

        report.add_h3("Heatmap of posterior mean (Original Scale)")
        report.add_image(f"{assets_dir.name}/{heat_name}", alt=f"Heatmap for {metric}")

        report.add_h3(f"Key architecture contrasts (within each dataset, baseline arch={BASELINE_ARCH}; ROPE={rope_transformed} on transformed scale)")
        contrast_view = contrast_df[
            ["dataset", "contrast", "mean_original", "hdi_lo_original", "hdi_hi_original", "Pr(>0)", "Pr(|Δ|<ROPE)"]
        ].copy().rename(columns={
            "mean_original": "Δ_mean",
            "hdi_lo_original": "Δ_lo",
            "hdi_hi_original": "Δ_hi",
        })
        add_table_with_column_guide(report, contrast_view, COLGUIDE_CONTRASTS_DATASET)

        if not contrast_df.empty:
            report.add_h3("Contrast plot")
            report.add_image(
                f"{assets_dir.name}/{contrast_plot_name}",
                alt=f"Contrast plot for {metric}",
            )

        report.add_h3(f"Key dataset contrasts (within each architecture, baseline dataset={BASELINE_DATASET}; ROPE={rope_transformed} on transformed scale)")
        dataset_contrast_view = dataset_contrast_df[
            ["arch", "contrast", "mean_original", "hdi_lo_original", "hdi_hi_original", "Pr(>0)", "Pr(|Δ|<ROPE)"]
        ].copy().rename(columns={
            "mean_original": "Δ_mean",
            "hdi_lo_original": "Δ_lo",
            "hdi_hi_original": "Δ_hi",
        })
        add_table_with_column_guide(report, dataset_contrast_view, COLGUIDE_CONTRASTS_ARCH)

        report.add_h3(f"Overall architecture contrasts (marginal over datasets, baseline arch={BASELINE_ARCH})")
        arch_overall_view = arch_overall_df[
            ["scope", "contrast", "mean_original", "hdi_lo_original", "hdi_hi_original", "Pr(>0)", "Pr(|Δ|<ROPE)"]
        ].copy()
        add_table_with_column_guide(report, arch_overall_view, COLGUIDE_CONTRASTS_OVERALL)

        if not arch_overall_df.empty:
            report.add_image(
                f"{assets_dir.name}/{arch_overall_plot_name}",
                alt=f"Overall architecture contrasts for {metric}",
            )

        report.add_h3(f"Overall dataset contrasts (marginal over architectures, baseline dataset={BASELINE_DATASET})")
        dataset_overall_view = dataset_overall_df[
            ["scope", "contrast", "mean_original", "hdi_lo_original", "hdi_hi_original", "Pr(>0)", "Pr(|Δ|<ROPE)"]
        ].copy()
        add_table_with_column_guide(report, dataset_overall_view, COLGUIDE_CONTRASTS_OVERALL)

        if not dataset_overall_df.empty:
            report.add_image(
                f"{assets_dir.name}/{dataset_overall_plot_name}",
                alt=f"Overall dataset contrasts for {metric}",
            )

        report.add_footnote(
            "Understanding the factorial section",
            """
<ul>
  <li><b>Effects coding:</b> Coefficients are not simple “add to intercept” baselines.</li>
  <li><b>Interpretable effects:</b> We compute OR/Δp (or absolute change) from <b>contrasts on predicted means</b>, which is correct under effects coding.</li>
  <li><b>BF sensitivity:</b> If BF values are similar across bandwidths, the evidence is robust.</li>
  <li><b>ROPE on transformed scale:</b> For consistency with the model parameter space where effects are estimated.</li>
</ul>
""",
        )
        report.add_card_end()

        report.close_tab_content()

    # Overall tab
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

            aligned_draws: Dict[str, np.ndarray] = {}
            for m in METRICS:
                if m not in mu_draws_by_metric:
                    continue
                cols = list(groups_by_metric[m])
                idx = [cols.index(g) for g in group_names]
                aligned_draws[m] = mu_draws_by_metric[m][:, idx]

            _, overall_rank_table, used_weights = overall_ranking_across_metrics(
                mu_draws_by_metric=aligned_draws,
                higher_is_better_by_metric=hib_by_metric,
                group_names=group_names,
                weights=METRIC_WEIGHTS,
            )

            overall_plot_name = "overall_prbest.png"
            plot_overall_pr_best(overall_rank_table, assets_dir / overall_plot_name)

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
