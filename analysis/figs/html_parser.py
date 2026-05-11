from __future__ import annotations

import re
import html as htmllib
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================================
# Shared HTML pre-cleaning
# ============================================================================

def _sanitize_invalid_rope_header_tokens(html_text: str) -> str:
    """
    Some report versions contain raw '<' in plain-text headers (invalid HTML), e.g.:
      Pr(|Δ|<ROPE)
      Pr(Δ<-ROPE)

    HTML parsers may treat these as tags and corrupt the header row.
    Escape these specific patterns before parsing tables.
    """
    replacements = {
        "Pr(|Δ|<ROPE)": "Pr(|Δ|&lt;ROPE)",
        "Pr(Δ<-ROPE)":  "Pr(Δ&lt;-ROPE)",
        # ASCII variants (future-proofing)
        "Pr(|d|<ROPE)": "Pr(|d|&lt;ROPE)",
        "Pr(d<-ROPE)":  "Pr(d&lt;-ROPE)",
    }
    for bad, good in replacements.items():
        html_text = html_text.replace(bad, good)
    return html_text


def _load_all_tables(html_path: Path) -> List[pd.DataFrame]:
    raw_html = html_path.read_text(encoding="utf-8", errors="replace")
    raw_html = _sanitize_invalid_rope_header_tokens(raw_html)
    return pd.read_html(StringIO(raw_html))


# ============================================================================
# Utilities
# ============================================================================

_METRIC_ORDER = ["IoU", "Dice", "NSD", "Epistemic", "Aleatoric"]
_ROPE_PROB_COLS = ["Pr(Δ>ROPE)", "Pr(|Δ|<ROPE)", "Pr(Δ<-ROPE)", "Pr(>0)"]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df.columns are simple strings (handles MultiIndex from read_html)."""
    if df is None:
        return df
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            " ".join([str(x) for x in tup if str(x) != "nan"]).strip()
            for tup in out.columns.values
        ]
    else:
        out.columns = [str(c) for c in out.columns]
    return out


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan}),
        errors="coerce",
    )


def _standardize_minus_and_spaces(s: pd.Series) -> pd.Series:
    out = s.astype(str)
    out = out.str.replace("−", "-", regex=False).str.replace("–", "-", regex=False).str.replace("—", "-", regex=False)
    out = out.str.replace(r"\s*-\s*", " - ", regex=True)
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()
    return out


# ============================================================================
# ── SECTION 1: Factorial effects ─────────────────────────────────────────────
# ============================================================================

def extract_factorial_effects(html_path: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Extract the 5 factorial-effects tables from the report and return them in plot-ready form.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: "IoU", "Dice", "NSD", "Epistemic", "Aleatoric"
    """
    html_path = Path(html_path)
    if not html_path.exists():
        raise FileNotFoundError(f"HTML report not found: {html_path}")

    tables = _load_all_tables(html_path)

    factorial_tables = []
    for idx, df in enumerate(tables):
        if df is None or df.empty:
            continue

        df = _flatten_columns(df)
        df = _canonicalize_dataframe_columns(df)

        if not _is_factorial_effects_table(df):
            continue

        df = _prepare_factorial_dataframe(df)
        factorial_tables.append((idx, df))

    if len(factorial_tables) < 5:
        found_idxs = [i for i, _ in factorial_tables]
        raise ValueError(
            f"Expected >=5 factorial-effects tables, found {len(factorial_tables)} "
            f"(table indices: {found_idxs})."
        )

    metric_order = _METRIC_ORDER

    factorial_effects: Dict[str, pd.DataFrame] = {}
    for metric, (_, df) in zip(metric_order, factorial_tables[:5]):
        _validate_required_columns(df, metric=metric)
        factorial_effects[metric] = df

    return factorial_effects


def _is_factorial_effects_table(df: pd.DataFrame) -> bool:
    """Identify factorial-effects tables by presence of 'param' and beta_* rows."""
    if "param" not in df.columns:
        return False
    s = df["param"].astype(str)
    return s.str.contains(r"beta_(?:dataset|arch|interaction)", regex=True, na=False).any()


def _canonicalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for factorial tables."""
    rename_map = {c: _canonicalize_column_name(c) for c in df.columns}
    return df.rename(columns=rename_map)


def _canonicalize_column_name(col) -> str:
    s = str(col)
    s = htmllib.unescape(s)
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s_stripped = re.sub(r"\s+", "", s)
    low = s_stripped.lower()

    # Core columns
    if low.startswith("param"):
        return "param"
    if low.startswith("baseline"):
        return "baseline"

    # Posterior probability columns
    if low.startswith("pr(>0)"):
        return "Pr(>0)"
    if low.startswith("pr(δ>rope") or low.startswith("pr(Δ>rope".lower()) or low.startswith("pr(d>rope"):
        return "Pr(Δ>ROPE)"
    if low.startswith("pr(|δ|<rope") or low.startswith("pr(|Δ|<rope".lower()) or low.startswith("pr(|d|<rope"):
        return "Pr(|Δ|<ROPE)"
    if low.startswith("pr(δ<-ro") or low.startswith("pr(Δ<-ro".lower()) or low.startswith("pr(d<-ro"):
        return "Pr(Δ<-ROPE)"

    # Effect-size columns (bounded metrics)
    if low.startswith("or"):
        return "OR"
    if low.startswith("δp_mean") or low.startswith("Δp_mean".lower()) or low.startswith("dp_mean"):
        return "Δp_mean"
    if low.startswith("δp_lo") or low.startswith("Δp_lo".lower()) or low.startswith("dp_lo"):
        return "Δp_lo"
    if low.startswith("δp_hi") or low.startswith("Δp_hi".lower()) or low.startswith("dp_hi"):
        return "Δp_hi"

    # Effect-size columns (uncertainty metrics)
    if low.startswith("abs_δ_mean") or low.startswith("abs_Δ_mean".lower()) or low.startswith("abs_d_mean"):
        return "Abs_Δ_mean"
    if low.startswith("abs_δ_lo") or low.startswith("abs_Δ_lo".lower()) or low.startswith("abs_d_lo"):
        return "Abs_Δ_lo"
    if low.startswith("abs_δ_hi") or low.startswith("abs_Δ_hi".lower()) or low.startswith("abs_d_hi"):
        return "Abs_Δ_hi"
    if "%" in low and "change" in low:
        return "%_change"

    return str(col).strip()


def _clean_numeric_column(series: pd.Series) -> pd.Series:
    s = series.copy().astype(str).str.strip()
    replacements = {
        "":      np.nan,
        "NaN":   np.nan,
        "nan":   np.nan,
        "None":  np.nan,
        "∞":     np.inf,
        "-∞":    -np.inf,
    }
    s = s.replace(replacements)
    return pd.to_numeric(s, errors="coerce")


def _prepare_factorial_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "Pr(>0)", "Pr(Δ>ROPE)", "Pr(|Δ|<ROPE)", "Pr(Δ<-ROPE)",
        "OR",
        "Δp_mean", "Δp_lo", "Δp_hi",
        "Abs_Δ_mean", "Abs_Δ_lo", "Abs_Δ_hi",
        "%_change",
        "baseline",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _clean_numeric_column(df[col])

    p = df["param"].astype(str)
    df["effect_type"] = "other"
    df.loc[p.str.contains("beta_dataset",     na=False), "effect_type"] = "dataset"
    df.loc[p.str.contains("beta_arch",        na=False), "effect_type"] = "arch"
    df.loc[p.str.contains("beta_interaction", na=False), "effect_type"] = "interaction"

    df["param_clean"] = (
        p.str.replace(r"beta_dataset|beta_arch|beta_interaction", "", regex=True)
         .str.strip("[]")
         .str.strip()
    )

    return df


def _validate_required_columns(df: pd.DataFrame, metric: str = "") -> None:
    core_required = ["param", "effect_type", "param_clean", "baseline"]
    missing_core = [c for c in core_required if c not in df.columns]
    if missing_core:
        raise ValueError(f"{metric}: missing core columns {missing_core}. Got: {list(df.columns)}")

    rope_required = ["Pr(Δ>ROPE)", "Pr(|Δ|<ROPE)", "Pr(Δ<-ROPE)"]
    missing_rope = [c for c in rope_required if c not in df.columns]
    if missing_rope:
        raise ValueError(f"{metric}: missing ROPE probability columns {missing_rope}. Got: {list(df.columns)}")

    bounded_ok = all(c in df.columns for c in ["Δp_mean", "Δp_lo", "Δp_hi"])
    unct_ok    = all(c in df.columns for c in ["Abs_Δ_mean", "Abs_Δ_lo", "Abs_Δ_hi"])
    if not (bounded_ok or unct_ok):
        raise ValueError(f"{metric}: missing effect interval columns. Got: {list(df.columns)}")


# ============================================================================
# ── SECTION 2: Contrast column normalization (shared) ────────────────────────
# ============================================================================

def _clean_contrast_col(col: str) -> str:
    """Canonicalize column names used by contrast tables (arch + dataset)."""
    s = htmllib.unescape(str(col)).strip()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s_nows = re.sub(r"\s+", "", s)
    low = s_nows.lower()

    # Core id columns
    if low in {"dataset"}:
        return "dataset"
    if low in {"arch", "architecture"}:
        return "arch"
    if low in {"scope"}:
        return "scope"
    if low in {"contrast"}:
        return "contrast"

    # Overall-contrast columns
    if low in {"mean_original", "meanoriginal"}:
        return "mean_original"
    if low in {"hdi_lo_original", "hdilo_original", "hdilooriginal"}:
        return "hdi_lo_original"
    if low in {"hdi_hi_original", "hdihi_original", "hdihioriginal"}:
        return "hdi_hi_original"

    # Delta columns
    if re.match(r"^[δΔd]_mean$", s_nows, flags=re.IGNORECASE):
        return "Δ_mean"
    if re.match(r"^[δΔd]_lo$", s_nows, flags=re.IGNORECASE):
        return "Δ_lo"
    if re.match(r"^[δΔd]_hi$", s_nows, flags=re.IGNORECASE):
        return "Δ_hi"

    # ROPE probability columns
    if re.match(r"^Pr\(\|[δΔd]\|<ROPE\)$", s_nows):
        return "Pr(|Δ|<ROPE)"
    if re.match(r"^Pr\([δΔd]>ROPE\)$", s_nows):
        return "Pr(Δ>ROPE)"
    if re.match(r"^Pr\([δΔd]<-ROPE\)$", s_nows):
        return "Pr(Δ<-ROPE)"
    if s_nows == "Pr(>0)":
        return "Pr(>0)"

    return str(col).strip()


def _normalise_contrast_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_columns(df)
    return df.rename(columns={c: _clean_contrast_col(c) for c in df.columns})


def _add_arch_contrast_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip the ' - U-Net' suffix to get just the comparison architecture name."""
    df = df.copy()
    df["contrast_clean"] = (
        df["contrast"]
        .astype(str)
        .str.replace("−", "-", regex=False)
        .str.replace(r"\s*-\s*U-Net\s*$", "", regex=True)
        .str.strip()
        .str.replace("Terramind", "TM", regex=False)
    )
    return df


def _add_dataset_contrast_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    For dataset contrasts baseline=AE, keep readable labels like 'PS - AE', 'S2 - AE'.
    """
    df = df.copy()
    df["contrast_clean"] = _standardize_minus_and_spaces(df["contrast"])

    # Optional normalization if report uses long names
    df["contrast_clean"] = (
        df["contrast_clean"]
          .str.replace(r"\bPlanetScope\b", "PS", regex=True)
          .str.replace(r"\bPlanetscope\b", "PS", regex=True)
          .str.replace(r"\bSentinel[- ]?2\b", "S2", regex=True)
          .str.replace(r"\bAerial\b", "AE", regex=True)
    )
    return df


# ============================================================================
# ── SECTION 3: Architecture contrasts ────────────────────────────────────────
# ============================================================================

def extract_arch_contrasts(html_path: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Extract architecture contrast tables (baseline = U-Net, within each dataset).
    """
    html_path = Path(html_path)
    if not html_path.exists():
        raise FileNotFoundError(f"HTML report not found: {html_path}")

    tables = _load_all_tables(html_path)

    arch_contrast_tables = []
    for df in tables:
        if _is_arch_contrast_table(df):
            arch_contrast_tables.append(_prepare_arch_contrast_df(df))

    if len(arch_contrast_tables) < len(_METRIC_ORDER):
        raise ValueError(
            f"Expected {len(_METRIC_ORDER)} arch-contrast tables, found {len(arch_contrast_tables)}."
        )

    return {metric: arch_contrast_tables[i] for i, metric in enumerate(_METRIC_ORDER)}


def extract_overall_arch_contrasts(html_path: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Extract marginal (overall) architecture contrast tables — averaged over datasets, baseline = U-Net.
    """
    html_path = Path(html_path)
    if not html_path.exists():
        raise FileNotFoundError(f"HTML report not found: {html_path}")

    tables = _load_all_tables(html_path)

    overall_tables = []
    for df in tables:
        if _is_overall_arch_contrast_table(df):
            overall_tables.append(_prepare_overall_contrast_df(df))

    if len(overall_tables) < len(_METRIC_ORDER):
        raise ValueError(
            f"Expected {len(_METRIC_ORDER)} overall arch-contrast tables, found {len(overall_tables)}."
        )

    return {metric: overall_tables[i] for i, metric in enumerate(_METRIC_ORDER)}


def _is_arch_contrast_table(df: pd.DataFrame) -> bool:
    """
    True for within-dataset arch contrast tables:
      must have columns dataset, contrast, Δ_mean; and contrast values should reference U-Net baseline.
    """
    if df is None or df.empty:
        return False
    d = _normalise_contrast_cols(df)
    cols = {str(c).strip().lower() for c in d.columns}

    if not ({"dataset", "contrast", "Δ_mean"} <= set(d.columns)):
        # use canonical name check
        if not ("dataset" in cols and "contrast" in cols and "Δ_mean".lower() in cols):
            return False

    # Baseline filter: contrasts should include U-Net
    if "contrast" not in d.columns:
        return False
    cvals = d["contrast"].astype(str)
    return cvals.str.contains(r"U-?Net", case=False, na=False).any()


def _is_overall_arch_contrast_table(df: pd.DataFrame) -> bool:
    """
    True for marginal arch contrast tables:
      must have scope, contrast, mean_original
      and scope text contains 'avg over datasets' (not avg over arch).
    """
    if df is None or df.empty:
        return False

    d = _normalise_contrast_cols(df)
    cols = {str(c).strip().lower() for c in d.columns}
    if not ("scope" in cols and "contrast" in cols and "mean_original" in cols):
        return False

    scope_vals = d["scope"].astype(str).str.lower()
    ok_scope = scope_vals.str.contains(r"avg\.?\s*over\s*datasets|averag(ed)?\s*over\s*datasets", na=False).any()
    if not ok_scope:
        return False

    # Baseline filter: contrasts should include U-Net
    cvals = d["contrast"].astype(str)
    return cvals.str.contains(r"U-?Net", case=False, na=False).any()


def _prepare_arch_contrast_df(df: pd.DataFrame) -> pd.DataFrame:
    d = _normalise_contrast_cols(df).copy()
    for col in ["Δ_mean", "Δ_lo", "Δ_hi"] + _ROPE_PROB_COLS:
        if col in d.columns:
            d[col] = _to_numeric(d[col])
    return _add_arch_contrast_clean(d).reset_index(drop=True)


def _prepare_overall_contrast_df(df: pd.DataFrame) -> pd.DataFrame:
    d = _normalise_contrast_cols(df).copy()
    for col in ["mean_original", "hdi_lo_original", "hdi_hi_original"] + _ROPE_PROB_COLS:
        if col in d.columns:
            d[col] = _to_numeric(d[col])
    return _add_arch_contrast_clean(d).reset_index(drop=True)


# ============================================================================
# ── SECTION 4: Dataset contrasts ─────────────────────────────────────────────
# ============================================================================

def extract_dataset_contrasts(html_path: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Extract dataset contrast tables (baseline = AE, within each architecture).
    """
    html_path = Path(html_path)
    if not html_path.exists():
        raise FileNotFoundError(f"HTML report not found: {html_path}")

    tables = _load_all_tables(html_path)

    ds_contrast_tables = []
    for df in tables:
        if _is_dataset_contrast_table(df):
            ds_contrast_tables.append(_prepare_dataset_contrast_df(df))

    if len(ds_contrast_tables) < len(_METRIC_ORDER):
        raise ValueError(
            f"Expected {len(_METRIC_ORDER)} dataset-contrast tables, found {len(ds_contrast_tables)}."
        )

    return {metric: ds_contrast_tables[i] for i, metric in enumerate(_METRIC_ORDER)}


def extract_overall_dataset_contrasts(html_path: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Extract marginal (overall) dataset contrast tables — averaged over architectures, baseline = AE.
    """
    html_path = Path(html_path)
    if not html_path.exists():
        raise FileNotFoundError(f"HTML report not found: {html_path}")

    tables = _load_all_tables(html_path)

    overall_tables = []
    for df in tables:
        if _is_overall_dataset_contrast_table(df):
            overall_tables.append(_prepare_overall_dataset_contrast_df(df))

    if len(overall_tables) < len(_METRIC_ORDER):
        raise ValueError(
            f"Expected {len(_METRIC_ORDER)} overall dataset-contrast tables, found {len(overall_tables)}."
        )

    return {metric: overall_tables[i] for i, metric in enumerate(_METRIC_ORDER)}


def _is_dataset_contrast_table(df: pd.DataFrame) -> bool:
    """
    True for within-architecture dataset contrast tables:
      must have columns arch, contrast, Δ_mean; and contrast values should reference AE baseline.
    """
    if df is None or df.empty:
        return False

    d = _normalise_contrast_cols(df)
    cols = {str(c).strip().lower() for c in d.columns}
    if not ("arch" in cols and "contrast" in cols):
        return False
    if "Δ_mean" not in d.columns:
        # after canonicalization, it should be exactly "Δ_mean"
        return False

    # Baseline filter: dataset contrasts should include AE
    cvals = d["contrast"].astype(str)
    return cvals.str.contains(r"\bAE\b", case=False, na=False).any()


def _is_overall_dataset_contrast_table(df: pd.DataFrame) -> bool:
    """
    True for marginal dataset contrast tables:
      must have scope, contrast, mean_original
      and scope text contains 'avg over arch' / 'avg over architectures'.
    """
    if df is None or df.empty:
        return False

    d = _normalise_contrast_cols(df)
    cols = {str(c).strip().lower() for c in d.columns}
    if not ("scope" in cols and "contrast" in cols and "mean_original" in cols):
        return False

    scope_vals = d["scope"].astype(str).str.lower()
    ok_scope = scope_vals.str.contains(
        r"avg\.?\s*over\s*arch|avg\.?\s*over\s*architectures|averag(ed)?\s*over\s*arch|averag(ed)?\s*over\s*architectures",
        na=False
    ).any()
    if not ok_scope:
        return False

    # Baseline filter: dataset contrasts should include AE
    cvals = d["contrast"].astype(str)
    return cvals.str.contains(r"\bAE\b", case=False, na=False).any()


def _prepare_dataset_contrast_df(df: pd.DataFrame) -> pd.DataFrame:
    d = _normalise_contrast_cols(df).copy()

    if "arch" in d.columns:
        d["arch"] = (
            d["arch"].astype(str)
             .str.replace("Terramind", "TM", regex=False)
             .str.strip()
        )

    for col in ["Δ_mean", "Δ_lo", "Δ_hi"] + _ROPE_PROB_COLS:
        if col in d.columns:
            d[col] = _to_numeric(d[col])

    return _add_dataset_contrast_clean(d).reset_index(drop=True)


def _prepare_overall_dataset_contrast_df(df: pd.DataFrame) -> pd.DataFrame:
    d = _normalise_contrast_cols(df).copy()
    for col in ["mean_original", "hdi_lo_original", "hdi_hi_original"] + _ROPE_PROB_COLS:
        if col in d.columns:
            d[col] = _to_numeric(d[col])
    return _add_dataset_contrast_clean(d).reset_index(drop=True)


# ============================================================================
# ── SECTION 5: Multi-metric composite ranking ────────────────────────────────
# ============================================================================

@dataclass
class CompositeRankingResult:
    """
    Container for all tables extracted from the 'Overall multi-metric ranking' section.

    Attributes
    ----------
    groups : pd.DataFrame
        Main 9-combination ranking table. Columns include:
          group, Pr(rank=1) … Pr(rank=9), E[rank], Pr(best), Pr(top2),
          rank_ci90 (str), entropy_rank, N_eff_winners,
          Pr(Δ>ROPE), Pr(|Δ|<ROPE), Pr(Δ<-ROPE),
          runner_up (str, NaN for non-top groups),
          Pr(win_vs_runnerup), Pr(tie_vs_runnerup), draw_index
        All columns except group, rank_ci90, and runner_up are float.

    arch_marginal : pd.DataFrame
        Marginal architecture ranking (3 rows: one per architecture). Columns:
          group, Pr(rank=1), Pr(rank=2), Pr(rank=3),
          E[rank], Pr(best), Pr(Δ>ROPE), Pr(|Δ|<ROPE), Pr(Δ<-ROPE)

    dataset_marginal : pd.DataFrame
        Marginal dataset/sensor ranking (3 rows: AE, PS, S2). Same columns
        as arch_marginal.

    weights : pd.DataFrame
        Metric weights used in composite scoring. Columns: metric, weight.
        Metric names are normalised to match _METRIC_ORDER where possible:
          IoU, Dice, NSD, Epistemic, Aleatoric.
    """
    groups: pd.DataFrame
    arch_marginal: pd.DataFrame
    dataset_marginal: pd.DataFrame
    weights: pd.DataFrame


# Columns in the main groups table that are numeric (all except these string cols)
_GROUPS_STRING_COLS = {"group", "rank_ci90", "runner_up"}

# Canonical metric name map for the weights table
_WEIGHT_METRIC_MAP = {
    "iou":                          "IoU",
    "dice_coef":                    "Dice",
    "dice":                         "Dice",
    "normalized_surface_distance":  "NSD",
    "nsd":                          "NSD",
    "mean_epistemic_uncertainty":   "Epistemic",
    "epistemic":                    "Epistemic",
    "mean_aleatoric_uncertainty":   "Aleatoric",
    "aleatoric":                    "Aleatoric",
}


def extract_composite_ranking(html_path: str | Path) -> CompositeRankingResult:
    """
    Extract all four tables from the 'Overall multi-metric ranking' section of the
    stats report and return them as a CompositeRankingResult.

    The function identifies each table by its structural signature rather than
    by position, so it is robust to changes in table order elsewhere in the report.

    Parameters
    ----------
    html_path : str or Path
        Path to the stats_report.html file.

    Returns
    -------
    CompositeRankingResult
        See class docstring for field descriptions.

    Raises
    ------
    FileNotFoundError
        If html_path does not exist.
    ValueError
        If any of the four expected tables cannot be identified.
    """
    html_path = Path(html_path)
    if not html_path.exists():
        raise FileNotFoundError(f"HTML report not found: {html_path}")

    tables = _load_all_tables(html_path)

    groups_df       : Optional[pd.DataFrame] = None
    arch_marginal   : Optional[pd.DataFrame] = None
    dataset_marginal: Optional[pd.DataFrame] = None
    weights_df      : Optional[pd.DataFrame] = None

    for df in tables:
        if df is None or df.empty:
            continue

        df = _flatten_columns(df)
        cols_lower = {str(c).lower().strip() for c in df.columns}
        first_col  = str(df.columns[0]).strip()

        # ── Main 9-combination ranking table ──────────────────────────────────
        # Signature: has 'group' and 'Pr(rank=1)' … 'Pr(rank=9)' (nine rank cols),
        # plus draw-diagnostic cols (runner_up, draw_index).
        if (
            groups_df is None
            and first_col == "group"
            and _has_rank_cols(df, n=9)
            and "draw_index" in cols_lower
            and len(df) == 9
        ):
            groups_df = _prepare_groups_table(df)
            continue

        # ── Marginal architecture ranking ─────────────────────────────────────
        # Signature: has 'group' and exactly 3 rank cols (Pr(rank=1/2/3)),
        # 3 rows, and group values are architecture names (Swin / U-Net / Terramind).
        if (
            arch_marginal is None
            and first_col == "group"
            and _has_rank_cols(df, n=3)
            and len(df) == 3
            and _group_values_match(df, {"swin", "u-net", "unet", "terramind", "tm"})
        ):
            arch_marginal = _prepare_marginal_table(df)
            continue

        # ── Marginal dataset/sensor ranking ───────────────────────────────────
        # Signature: same shape as arch_marginal but group values are sensor names
        # (PS / AE / S2).
        if (
            dataset_marginal is None
            and first_col == "group"
            and _has_rank_cols(df, n=3)
            and len(df) == 3
            and _group_values_match(df, {"ps", "ae", "s2",
                                         "planetscope", "aerial", "sentinel-2", "sentinel2"})
        ):
            dataset_marginal = _prepare_marginal_table(df)
            continue

        # ── Weights table ─────────────────────────────────────────────────────
        # Signature: exactly two columns 'metric' and 'weight', 5 rows.
        if (
            weights_df is None
            and set(c.lower().strip() for c in df.columns) == {"metric", "weight"}
            and len(df) == 5
        ):
            weights_df = _prepare_weights_table(df)
            continue

    # ── Validate all four were found ──────────────────────────────────────────
    missing = []
    if groups_df is None:
        missing.append("main groups ranking table (9 rows, 9 rank cols, draw_index)")
    if arch_marginal is None:
        missing.append("marginal architecture ranking table (3 rows, arch names)")
    if dataset_marginal is None:
        missing.append("marginal dataset/sensor ranking table (3 rows, sensor names)")
    if weights_df is None:
        missing.append("metric weights table (5 rows, metric/weight cols)")

    if missing:
        raise ValueError(
            "extract_composite_ranking: could not identify the following tables:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    return CompositeRankingResult(
        groups=groups_df,
        arch_marginal=arch_marginal,
        dataset_marginal=dataset_marginal,
        weights=weights_df,
    )


# ── Private helpers for composite ranking extraction ─────────────────────────

def _has_rank_cols(df: pd.DataFrame, n: int) -> bool:
    """Return True if df has at least n columns matching 'Pr(rank=<k>)'."""
    pattern = re.compile(r"Pr\(rank=\d+\)", re.IGNORECASE)
    count = sum(1 for c in df.columns if pattern.search(str(c)))
    return count >= n


def _group_values_match(df: pd.DataFrame, valid_names: set) -> bool:
    """
    Return True if every value in the first column (lowercased, stripped)
    matches one of the provided valid_names.
    """
    vals = df.iloc[:, 0].astype(str).str.lower().str.strip().tolist()
    return all(v in valid_names for v in vals)


def _prepare_groups_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast all numeric columns to float, leaving group / rank_ci90 / runner_up as str.
    Normalise unicode minus signs in rank_ci90.
    """
    out = df.copy().reset_index(drop=True)

    # Normalise column names: unescape HTML entities (e.g. &lt; → <)
    out.columns = [htmllib.unescape(str(c)) for c in out.columns]

    for col in out.columns:
        if col in _GROUPS_STRING_COLS:
            out[col] = out[col].astype(str).str.replace("−", "-", regex=False)
        else:
            out[col] = _to_numeric(out[col])

    return out


def _prepare_marginal_table(df: pd.DataFrame) -> pd.DataFrame:
    """Cast all columns except 'group' to float."""
    out = df.copy().reset_index(drop=True)
    out.columns = [htmllib.unescape(str(c)) for c in out.columns]

    for col in out.columns:
        if col != "group":
            out[col] = _to_numeric(out[col])

    return out


def _prepare_weights_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise metric names to match _METRIC_ORDER labels and cast weight to float.
    """
    out = df.copy().reset_index(drop=True)
    out.columns = [c.lower().strip() for c in out.columns]  # → ['metric', 'weight']

    out["weight"] = _to_numeric(out["weight"])
    out["metric"] = (
        out["metric"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda x: _WEIGHT_METRIC_MAP.get(x, x))
    )

    return out


# ============================================================================
# ── CLI / debug ───────────────────────────────────────────────────────────────
# ============================================================================

if __name__ == "__main__":
    import sys

    html_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/isipd/projects/p_planetdw/git/DriftwoodMappingBenchmark/analysis/stats_report.html"
    )

    # ── Factorial effects ──────────────────────────────────────────────────────
    print("=" * 60)
    print("FACTORIAL EFFECTS")
    print("=" * 60)

    data = extract_factorial_effects(html_path)
    print(f"\nExtracted data for {len(data)} metrics:\n")
    for metric, df in data.items():
        print(f"{metric}:")
        print(f"  n rows: {len(df)}")
        print(f"  columns: {list(df.columns)}")
        print(f"  dataset effects: {int((df['effect_type'] == 'dataset').sum())}")
        print(f"  arch effects:    {int((df['effect_type'] == 'arch').sum())}")
        print(f"  interactions:    {int((df['effect_type'] == 'interaction').sum())}")
        print("  ROPE cols present:",
              all(c in df.columns for c in ["Pr(Δ>ROPE)", "Pr(|Δ|<ROPE)", "Pr(Δ<-ROPE)"]))
        preview_cols = [c for c in [
            "param_clean", "Pr(>0)", "Pr(Δ>ROPE)", "Pr(|Δ|<ROPE)", "Pr(Δ<-ROPE)",
            "Δp_mean", "Δp_lo", "Δp_hi",
            "Abs_Δ_mean", "Abs_Δ_lo", "Abs_Δ_hi",
        ] if c in df.columns]
        print(df[preview_cols].head(3).to_string(index=False))
        print()

    # ── Within-dataset arch contrasts ─────────────────────────────────────────
    print("=" * 60)
    print("WITHIN-DATASET ARCHITECTURE CONTRASTS (baseline = U-Net)")
    print("=" * 60)

    arch_w = extract_arch_contrasts(html_path)
    for metric, df in arch_w.items():
        print(f"\n{metric} ({len(df)} rows):")
        print(df.to_string(index=False))

    # ── Overall (marginal) arch contrasts ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("OVERALL (MARGINAL) ARCHITECTURE CONTRASTS (avg over datasets)")
    print("=" * 60)

    arch_o = extract_overall_arch_contrasts(html_path)
    for metric, df in arch_o.items():
        print(f"\n{metric} ({len(df)} rows):")
        print(df.to_string(index=False))

    # ── Within-arch dataset contrasts ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("WITHIN-ARCHITECTURE DATASET CONTRASTS (baseline = AE)")
    print("=" * 60)

    ds_w = extract_dataset_contrasts(html_path)
    for metric, df in ds_w.items():
        print(f"\n{metric} ({len(df)} rows):")
        print(df.to_string(index=False))

    # ── Overall (marginal) dataset contrasts ──────────────────────────────────
    print("\n" + "=" * 60)
    print("OVERALL (MARGINAL) DATASET CONTRASTS (avg over architectures)")
    print("=" * 60)

    ds_o = extract_overall_dataset_contrasts(html_path)
    for metric, df in ds_o.items():
        print(f"\n{metric} ({len(df)} rows):")
        print(df.to_string(index=False))

    # ── Multi-metric composite ranking ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MULTI-METRIC COMPOSITE RANKING")
    print("=" * 60)

    result = extract_composite_ranking(html_path)

    print("\n── Main ranking table (9 combinations) ──")
    preview_cols = [c for c in [
        "group", "Pr(rank=1)", "Pr(rank=2)", "E[rank]", "Pr(best)",
        "Pr(top2)", "rank_ci90", "entropy_rank",
        "Pr(|Δ|<ROPE)", "runner_up", "draw_index",
    ] if c in result.groups.columns]
    print(result.groups[preview_cols].to_string(index=False))

    print("\n── Marginal architecture ranking ──")
    print(result.arch_marginal.to_string(index=False))

    print("\n── Marginal dataset/sensor ranking ──")
    print(result.dataset_marginal.to_string(index=False))

    print("\n── Metric weights ──")
    print(result.weights.to_string(index=False))