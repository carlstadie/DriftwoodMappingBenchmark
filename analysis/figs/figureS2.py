import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import geopandas as gpd

# =============================================================================
# RSE / publication-style CONFIG
# =============================================================================
FIG_WIDTH_MM = 180
FIG_MIN_HEIGHT_IN = 3.2
FIG_ROW_HEIGHT_IN = 0.30

FONT_FAMILY = "Arial"
BASE_FONTSIZE = 9
LABEL_FONTSIZE = 9
TICK_FONTSIZE = 8
LEGEND_FONTSIZE = 8

AXES_LINEWIDTH = 0.8
TICK_WIDTH = 0.8
TICK_LENGTH = 3.0

STRIPE_COLOR = "0.92"
STRIPE_LINEWIDTH = 12
STRIPE_ALPHA = 1.0

MARKER_STYLE = "|"
MARKER_SIZE = 12
MARKER_EDGEWIDTH = 1.6

X_TICK_DAY_INTERVAL = 2

OUT_DIR = "./figs/out"
OUT_BASE = "AQC_times_RSE"
SAVE_PNG = True
PNG_DPI = 600
SAVE_PDF = True

LEGEND_LABEL_MAP = {
    "Sentinel-2": "S2",
    "Aerial": "AE",
    "AE": "AE",
}
LEGEND_ORDER = {"AE": 0, "S2": 1, "PS": 2, "Other": 99}

# =============================================================================
# Matplotlib rcParams
# =============================================================================
plt.rcParams.update({
    "font.family": FONT_FAMILY if FONT_FAMILY else "DejaVu Sans",
    "font.size": BASE_FONTSIZE,
    "axes.labelsize": LABEL_FONTSIZE,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
    "axes.linewidth": AXES_LINEWIDTH,
    "xtick.major.width": TICK_WIDTH,
    "ytick.major.width": TICK_WIDTH,
    "xtick.major.size": TICK_LENGTH,
    "ytick.major.size": TICK_LENGTH,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

def mm_to_in(mm: float) -> float:
    return mm / 25.4


# =============================================================================
# Collision-avoidance: shift x only when visually colliding (plot-only)
# =============================================================================
def _spread_1d_with_min_sep(x0, min_sep, lower, upper, n_iter=6, pull=0.35):
    """
    Given 1D positions x0, find adjusted positions x that satisfy spacing >= min_sep.
    Uses forward/backward constraint passes and a gentle pull back toward original positions.
    Works in continuous units (here: matplotlib date numbers in DAYS).
    """
    x0 = np.asarray(x0, dtype=float)
    if x0.size <= 1:
        return x0.copy()

    order = np.argsort(x0)
    x_orig = x0[order]
    x_adj = x_orig.copy()

    for _ in range(n_iter):
        # forward enforce
        for i in range(1, len(x_adj)):
            if x_adj[i] - x_adj[i - 1] < min_sep:
                x_adj[i] = x_adj[i - 1] + min_sep

        # backward enforce
        for i in range(len(x_adj) - 2, -1, -1):
            if x_adj[i + 1] - x_adj[i] < min_sep:
                x_adj[i] = x_adj[i + 1] - min_sep

        # keep within bounds by shifting the whole set (preserves spacings)
        if x_adj[0] < lower:
            x_adj += (lower - x_adj[0])
        if x_adj[-1] > upper:
            x_adj += (upper - x_adj[-1])

        # pull toward original (then next loop re-enforces constraints)
        x_adj = x_adj + pull * (x_orig - x_adj)

    # final hard enforce + bounds
    for i in range(1, len(x_adj)):
        if x_adj[i] - x_adj[i - 1] < min_sep:
            x_adj[i] = x_adj[i - 1] + min_sep
    for i in range(len(x_adj) - 2, -1, -1):
        if x_adj[i + 1] - x_adj[i] < min_sep:
            x_adj[i] = x_adj[i + 1] - min_sep

    if x_adj[0] < lower:
        x_adj += (lower - x_adj[0])
    if x_adj[-1] > upper:
        x_adj += (upper - x_adj[-1])

    out = np.empty_like(x0)
    out[order] = x_adj
    return out


def dodge_colliding_ticks_by_aoi(points: pd.DataFrame, ax, fig,
                                 window_start: pd.Timestamp, window_end: pd.Timestamp,
                                 marker_edgewidth_pts: float,
                                 safety_factor: float = 1.10) -> pd.DataFrame:
    """
    Compute x_plot for each point by AOI:
    - determine min separable distance in *days* based on:
      marker edgewidth in points, axis width in points, and the x-range in days
    - spread points within each AOI to satisfy that min spacing
    """
    p = points.copy()
    p = p.dropna(subset=["AOI", "time"]).copy()
    p["time"] = pd.to_datetime(p["time"], errors="coerce")
    p = p.dropna(subset=["time"]).copy()

    # Need a drawn canvas to get accurate axis width after layout
    fig.canvas.draw()
    ax_box = ax.get_window_extent()  # pixels
    axis_width_points = ax_box.width * 72.0 / fig.dpi

    x_min = mdates.date2num(pd.Timestamp(window_start).to_pydatetime())
    x_max = mdates.date2num(pd.Timestamp(window_end).to_pydatetime())
    x_range_days = float(x_max - x_min)

    # Marker width (points) -> min spacing in days
    # Using safety_factor slightly > 1 avoids "still touching" due to antialiasing.
    min_sep_points = float(marker_edgewidth_pts) * float(safety_factor)
    min_sep_days = (min_sep_points / max(axis_width_points, 1e-9)) * x_range_days

    p["x0_days"] = mdates.date2num(p["time"].dt.to_pydatetime())

    # Apply spreading within each AOI
    x_adj = np.empty(len(p), dtype=float)
    for aoi, g in p.groupby("AOI", sort=False):
        idx = g.index.to_numpy()
        x0 = g["x0_days"].to_numpy()
        x1 = _spread_1d_with_min_sep(x0, min_sep_days, x_min, x_max)
        x_adj[p.index.get_indexer(idx)] = x1  # fill aligned to p order

    # Convert adjustment back into datetimes without timezone weirdness:
    dx_days = x_adj - p["x0_days"].to_numpy()
    p["x_plot"] = p["time"] + pd.to_timedelta(dx_days, unit="D")
    return p


# =============================================================================
# --- Load ---
# =============================================================================
csv_path = r"\\smb.isipd.dmawi.de\projects\p_planetdw\data\methods_test\aois\S2_GEE_export_summary.csv"
ps_csv_path = r"\\smb.isipd.dmawi.de\projects\p_planetdw\data\methods_test\aois\PSScene_used_selection_summary.csv"
gpkg_path = r"\\smb.isipd.dmawi.de\projects\p_planetdw\data\methods_test\auxilliary_data\aoi_ext.gpkg"

df = pd.read_csv(csv_path)
ps_df = pd.read_csv(ps_csv_path)

# guard against duplicate columns from messy CSV exports
df = df.loc[:, ~df.columns.duplicated()].copy()
ps_df = ps_df.loc[:, ~ps_df.columns.duplicated()].copy()

# GeoPandas: avoid ignore_geometry=True (not supported in many versions)
aerialaq = gpd.read_file(gpkg_path)
aerialaq = aerialaq.drop(columns="geometry", errors="ignore")

# =============================================================================
# --- Aerial (GPKG): build datetime ---
# =============================================================================
aerialaq["Date"] = aerialaq["layer"].astype(str).str.split("_").str[2]
aerialaq["Date"] = pd.to_datetime(aerialaq["Date"], format="%Y%m%d", errors="coerce")
aerialaq["time"] = aerialaq["time"].astype(str).str.strip()

aerialaq["DateTime"] = pd.to_datetime(
    aerialaq["Date"].dt.strftime("%Y-%m-%d") + " " + aerialaq["time"],
    utc=True,
    errors="coerce",
).dt.tz_convert(None)

# =============================================================================
# --- S2/Other CSV datetime ---
# =============================================================================
df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_convert(None)

# --- PS datetime ---
ps_df["Date"] = pd.to_datetime(ps_df["Acquired"], utc=True, errors="coerce").dt.tz_convert(None)
ps_df["Sensor"] = "PS"

df["DateDeltaDays"] = pd.to_numeric(df.get("DateDeltaDays", 0), errors="coerce").fillna(0)
ps_df["DateDeltaDays"] = pd.to_numeric(ps_df.get("DateDeltaDays", 0), errors="coerce").fillna(0)

def infer_sensor(row):
    s = " ".join(str(row.get(c, "")) for c in ["ExportTask", "SystemIndex"]).upper()
    if ("_AE" in s) or ("AE_" in s) or (" AE " in s):
        return "AE"
    if "AERIAL" in s:
        return "Aerial"
    if ("S2" in s) or ("SENTINEL-2" in s) or ("SENTINEL2" in s):
        return "Sentinel-2"
    return "Other"

if "Sensor" not in df.columns:
    df["Sensor"] = df.apply(infer_sensor, axis=1)

# Drop Aerial coming from the CSV: only use Aerial from GPKG
df = df[df["Sensor"] != "Aerial"].copy()

df["plot_date"] = df["Date"]
ps_df["plot_date"] = ps_df["Date"]

# =============================================================================
# Window
# =============================================================================
window_start = pd.Timestamp("2025-07-01 00:00")
window_end   = pd.Timestamp("2025-08-31 23:59")

df_plot = df[df["plot_date"].between(window_start, window_end)].copy()
ps_df_plot = ps_df[ps_df["plot_date"].between(window_start, window_end)].copy()

non_aerial_plot = pd.concat([df_plot, ps_df_plot], ignore_index=True)

aerial_points = aerialaq[aerialaq["DateTime"].between(window_start, window_end)].copy()
aerial_points["Sensor"] = "Aerial"

# =============================================================================
# AOI ordering (earliest aerial first, then earliest non-aerial)
# =============================================================================
aois_in_plot = non_aerial_plot["AOI"].dropna().unique()

aerial_min = (
    aerial_points[aerial_points["target_name"].isin(aois_in_plot)]
    .dropna(subset=["DateTime", "target_name"])
    .groupby("target_name")["DateTime"]
    .min()
)
aois_with_aerial = aerial_min.sort_values().index.tolist()

remaining_aois = [a for a in aois_in_plot if a not in aois_with_aerial]
if remaining_aois:
    remaining_min = (
        non_aerial_plot[non_aerial_plot["AOI"].isin(remaining_aois)]
        .groupby("AOI")["plot_date"]
        .min()
        .sort_values()
        .index
        .tolist()
    )
else:
    remaining_min = []

aoi_order = aois_with_aerial + remaining_min
aoi_to_y = {aoi: i for i, aoi in enumerate(aoi_order)}

# =============================================================================
# Figure (publication sizing)
# =============================================================================
fig_w_in = mm_to_in(FIG_WIDTH_MM)
fig_h_in = max(FIG_MIN_HEIGHT_IN, FIG_ROW_HEIGHT_IN * len(aoi_order))
fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))

# stripes (old style)
for aoi in aoi_order:
    y = aoi_to_y[aoi]
    ax.hlines(
        y=y,
        xmin=window_start,
        xmax=window_end,
        color=STRIPE_COLOR,
        linewidth=STRIPE_LINEWIDTH,
        alpha=STRIPE_ALPHA,
        zorder=1,
    )

# Axes formatting BEFORE collision math (so layout/width are correct)
ax.set_yticks([aoi_to_y[a] for a in aoi_order])
ax.set_yticklabels(aoi_order)
ax.invert_yaxis()
ax.set_xlim(window_start, window_end)

day_locator = mdates.DayLocator(interval=X_TICK_DAY_INTERVAL)
ax.xaxis.set_major_locator(day_locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(day_locator))

ax.set_xlabel("Acquisition datetime (UTC)")
ax.set_ylabel("AOI")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(direction="out")

# Tight layout now so axis width is final-ish (important for collision spacing)
fig.tight_layout()

# =============================================================================
# Build unified points table and dodge only when visually colliding
# =============================================================================
points_non = pd.DataFrame({
    "AOI": non_aerial_plot["AOI"].astype(str),
    "Sensor": non_aerial_plot["Sensor"].fillna("Other").astype(str),
    "time": non_aerial_plot["plot_date"],
}).dropna(subset=["AOI", "time"])

points_air = pd.DataFrame({
    "AOI": aerial_points["target_name"].astype(str),
    "Sensor": "Aerial",
    "time": aerial_points["DateTime"],
}).dropna(subset=["AOI", "time"])

points = pd.concat([points_non, points_air], ignore_index=True)

# Keep only AOIs actually in plot order
points = points[points["AOI"].isin(aoi_to_y)].copy()

# Compute x_plot with collision-aware dodging per AOI
points = dodge_colliding_ticks_by_aoi(
    points, ax=ax, fig=fig,
    window_start=window_start, window_end=window_end,
    marker_edgewidth_pts=MARKER_EDGEWIDTH,
    safety_factor=1.10,  # tweak 1.00..1.30 if you want more/less separation
)

# =============================================================================
# Plot points (old: same y for all sensors)
# =============================================================================
sensor_colors = {
    "Sentinel-2": "#eaa975",
    "Aerial": "#9dc2d5",
    "AE": "#9dc2d5",
    "PS": "#9fcd9c",
    "Other": "#7f7f7f",
}

legend_done = set()
for sensor, g in points.groupby("Sensor", sort=False):
    color = sensor_colors.get(sensor, sensor_colors["Other"])
    label = sensor if sensor not in legend_done else "_nolegend_"
    y = g["AOI"].map(aoi_to_y).to_numpy()

    ax.plot(
        g["x_plot"].to_numpy(),
        y,
        marker=MARKER_STYLE,
        markersize=MARKER_SIZE,
        markeredgewidth=MARKER_EDGEWIDTH,
        linestyle="None",
        markerfacecolor="none",
        markeredgecolor=color,
        alpha=0.95,
        zorder=3,
        label=label,
    )
    legend_done.add(sensor)

# =============================================================================
# Legend (rename labels in legend only)
# =============================================================================
handles, labels = ax.get_legend_handles_labels()
labels = [LEGEND_LABEL_MAP.get(l, l) for l in labels]

uniq = {}
for h, l in zip(handles, labels):
    if l not in uniq:
        uniq[l] = h

sorted_labels = sorted(uniq.keys(), key=lambda x: LEGEND_ORDER.get(x, 50))
sorted_handles = [uniq[l] for l in sorted_labels]

leg = ax.legend(
    sorted_handles,
    sorted_labels,
    title="Sensor",
    loc="upper right",
    frameon=True,
    fancybox=True,
    framealpha=0.9,
    edgecolor="black",
    facecolor="white",
)
if leg is not None:
    leg.get_frame().set_linewidth(0.8)

# =============================================================================
# Save
# =============================================================================
os.makedirs(OUT_DIR, exist_ok=True)
if SAVE_PNG:
    fig.savefig(os.path.join(OUT_DIR, f"{OUT_BASE}.png"), dpi=PNG_DPI)
if SAVE_PDF:
    fig.savefig(os.path.join(OUT_DIR, f"{OUT_BASE}.pdf"))

plt.show()

# =============================================================================
# Stats: nearest aerial per acquisition (non-aerial only)
# =============================================================================
tmp_aerial = aerial_points.dropna(subset=["DateTime", "target_name"])[["target_name", "DateTime"]].copy()
tmp_aerial = tmp_aerial.rename(columns={"target_name": "AOI", "DateTime": "AerialDateTime"})

aerial_times_by_aoi = {
    aoi: times.sort_values().to_list()
    for aoi, times in tmp_aerial.groupby("AOI")["AerialDateTime"]
}

table = non_aerial_plot[["AOI", "Sensor", "plot_date"]].copy().sort_values(["AOI", "plot_date"])

def nearest_aerial(row):
    aoi = row["AOI"]
    t = row["plot_date"]
    times = aerial_times_by_aoi.get(aoi)
    if (times is None) or (len(times) == 0) or pd.isna(t):
        return pd.Series({"NearestAerial": pd.NaT, "Diff_hours": np.nan})

    diffs_sec = np.array([(t - a).total_seconds() for a in times])  # signed
    idx = int(np.argmin(np.abs(diffs_sec)))
    nearest_time = times[idx]
    diff_hours = diffs_sec[idx] / 3600.0
    return pd.Series({"NearestAerial": nearest_time, "Diff_hours": diff_hours})

table[["NearestAerial", "Diff_hours"]] = table.apply(nearest_aerial, axis=1)
table["Diff_hours"] = pd.to_numeric(table["Diff_hours"], errors="coerce")
table["Diff_hours_abs"] = table["Diff_hours"].abs()
table = table.rename(columns={"plot_date": "AcqDateTime"})

def sensor_vs_aerial_stats(table_df: pd.DataFrame, sensor_name: str):
    t = table_df[(table_df["Sensor"] == sensor_name) & table_df["NearestAerial"].notna()].copy()
    t = t.dropna(subset=["Diff_hours"])
    if t.empty:
        return None, None

    per_aoi = (
        t.groupby("AOI")
         .agg(
             count=("Diff_hours", "count"),
             mean_signed=("Diff_hours", "mean"),
             median_signed=("Diff_hours", "median"),
             min_signed=("Diff_hours", "min"),
             max_signed=("Diff_hours", "max"),
             mean_abs=("Diff_hours_abs", "mean"),
             median_abs=("Diff_hours_abs", "median"),
             min_abs=("Diff_hours_abs", "min"),
             max_abs=("Diff_hours_abs", "max"),
         )
         .reset_index()
         .sort_values("AOI")
    )
    overall = t.agg(
        count=("Diff_hours", "count"),
        mean_signed=("Diff_hours", "mean"),
        median_signed=("Diff_hours", "median"),
        min_signed=("Diff_hours", "min"),
        max_signed=("Diff_hours", "max"),
        mean_abs=("Diff_hours_abs", "mean"),
        median_abs=("Diff_hours_abs", "median"),
        min_abs=("Diff_hours_abs", "min"),
        max_abs=("Diff_hours_abs", "max"),
    )
    return per_aoi, overall

pd.set_option("display.float_format", lambda x: f"{x:6.2f}")

stats_s2_vs_ae, overall_s2_vs_ae = sensor_vs_aerial_stats(table, "Sentinel-2")
stats_ps_vs_ae, overall_ps_vs_ae = sensor_vs_aerial_stats(table, "PS")

print("\n========================================================")
print("S2 vs nearest AE (Aerial): per AOI stats (hours)")
print("  signed: sensor_time - aerial_time  (+: sensor after AE)")
print("  abs: |sensor_time - aerial_time|")
print("========================================================")
if stats_s2_vs_ae is None:
    print("No S2 rows with a nearest AE found in the selected window.")
else:
    print(stats_s2_vs_ae.to_string(index=False))
    print("\nS2 vs nearest AE: OVERALL stats")
    print(overall_s2_vs_ae)

print("\n========================================================")
print("PS vs nearest AE (Aerial): per AOI stats (hours)")
print("  signed: sensor_time - aerial_time  (+: sensor after AE)")
print("  abs: |sensor_time - aerial_time|")
print("========================================================")
if stats_ps_vs_ae is None:
    print("No PS rows with a nearest AE found in the selected window.")
else:
    print(stats_ps_vs_ae.to_string(index=False))
    print("\nPS vs nearest AE: OVERALL stats")
    print(overall_ps_vs_ae)

print("\n========================================================")
print("Per-acquisition nearest AE table (raw)")
print("========================================================")
print(
    table[["AOI", "Sensor", "AcqDateTime", "NearestAerial", "Diff_hours", "Diff_hours_abs"]]
    .to_string(index=False)
)
