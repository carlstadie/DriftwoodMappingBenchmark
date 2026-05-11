"""

Figure 3: Boxplots of model performance by dataset and model.
--> Raw data as it comes from the evaluation
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase

results_dir = "/isipd/projects/p_planetdw/git/DriftwoodMappingBenchmark/analysis/result_safe/dataset_level"
POINTS = False

csv_list = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".csv")]

combined_df = pd.concat((pd.read_csv(p) for p in csv_list), ignore_index=True)

# --- CVD-SAFE COLOR MAP (Paul Tol "bright") ---
MODEL_COLORS = {
    "UNet": "#c27ba0",
    "Swin": "#e06666",
    "Terramind": "#ffd966",
}

# Redundant hatch channel — survives grayscale printing
MODEL_HATCHES = {
    "UNet":      "",      # no hatch
    "Swin":      "",      # no hatch
    "Terramind": "",      # no hatch
}

# --- DISPLAY LABEL MAPPINGS ---
DATASET_DISPLAY = {"AE": "AE", "PS": "PS", "S2": "S2"}
ARCH_DISPLAY    = {"unet": "U-Net", "swin": "Swin", "terramind": "TM", "terraminf": "TM"}


def infer_model(run_name: str) -> str:
    s = str(run_name).lower()
    if "terramind" in s or "terraminf" in s:
        return "Terramind"
    if "unet" in s:
        return "UNet"
    if "swin" in s:
        return "Swin"


def infer_modality(run_name: str) -> str:
    """Extract modality (AE, PS, S2) from run name."""
    s = str(run_name).upper()
    if "AE" in s:
        return "AE"
    if "PS" in s:
        return "PS"
    if "S2" in s:
        return "S2"
    return "unknown"


def adjust_color_brightness(hex_color: str, factor: float) -> str:
    """Adjust brightness of a hex color. Factor > 1 brightens, < 1 darkens."""
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    # Convert to RGB
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    # Adjust brightness
    r = int(min(255, max(0, r * factor)))
    g = int(min(255, max(0, g * factor)))
    b = int(min(255, max(0, b * factor)))
    # Convert back to hex
    return f"#{r:02x}{g:02x}{b:02x}"


def get_modality_color(model: str, modality: str) -> str:
    """Get the model color (consistent for all modalities)."""
    return MODEL_COLORS.get(model, "#dddddd")


def display_run_name(rn: str) -> str:
    """Convert e.g. 'UNetxAE' -> 'U-Net×AE'"""
    if "x" not in rn:
        return rn
    arch_raw, dataset_raw = rn.split("x", 1)
    arch    = ARCH_DISPLAY.get(arch_raw.lower(), arch_raw)
    dataset = DATASET_DISPLAY.get(dataset_raw.upper(), dataset_raw.upper())
    return f"{arch}\u00d7{dataset}"   # × is U+00D7


# Make run_name if it does not exist
if "run_name" not in combined_df.columns:
    def make_run_name(row):
        model    = str(row.get("model", "unknown"))
        modality = str(row.get("modality", "unknown"))
        return f"{model}x{modality}"

    combined_df["run_name"] = combined_df.apply(make_run_name, axis=1)

# Add model column for sorting
combined_df["model"] = combined_df["run_name"].apply(infer_model)

# --- MEANINGFUL ORDERING: by model type, then by run_name alphabetically ---
model_order = ["UNet", "Swin", "Terramind"]
run_names = sorted(
    combined_df["run_name"].dropna().unique(),
    key=lambda rn: (model_order.index(infer_model(rn)), rn)
)

# Pre-compute display labels once
display_labels = [display_run_name(rn) for rn in run_names]


def grouped_data(col):
    return [
        combined_df.loc[combined_df["run_name"] == rn, col].dropna().values
        for rn in run_names
    ]


def colored_boxplot(ax, col, title, points: bool = True):
    data = grouped_data(col)
    bp = ax.boxplot(data, tick_labels=display_labels, patch_artist=True, vert=True,
                    showfliers=False, widths=0.6)

    # Explicit ticks so they render even if autoscale is tight
    ax.set_xticks(np.arange(1, len(run_names) + 1))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=7)

    # Color + hatch each box by model and modality
    for box, rn in zip(bp["boxes"], run_names):
        model = infer_model(rn)
        modality = infer_modality(rn)
        box.set_facecolor(get_modality_color(model, modality))
        box.set_hatch(MODEL_HATCHES.get(model, ""))
        box.set_edgecolor("black")
        box.set_linewidth(0.8)
        box.set_alpha(0.85)

    # Style boxplot elements
    for line in bp["whiskers"]:
        line.set_color("black")
        line.set_linewidth(0.8)
    for line in bp["caps"]:
        line.set_color("black")
        line.set_linewidth(0.8)
    for line in bp["medians"]:
        line.set_color("black")
        line.set_linewidth(1.2)

    # Individual data points (optional)
    if points:
        for i, (rn, values) in enumerate(zip(run_names, data)):
            if len(values) == 0:
                continue
            x_pos = np.random.normal(i + 1, 0.03, size=len(values))
            model = infer_model(rn)
            modality = infer_modality(rn)
            color = get_modality_color(model, modality)
            ax.scatter(x_pos, values, alpha=0.5, s=15,
                       color=color, edgecolors="black", linewidth=0.3, zorder=3)

    ax.set_title(title, fontsize=8, fontweight="bold", pad=8)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=7, pad=2, bottom=True, top=False,
                   labelbottom=True, direction="out", length=4, width=1.0)
    ax.tick_params(axis="y", labelsize=7, pad=2, left=True, right=False,
                   labelleft=True, direction="out", length=4, width=1.0)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.margins(x=0.02)


# Custom legend handler for mini boxplots
class BoxplotHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        box_width  = width * 0.6
        box_height = height * 0.5
        x_center   = width / 2
        y_center   = height / 2

        box = Rectangle(
            (x_center - box_width / 2, y_center - box_height / 2),
            box_width, box_height,
            facecolor=orig_handle.get_facecolor(),
            edgecolor="black",
            hatch=orig_handle.get_hatch(),
            linewidth=0.8,
            alpha=0.85,
            transform=trans,
        )
        median = Line2D(
            [x_center - box_width / 2, x_center + box_width / 2],
            [y_center, y_center],
            color="black", linewidth=1.2, transform=trans,
        )
        whisker_top = Line2D(
            [x_center, x_center],
            [y_center + box_height / 2, y_center + box_height / 2 + height * 0.15],
            color="black", linewidth=0.8, transform=trans,
        )
        whisker_bottom = Line2D(
            [x_center, x_center],
            [y_center - box_height / 2, y_center - box_height / 2 - height * 0.15],
            color="black", linewidth=0.8, transform=trans,
        )
        return [box, median, whisker_top, whisker_bottom]


# --- FIGURE: 2×3 grid (a b c / d e legend) ---
fig = plt.figure(figsize=(16 / 2.54, 13 / 2.54))

gs = fig.add_gridspec(
    2, 3,
    hspace=0.5, wspace=0.35,
    left=0.13, right=0.98, top=0.92, bottom=0.12,
)

ax_a   = fig.add_subplot(gs[0, 0])
ax_b   = fig.add_subplot(gs[0, 1])
ax_c   = fig.add_subplot(gs[0, 2])
ax_d   = fig.add_subplot(gs[1, 0])
ax_e   = fig.add_subplot(gs[1, 1])
ax_leg = fig.add_subplot(gs[1, 2])
ax_leg.axis("off")

colored_boxplot(ax_a, "IoU",                          r"IoU ($\uparrow$)",                          points=POINTS)
colored_boxplot(ax_b, "dice_coef",                    r"DSC ($\uparrow$)",                          points=POINTS)
colored_boxplot(ax_c, "normalized_surface_distance",  r"NSD ($\uparrow$)",                          points=POINTS)
colored_boxplot(ax_d, "mean_epistemic_uncertainty",   r"Uncertainty$_\mathrm{epist.}$ ($\downarrow$)", points=POINTS)
colored_boxplot(ax_e, "mean_aleatoric_uncertainty",   r"Uncertainty$_\mathrm{alea.}$ ($\downarrow$)",  points=POINTS)

# Add y-axis labels to the left column.
for ax in [ax_a, ax_d]:
    ax.set_ylabel("Metric value", fontsize=9, labelpad=10)

fig.align_ylabels([ax_a, ax_d])

# Set y-axis limits for upper plots
ax_a.set_ylim(0, 1)
ax_b.set_ylim(0, 1)
ax_c.set_ylim(0, 1)


# Panel IDs
for ax, sid in zip([ax_a, ax_b, ax_c, ax_d, ax_e],
                   ["(a)", "(b)", "(c)", "(d)", "(e)"]):
    ax.text(-0.15, 1.12, sid, transform=ax.transAxes,
            ha="left", va="top", fontsize=9, fontweight="bold")

# Legend handles — include hatch so handler renders it correctly
legend_handles = [
    Patch(
        facecolor=MODEL_COLORS[m],
        edgecolor="black",
        hatch=MODEL_HATCHES[m],
        label=m,
        alpha=0.85,
        linewidth=0.8,
    )
    for m in model_order if m in MODEL_COLORS
]

ax_leg.legend(
    handles=legend_handles,
    loc="center",
    frameon=False,
    fontsize=8,
    handler_map={Patch: BoxplotHandler()},
    handlelength=2.0,
    handleheight=1.5,
)

OUT_DIR = "./figs/out"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "Figure_3_data.csv")
combined_df.to_csv(OUT_CSV, index=False)
print(f"Saved → {OUT_CSV}")

plt.savefig(os.path.join(OUT_DIR, "Figure_3_input_data_bayes.pdf"), dpi=500, bbox_inches="tight")
print(f"Saved → {os.path.join(OUT_DIR, 'Figure_3_input_data_bayes.pdf')}")
plt.savefig(os.path.join(OUT_DIR, "Figure_3_input_data_bayes.png"), dpi=500, bbox_inches="tight")
print(f"Saved → {os.path.join(OUT_DIR, 'Figure_3_input_data_bayes.png')}")
plt.show()