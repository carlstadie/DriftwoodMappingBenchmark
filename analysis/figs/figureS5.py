"""
figureS5.py
Supplementary figure: Posterior predictive checks across all metrics.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
ASSETS_DIR = THIS_DIR.parent / "stats_report_assets"
OUT_DIR = THIS_DIR / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# STYLE (aligned with other figure scripts)
# -----------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


# -----------------------------------------------------------------------------
# PANELS
# -----------------------------------------------------------------------------
PANELS = [
    ("iou", "IoU", "a"),
    ("dice-coef", "Dice coefficient", "b"),
    ("normalized-surface-distance", "Normalized Surface Distance", "c"),
    ("mean-epistemic-uncertainty", "Mean epistemic uncertainty", "d"),
    ("mean-aleatoric-uncertainty", "Mean aleatoric uncertainty", "e"),
]


def _draw_missing_panel(ax, panel_title: str, missing_path: Path) -> None:
    ax.set_axis_off()
    ax.text(
        0.5,
        0.55,
        "Missing source image",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#7A1F1F",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.4,
        str(missing_path.name),
        ha="center",
        va="center",
        fontsize=8,
        color="#7A1F1F",
        transform=ax.transAxes,
    )
    ax.set_title(panel_title, pad=8, fontweight="bold")


def main() -> None:
    # About one page in portrait orientation
    fig = plt.figure(figsize=(190 / 25.4, 250 / 25.4))
    axes = fig.subplots(3, 2)
    axes_flat = axes.ravel()

    for idx, (slug, label, letter) in enumerate(PANELS):
        ax = axes_flat[idx]
        img_path = ASSETS_DIR / f"post_pred_{slug}.png"
        title = f"({letter}) {label}"

        if img_path.exists():
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.set_axis_off()
            ax.set_title(title, pad=8, fontweight="bold")
        else:
            _draw_missing_panel(ax, title, img_path)

    axes_flat[-1].set_axis_off()

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.975], h_pad=1.0, w_pad=0.8)

    out_png = OUT_DIR / "figureS5_posterior_predictive_checks.png"
    out_pdf = OUT_DIR / "figureS5_posterior_predictive_checks.pdf"
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
