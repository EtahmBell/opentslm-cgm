"""
Generate report-ready figures for the OpenTSLM CGM Phase I report.

Run from:
    C:\tslm\opentslm-cgm

Example:
    C:\tslm\OpenTSLM\.venv\Scripts\Activate.ps1
    python .\experiments\generate_phase1_report_figures.py

Outputs:
    artifacts\report_figures\*.png
    artifacts\report_figures\*.pdf
"""

from pathlib import Path
import math
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "report_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

TSQA_TASKS = [
    ("Above 160 mg/dL", 80.6),
    ("Above 180 mg/dL", 83.7),
    ("Sustained >180", 89.2),
    ("Rise >50 / ~30 min", 70.7),
    ("Fall >50 / ~30 min", 69.4),
    ("TIR 70–180 >90%", 50.0),
]

LAYER_LABELS = ["Patch", "T1", "T2", "T3", "T4", "T5", "T6", "Projector"]

TSQA_LAYER_SD = [
    1.22148, 0.49157, 0.55658, 0.48780,
    0.46146, 0.44926, 0.27245, 0.07451,
]

STAGE5_LAYER_SD = [
    0.14984480, 0.05644167, 0.03895127, 0.02033280,
    0.01783406, 0.00039898, 0.00000119, 0.00000073,
]

STAGE_A_ARCHITECTURES = [
    ("Original Stage-5\nT5/T6 + S5 projector", 67.3),
    ("TSQA T5/T6\n+ TSQA projector", 74.5),
    ("TSQA T5/T6\n+ S5 projector", 77.6),
]

STAGE_A_ABLATIONS = [
    ("Real CGM", 77.6),
    ("Zeroed CGM", 50.0),
    ("Opposite-class CGM\nvs original label", 22.4),
]

STAGE_B_EPOCHS = [0, 1, 2, 3, 4, 6, 8]
STAGE_B_GENERATION_BA = [49.0, 77.6, 75.5, 72.4, 77.6, 74.5, 68.4]
STAGE_B_PAIRWISE_BA = [67.3, 77.6, 75.5, 72.4, 77.6, 74.5, 68.4]


def save_figure(fig, stem):
    png_path = OUT_DIR / f"{stem}.png"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def figure_tsqa_multitask():
    labels = [name for name, _ in TSQA_TASKS]
    values = [value for _, value in TSQA_TASKS]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bars = ax.barh(labels, values)
    ax.axvline(50, linestyle="--", linewidth=1)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Validation balanced accuracy (%)")
    ax.set_title("TSQA established multi-task CGM feasibility")
    ax.invert_yaxis()

    for bar, value in zip(bars, values):
        ax.text(
            value + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    save_figure(fig, "figure1_tsqa_multitask_validation")


def figure_layerwise_signal():
    x = list(range(len(LAYER_LABELS)))

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(x, TSQA_LAYER_SD, marker="o", linewidth=2, label="TSQA")
    ax.plot(x, STAGE5_LAYER_SD, marker="o", linewidth=2, label="Stage-5 ECG")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_LABELS)
    ax.set_ylabel("Mean feature variation (SD, log scale)")
    ax.set_title("Stage-5 CGM signal variation collapses in late sensor blocks")
    ax.legend()
    ax.axvspan(4.5, 6.5, alpha=0.08)

    ax.text(
        5.5,
        math.sqrt(STAGE5_LAYER_SD[4] * STAGE5_LAYER_SD[6]),
        "largest Stage-5\nsuppression",
        ha="center",
        va="center",
        fontsize=9,
    )

    fig.tight_layout()
    save_figure(fig, "figure2_layerwise_signal_variation")


def figure_stage_a_architecture():
    labels = [name for name, _ in STAGE_A_ARCHITECTURES]
    values = [value for _, value in STAGE_A_ARCHITECTURES]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar(labels, values)
    ax.axhline(50, linestyle="--", linewidth=1)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Validation balanced accuracy (%)")
    ax.set_title("Late-layer TSQA initialisation improves Stage-5 CGM transfer")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.1f}%",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    save_figure(fig, "figure3_stage_a_architecture_comparison")


def figure_stage_a_ablation():
    labels = [name for name, _ in STAGE_A_ABLATIONS]
    values = [value for _, value in STAGE_A_ABLATIONS]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bars = ax.bar(labels, values)
    ax.axhline(50, linestyle="--", linewidth=1)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Validation balanced accuracy (%)")
    ax.set_title("Stage-A predictions depend on the CGM waveform")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.1f}%",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    save_figure(fig, "figure4_stage_a_signal_ablation")


def figure_stage_b_language():
    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    ax.plot(
        STAGE_B_EPOCHS,
        STAGE_B_GENERATION_BA,
        marker="o",
        linewidth=2,
        label="Generated Yes/No BA",
    )

    ax.plot(
        STAGE_B_EPOCHS,
        STAGE_B_PAIRWISE_BA,
        marker="o",
        linewidth=2,
        linestyle="--",
        label="Pairwise Yes/No BA",
    )

    ax.axhline(50, linestyle=":", linewidth=1)
    ax.set_ylim(40, 100)
    ax.set_xlabel("LoRA language-alignment epoch")
    ax.set_ylabel("Validation balanced accuracy (%)")
    ax.set_title("One epoch of LoRA alignment exposes the frozen CGM representation")
    ax.legend()

    ax.annotate(
        "Best state\n77.6% BA\n100% valid Yes/No",
        xy=(1, 77.6),
        xytext=(2.2, 88),
        arrowprops={"arrowstyle": "->"},
        fontsize=9,
    )

    fig.tight_layout()
    save_figure(fig, "figure5_stage_b1_language_alignment")


def main():
    print(f"Writing report figures to:\n{OUT_DIR}\n")
    figure_tsqa_multitask()
    figure_layerwise_signal()
    figure_stage_a_architecture()
    figure_stage_a_ablation()
    figure_stage_b_language()
    print("\nDone.")


if __name__ == "__main__":
    main()
