"""Generate simple PDF figures for the thesis when Visio exports are unavailable."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "csuthesis_v1_0_5" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def add_box(ax, xy, text, width=2.2, height=0.7, fc="#eef3ff"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.0,
        edgecolor="#334155",
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)


def framework_figure() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        ((0.3, 3.2), "Review text"),
        ((2.8, 3.2), "Hybrid encoder"),
        ((5.3, 3.2), "Block router"),
        ((7.8, 3.2), "Sentiment head"),
        ((7.8, 2.1), "Recoverability head"),
        ((0.3, 1.0), "Counterfactual condition"),
        ((2.8, 1.0), "Offline GAN"),
        ((5.3, 1.0), "Counterfactual review"),
        ((7.8, 1.0), "Intervention losses"),
    ]
    for pos, text in boxes:
        add_box(ax, pos, text)
    for start, end in [((2.5, 3.55), (2.8, 3.55)), ((5.0, 3.55), (5.3, 3.55)), ((7.5, 3.55), (7.8, 3.55))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.0))
    ax.text(4.0, 4.5, "Online classifier", ha="center", fontsize=10, weight="bold")
    ax.text(4.0, 0.2, "Offline counterfactual augmentation", ha="center", fontsize=10, weight="bold")
    save(fig, "fig_framework.pdf")


def block_sparse_figure() -> None:
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    for i in range(8):
        fc = "#cfe2ff" if i in (1, 5) else "#f3f4f6"
        add_box(ax, (0.4 + i * 1.1, 1.0), f"B{i+1}", width=0.8, height=0.55, fc=fc)
    add_box(ax, (3.2, 2.0), "Block router", width=2.0, height=0.6, fc="#fff4e6")
    ax.text(5.0, 0.35, "local window + same-block + routed global blocks", ha="center", fontsize=9)
    save(fig, "fig_block_sparse.pdf")


def counterfactual_figure() -> None:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.axis("off")
    add_box(ax, (0.3, 1.2), "Factual review")
    add_box(ax, (2.8, 1.2), "Conditional GAN")
    add_box(ax, (5.3, 1.2), "Counterfactual review")
    add_box(ax, (7.8, 1.2), "Shared classifier")
    add_box(ax, (5.3, 0.1), "Cls + CF + Int + Rec losses", width=3.0, fc="#fff4e6")
    save(fig, "fig_counterfactual.pdf")


def managerial_figure() -> None:
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3)
    ax.axis("off")
    add_box(ax, (0.2, 1.2), "Incoming\nreviews")
    add_box(ax, (2.5, 1.2), "Evidence\nlocalization")
    add_box(ax, (4.8, 1.2), "Recoverability\nestimation")
    add_box(ax, (7.1, 1.2), "Complaint\ntriage")
    add_box(ax, (9.0, 1.2), "Service\nrecovery")
    for x in (2.2, 4.5, 6.8, 8.9):
        ax.add_patch(FancyArrowPatch((x, 1.55), (x + 0.25, 1.55), arrowstyle="->", mutation_scale=12))
    save(fig, "fig_managerial_flow.pdf")


if __name__ == "__main__":
    framework_figure()
    block_sparse_figure()
    counterfactual_figure()
    managerial_figure()
