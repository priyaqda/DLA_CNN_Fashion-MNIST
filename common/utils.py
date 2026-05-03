"""
common/utils.py — Helper functions untuk plotting dan logging.
"""

import numpy as np
import matplotlib.pyplot as plt
from . import config


def plot_bar(labels, values, title, ylabel, filename=None, color="#6d28d9"):
    """Bar chart sederhana."""
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)
    bars = ax.bar(labels, values, color=color, edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        print(f"[saved] {filename}")
    plt.show()


def plot_comparison(labels, series_dict, title, ylabel, filename=None):
    """Grouped bar chart untuk perbandingan."""
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE, dpi=config.FIGURE_DPI)
    x = np.arange(len(labels))
    n = len(series_dict)
    width = 0.8 / n

    colors = ["#6d28d9", "#059669", "#d97706", "#dc2626", "#2563eb"]
    for i, (name, values) in enumerate(series_dict.items()):
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=name, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        print(f"[saved] {filename}")
    plt.show()


def plot_heatmap(data, title, xlabel="", ylabel="", filename=None, cmap="YlOrRd"):
    """Heatmap 2D."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=config.FIGURE_DPI)
    im = ax.imshow(data, cmap=cmap, aspect="auto")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        print(f"[saved] {filename}")
    plt.show()


def print_table(headers, rows, title=""):
    """Print formatted ASCII table."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
                  for i, h in enumerate(headers)]

    header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    print(f"  {header_line}")
    print(f"  {'-+-'.join('-' * w for w in col_widths)}")

    for row in rows:
        line = " | ".join(f"{str(v):<{col_widths[i]}}" for i, v in enumerate(row))
        print(f"  {line}")
    print()
