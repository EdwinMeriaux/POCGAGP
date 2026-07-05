import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

def bar_small_multiples_per_algo(
    csv_path: str,
    title: str = "Coverage by Algorithm (per-test bars, row order)",
    out_prefix: str = "coverage_bars_small_multiples",
    max_xticks_per_panel: int = 8,
):
    """
    CSV format: each column = algorithm, each row = a test (values 0–100).
    'World complexity rank' is implicit: row_index * 1000.
    Produces small-multiple bar charts (one per algorithm) in ORIGINAL ROW ORDER,
    with lowest rank first on the left.
    """
    # --- load
    df = pd.read_csv(csv_path)
    A = df.select_dtypes(include=[np.number]).clip(0, 100)
    if A.empty:
        raise ValueError("No numeric algorithm columns found.")

    algos = list(A.columns)
    n_algos = len(algos)
    n_tests = len(A)

    # world complexity rank = row_index * 1000 (1-based)
    ranks = (np.arange(1, n_tests + 1) * 1000 + 14000).astype(int)

    # --- style
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,   # remove top grey bar
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # grid layout
    ncols = min(3, n_algos)
    nrows = math.ceil(n_algos / ncols)
    fig_w = 3.4 * ncols + 0.6
    fig_h = 2.8 * nrows + 0.8

    # NOTE: no constrained_layout, we’ll manage margins
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    colors = plt.cm.tab10.colors

    # draw panels
    for k, algo in enumerate(algos):
        r, c = divmod(k, ncols)
        ax = axes[r, c]
        vals = A[algo].values.astype(float)

        # keep ORIGINAL ROW ORDER
        svals = vals
        sranks = ranks

        # bars
        ax.bar(np.arange(n_tests), svals, width=0.85,
               color=colors[k % len(colors)], alpha=0.9, edgecolor="none")

        # mean line + label
        mu = float(np.nanmean(vals))
        ax.axhline(mu, linestyle="--", linewidth=1.1, color="black", alpha=0.8)
        ax.text(len(svals) - 0.5, mu + 1, f"μ={mu:.1f}%",
                ha="right", va="bottom", fontsize=8, color="black")

        # axes
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(f"{algo}")

        # custom x ticks → full numbers (1000, 2000, …) rotated
        step = max(1, n_tests // max_xticks_per_panel)
        tick_pos = np.arange(0, n_tests, step)
        tick_labels = [str(r) for r in sranks[tick_pos]]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")
        ax.set_xlabel("World Complexity Rank")

        # lowest rank = left, increasing right (no inversion)

        if c == 0:
            ax.set_ylabel("Coverage (%)")

    # hide empty panels
    for k in range(n_algos, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    # suptitle + spacing
    if title:
        fig.suptitle(title, y=0.98, fontsize=18)

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])

    # save
    fig.savefig(f"{out_prefix}.png", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.show()
    print(f"Saved {out_prefix}.png and {out_prefix}.pdf")


# example call
bar_small_multiples_per_algo(
    "coverage_data.csv",
    title="Coverage by Algorithm",
    out_prefix="coverage_bars_small_multiples"
)
