import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def line_plot_all_algos(
    csv_path: str,
    title: str = "Steps by Algorithm (per-test, row order)",
    out_prefix: str = "steps_lineplot",
    max_xticks: int = 12,
):
    """
    CSV format: each column = algorithm, each row = a test (values are step counts).
    'World complexity rank' is implicit: row_index * 1000 (+14000 offset).
    Produces a single line plot with all algorithms overlaid, different markers/colors.
    """
    # --- load
    df = pd.read_csv(csv_path)
    A = df.select_dtypes(include=[np.number]).clip(0, 32000)
    if A.empty:
        raise ValueError("No numeric algorithm columns found.")

    algos = list(A.columns)
    n_tests = len(A)

    # world complexity rank = row_index * 1000 + 14000
    ranks = (np.arange(1, n_tests + 1) * 1000 + 14000).astype(int)

    # --- style
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors = plt.cm.tab10.colors
    markers = ["o", "s", "D", "^", "v", ">", "<", "P", "X", "*"]

    # draw lines
    for k, algo in enumerate(algos):
        vals = A[algo].values.astype(float)
        ax.plot(
            ranks, vals,
            label=algo,
            color=colors[k % len(colors)],
            marker=markers[k % len(markers)],
            markersize=4,
            linewidth=1.4,
            alpha=0.9,
        )

    # axes
    ax.set_ylim(0, 32000)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("World Complexity Rank")
    ax.set_ylabel("Steps")

    # x ticks (spaced nicely, full numbers with commas)
    step = max(1, n_tests // max_xticks)
    tick_pos = np.arange(0, n_tests, step)
    tick_labels = [f"{r:,}" for r in ranks[tick_pos]]
    ax.set_xticks(ranks[tick_pos])
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")

    # --- title + legend stacking ---
    fig.suptitle(title, y=0.825)   # place title just a bit above legend
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),  # legend just under title
        ncol=len(algos),
        frameon=False
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.86])  # leave space for both
    fig.savefig(f"{out_prefix}.png", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.show()
    print(f"Saved {out_prefix}.png and {out_prefix}.pdf")

line_plot_all_algos(
    "steps_data.csv",
    title="Steps by Algorithm",
    out_prefix="steps_lineplot"
)