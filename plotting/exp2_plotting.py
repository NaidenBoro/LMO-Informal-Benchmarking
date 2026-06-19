import matplotlib.pyplot as plt


def plot_distribution_summary(summary_df, gamma_matrix=None, filename="figures/individual_gamma_distribution.png"):
    m = summary_df["m_Dropped"].to_numpy()

    plt.figure(figsize=(10, 7))

    if gamma_matrix is not None:
        for individual_gamma in gamma_matrix.T:
            plt.plot(m, individual_gamma, color="gray", alpha=0.018, linewidth=0.65, zorder=1)

    plt.plot(m, summary_df["Median"], marker="o", linewidth=3.0, markersize=7, color="#D55E00", label="Median", zorder=3)
    plt.plot(m, summary_df["P90"], marker="s", linewidth=3.0, markersize=7, color="#7B1FA2", label="90th percentile", zorder=3)
    plt.plot(m, summary_df["P99"], marker="^", linewidth=3.0, markersize=7, color="#0057B8", label="99th percentile", zorder=3)
    plt.plot(m, summary_df["Max"], marker="D", linewidth=3.3, markersize=7, color="#C62828", label="Empirical maximum", zorder=4)
    plt.plot(
        m,
        summary_df["Theoretical"],
        linestyle="--",
        linewidth=3.5,
        color="black",
        label="Theoretical maximum",
        zorder=4,
    )

    plt.yscale("log")
    plt.xlabel("Subset size $m$", fontweight="bold")
    plt.ylabel(r"Individual $\Gamma$ (log scale)", fontweight="bold")
    plt.title("Experiment 2: Distribution of Individual Bounds", fontweight="bold")
    plt.xticks(m)
    plt.legend(fontsize=15, loc="upper left")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
