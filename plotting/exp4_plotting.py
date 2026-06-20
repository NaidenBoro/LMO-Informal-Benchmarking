import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PRINT_COLORS = {
    "beta_u": "#D55E00",
    "uniform": "#0072B2",
    "beta_bell": "#009E73",
}


def plot_gamma_comparison(summary_df, filename="figures/gamma_3way_dgp_comparison.png"):
    plt.figure(figsize=(10, 7))

    plot_order = ["beta_u", "uniform", "beta_bell"]
    labels = {
        "beta_u": "Boundary-heavy",
        "uniform": "Uniform",
        "beta_bell": "Center-heavy",
    }

    markers = {
        "beta_u": "s",
        "uniform": "o",
        "beta_bell": "v",
    }

    colors = PRINT_COLORS

    for dgp in plot_order:
        subset = summary_df[summary_df["DGP"] == dgp].sort_values("m_Dropped")

        plt.plot(
            subset["m_Dropped"],
            subset["Gamma"],
            marker=markers[dgp],
            linewidth=3.0,
            markersize=7,
            color=colors[dgp],
            label=labels[dgp],
        )

        plt.plot(
            subset["m_Dropped"],
            subset["Oracle_Gamma"],
            marker=markers[dgp],
            linewidth=2.8,
            markersize=7,
            linestyle="--",
            color=colors[dgp],
            alpha=0.8,
            markerfacecolor="white",
            label="_nolegend_",
        )

    theory_source = summary_df[summary_df["DGP"] == "uniform"].sort_values("m_Dropped")
    m_values = theory_source["m_Dropped"]
    theory = theory_source["Theoretical"]

    plt.plot(
        m_values,
        theory,
        marker="X",
        linewidth=3.4,
        markersize=8,
        linestyle="--",
        color="black",
        label="Theoretical maximum",
    )

    plt.yscale("log")
    plt.xlabel("Subset size $m$", fontweight="bold")
    plt.ylabel(r"LMO bound $\Gamma$ (log scale)", fontweight="bold")
    plt.title("Experiment 4: Covariate Geometry Controls LMO Recovery", fontweight="bold")
    plt.xticks(m_values)

    legend_handles = [
        Line2D([0], [0], color=colors[dgp], marker=markers[dgp], linewidth=3.0, markersize=7, label=labels[dgp])
        for dgp in plot_order
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="gray",
                marker="o",
                markerfacecolor="white",
                linestyle="--",
                linewidth=2.8,
                markersize=7,
                label="Oracle (dashed, same color)",
            ),
            Line2D([0], [0], color="black", marker="X", linestyle="--", linewidth=3.4, markersize=8, label="Theoretical maximum"),
        ]
    )

    plt.legend(handles=legend_handles, fontsize=15, loc="upper left")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
