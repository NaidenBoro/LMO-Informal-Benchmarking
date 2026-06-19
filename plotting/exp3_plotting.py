import matplotlib.pyplot as plt


def plot_corner_experiment(summary_df, filename="figures/gamma_corner_mc.png"):
    plt.figure(figsize=(10, 7))

    labels = {
        "standard": "Standard sample",
        "corners": "Corner-injected sample",
    }

    for dataset in ["standard", "corners"]:
        subset = summary_df[summary_df["Dataset"] == dataset].sort_values("m_Dropped")

        plt.plot(
            subset["m_Dropped"],
            subset["Gamma"],
            marker="o" if dataset == "standard" else "s",
            linewidth=3.0,
            markersize=7,
            linestyle="-" if dataset == "standard" else "--",
            label=labels[dataset],
        )

    theory = summary_df[summary_df["Dataset"] == "standard"].sort_values("m_Dropped")["Theoretical"]
    m_values = sorted(summary_df["m_Dropped"].unique())

    plt.plot(
        m_values,
        theory,
        marker="X",
        linewidth=3.4,
        markersize=8,
        linestyle=":",
        color="black",
        label="Theoretical maximum",
    )

    plt.yscale("log")
    plt.xlabel("Subset size $m$", fontweight="bold")
    plt.ylabel(r"Confounding strength $\Gamma$ (log scale)", fontweight="bold")
    plt.title("Experiment 3: Corner Injection Mechanism Test", fontweight="bold")
    plt.xticks(m_values)
    plt.legend(fontsize=15, loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
