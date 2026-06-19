import matplotlib.pyplot as plt

from utils import theoretical_gamma_curve


PRINT_COLORS = ["#0072B2", "#D55E00", "#009E73"]


def plot_oracle_vs_empirical(summary_df, weights, m_values, filename="figures/oracle_vs_empirical.png"):
    plt.figure(figsize=(10, 7))

    n_values = sorted(summary_df["N_Samples"].unique())
    colors = PRINT_COLORS[: len(n_values)]

    for i, n in enumerate(n_values):
        subset = summary_df[summary_df["N_Samples"] == n].sort_values("m_Dropped")

        plt.plot(
            subset["m_Dropped"],
            subset["Empirical_Gamma"],
            marker="o",
            linewidth=3.0,
            markersize=7,
            color=colors[i],
            label=f"Empirical, N={n}",
        )

        plt.plot(
            subset["m_Dropped"],
            subset["Oracle_Gamma"],
            marker="^",
            linewidth=2.8,
            markersize=7,
            linestyle="--",
            color=colors[i],
            alpha=0.75,
            label=f"Oracle, N={n}",
        )

    theoretical = theoretical_gamma_curve(weights, m_values)

    plt.plot(
        m_values,
        theoretical,
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
    plt.title("Experiment 1: Aggregate LMO Bounds", fontweight="bold")
    plt.xticks(m_values)
    plt.legend(fontsize=15, loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_recovery_ratio(summary_df, m_values, filename="figures/recovery_ratio.png"):
    plt.figure(figsize=(10, 7))

    n_values = sorted(summary_df["N_Samples"].unique())
    colors = PRINT_COLORS[: len(n_values)]

    for i, n in enumerate(n_values):
        subset = summary_df[summary_df["N_Samples"] == n].sort_values("m_Dropped")

        plt.plot(
            subset["m_Dropped"],
            subset["Empirical_Recovery"],
            marker="o",
            linewidth=3.0,
            markersize=7,
            color=colors[i],
            label=f"Empirical recovery, N={n}",
        )

        plt.plot(
            subset["m_Dropped"],
            subset["Oracle_Recovery"],
            marker="^",
            linewidth=2.8,
            markersize=7,
            linestyle="--",
            color=colors[i],
            alpha=0.75,
            label=f"Oracle recovery, N={n}",
        )

    plt.axhline(y=1.0, color="black", linestyle=":", linewidth=3.0, label="Perfect recovery")

    max_recovery = max(summary_df["Empirical_Recovery"].max(), summary_df["Oracle_Recovery"].max())

    plt.ylim(0, max(1.05, max_recovery * 1.08))
    plt.xlabel("Subset size $m$", fontweight="bold")
    plt.ylabel(r"Observed $\Gamma$ / theoretical $\Gamma$", fontweight="bold")
    plt.title("Experiment 1: Recovery of the Theoretical LMO Bound", fontweight="bold")
    plt.xticks(m_values)
    plt.legend(fontsize=15, loc="best")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
