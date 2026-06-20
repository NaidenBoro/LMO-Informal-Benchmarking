import pandas as pd


def aggregate_results(raw_df):
    summary = (
        raw_df
        .groupby(["N_Samples", "m_Dropped"], as_index=False)
        .agg(
            Theoretical=("Theoretical", "mean"),
            Empirical_Gamma=("Empirical_Gamma", "mean"),
            Empirical_Gamma_SD=("Empirical_Gamma", "std"),
            Oracle_Gamma=("Oracle_Gamma", "mean"),
            Oracle_Gamma_SD=("Oracle_Gamma", "std"),
        )
    )

    summary["Empirical_Recovery"] = summary["Empirical_Gamma"] / summary["Theoretical"]
    summary["Oracle_Recovery"] = summary["Oracle_Gamma"] / summary["Theoretical"]
    return summary


def make_reversal_summary(summary_df):
    rows = []

    for n, group in summary_df.groupby("N_Samples"):
        group = group.sort_values("m_Dropped")

        peak_idx = group["Empirical_Gamma"].idxmax()
        peak_row = group.loc[peak_idx]
        final_row = group.loc[group["m_Dropped"].idxmax()]

        gamma_peak = peak_row["Empirical_Gamma"]
        gamma_final = final_row["Empirical_Gamma"]
        theory_final = final_row["Theoretical"]

        rows.append(
            {
                "N": int(n),
                "m_peak": int(peak_row["m_Dropped"]),
                "gamma_peak": gamma_peak,
                "gamma_final": gamma_final,
                "drop_after_peak_pct": 100.0 * (1.0 - gamma_final / gamma_peak),
                "final_recovery": gamma_final / theory_final,
                "oracle_final_recovery": final_row["Oracle_Gamma"] / theory_final,
            }
        )

    return pd.DataFrame(rows)


def export_compact_latex_table(reversal_df, filename="outputs/reversal_hook_summary.tex"):
    rows = []

    for _, row in reversal_df.iterrows():
        n_label = f"{int(row['N'] / 1000)}k" if row["N"] >= 1000 else str(int(row["N"]))
        rows.append(
            f"{n_label} & "
            f"{int(row['m_peak'])} & "
            f"{row['gamma_peak']:.2f} & "
            f"{row['gamma_final']:.2f} & "
            f"{row['drop_after_peak_pct']:.1f}\\% & "
            f"{row['final_recovery']:.3f} \\\\"
        )

    table_body = "\n".join(rows)

    latex = rf"""
\begin{{table}}[tb]
\centering
\scriptsize
\caption{{Summary of the Reversal Hook in Experiment 1.}}
\label{{tab:reversal_hook_summary}}
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{lccccc}}
\toprule
$N$ & $m^*$ & $\hat{{\Gamma}}^*$ & $\hat{{\Gamma}}_p$ & Drop & Rec. \\
\midrule
{table_body}
\bottomrule
\end{{tabular}}
\end{{table}}
""".strip()

    with open(filename, "w", encoding="utf-8") as f:
        f.write(latex)

    return latex
