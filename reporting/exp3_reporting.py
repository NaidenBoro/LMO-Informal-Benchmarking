import pandas as pd


def aggregate_results(raw_df):
    summary = (
        raw_df
        .groupby(["Dataset", "m_Dropped"], as_index=False)
        .agg(
            Gamma=("Gamma", "mean"),
            Gamma_SD=("Gamma", "std"),
            Theoretical=("Theoretical", "mean"),
            Recovery=("Recovery", "mean"),
            Recovery_SD=("Recovery", "std"),
        )
    )
    return summary


def make_corner_summary(summary_df):
    rows = []

    for dataset, group in summary_df.groupby("Dataset"):
        group = group.sort_values("m_Dropped")

        peak_idx = group["Gamma"].idxmax()
        peak_row = group.loc[peak_idx]
        final_row = group.loc[group["m_Dropped"].idxmax()]

        rows.append(
            {
                "Dataset": dataset,
                "m_peak": int(peak_row["m_Dropped"]),
                "gamma_peak": peak_row["Gamma"],
                "gamma_final": final_row["Gamma"],
                "final_recovery": final_row["Recovery"],
            }
        )

    return pd.DataFrame(rows)
