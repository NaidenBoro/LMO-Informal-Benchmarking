import pandas as pd


def aggregate_results(raw_df):
    summary = (
        raw_df
        .groupby(["DGP", "m_Dropped"], as_index=False)
        .agg(
            Gamma=("Gamma", "mean"),
            Gamma_SD=("Gamma", "std"),
            Oracle_Gamma=("Oracle_Gamma", "mean"),
            Oracle_Gamma_SD=("Oracle_Gamma", "std"),
            Theoretical=("Theoretical", "mean"),
            Recovery=("Recovery", "mean"),
            Recovery_SD=("Recovery", "std"),
            Oracle_Recovery=("Oracle_Recovery", "mean"),
            Oracle_Recovery_SD=("Oracle_Recovery", "std"),
        )
    )
    return summary


def make_final_recovery_summary(summary_df):
    rows = []

    for dgp, group in summary_df.groupby("DGP"):
        group = group.sort_values("m_Dropped")
        final = group.loc[group["m_Dropped"].idxmax()]
        peak = group.loc[group["Gamma"].idxmax()]

        rows.append(
            {
                "DGP": dgp,
                "m_peak": int(peak["m_Dropped"]),
                "gamma_peak": peak["Gamma"],
                "gamma_final": final["Gamma"],
                "final_recovery": final["Recovery"],
                "oracle_gamma_final": final["Oracle_Gamma"],
                "oracle_final_recovery": final["Oracle_Recovery"],
            }
        )

    return pd.DataFrame(rows)
