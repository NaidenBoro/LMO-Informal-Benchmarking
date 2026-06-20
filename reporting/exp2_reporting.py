import numpy as np
import pandas as pd


def build_summary_table(gamma_matrix, theoretical, m_values):
    rows = []

    for row_idx, m in enumerate(m_values):
        gammas = gamma_matrix[row_idx, :]
        theory = theoretical[row_idx]

        rows.append(
            {
                "m_Dropped": m,
                "Theoretical": theory,
                "Median": np.median(gammas),
                "P90": np.percentile(gammas, 90),
                "P95": np.percentile(gammas, 95),
                "P99": np.percentile(gammas, 99),
                "Max": np.max(gammas),
                "Median_Recovery": np.median(gammas) / theory,
                "P90_Recovery": np.percentile(gammas, 90) / theory,
                "P95_Recovery": np.percentile(gammas, 95) / theory,
                "P99_Recovery": np.percentile(gammas, 99) / theory,
                "Max_Recovery": np.max(gammas) / theory,
                "Boundary_Individual": int(np.argmax(gammas)),
            }
        )

    return pd.DataFrame(rows)


def build_top_individual_table(gamma_matrix, X, weights, m_values):
    boundary_individuals = np.argmax(gamma_matrix, axis=1)
    unique_boundary_individuals = sorted(set(boundary_individuals))

    rows = []

    for individual_id in unique_boundary_individuals:
        x_i = X[individual_id]
        weighted_terms = x_i * weights

        rows.append(
            {
                "Individual": individual_id,
                "Max_Gamma": np.max(gamma_matrix[:, individual_id]),
                "Peak_m": m_values[int(np.argmax(gamma_matrix[:, individual_id]))],
                "Mean_X": np.mean(x_i),
                "Num_Positive_X": int(np.sum(x_i > 0)),
                "Num_Negative_X": int(np.sum(x_i < 0)),
                "Full_Logit_Xw": float(x_i @ weights),
                "Abs_Full_Logit_Xw": float(abs(x_i @ weights)),
                "Sum_Abs_Weighted_Terms": float(np.sum(np.abs(weighted_terms))),
                "Signed_Weighted_Term_Sum": float(np.sum(weighted_terms)),
            }
        )

    return pd.DataFrame(rows)


def build_long_table(gamma_matrix, theoretical, m_values):
    rows = []

    for row_idx, m in enumerate(m_values):
        theory = theoretical[row_idx]
        boundary_individual = int(np.argmax(gamma_matrix[row_idx, :]))

        for individual_id, gamma in enumerate(gamma_matrix[row_idx, :]):
            rows.append(
                {
                    "m_Dropped": m,
                    "Individual": individual_id,
                    "Gamma": gamma,
                    "Theoretical": theory,
                    "Recovery": gamma / theory,
                    "Is_Boundary_Individual": individual_id == boundary_individual,
                }
            )

    return pd.DataFrame(rows)
