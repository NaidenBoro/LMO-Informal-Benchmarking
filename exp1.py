"""Experiment 1: aggregate recovery of LMO informal benchmarking.

This script compares how three Gamma curves behave as the dropped subset
size m increases under the uniform synthetic DGP:

1. Empirical Gamma: the practical informal benchmark from fitted full and
   reduced logistic propensity models.
2. Oracle Gamma: a sample-realized structural benchmark using the known
   dropped logit contribution X_S @ w_S.
3. Theoretical Gamma: the analytical ceiling over the full bounded covariate
   space, attained only at aligned corners.

The experiment supports the thesis' main reversal pattern: the theoretical
ceiling rises monotonically with m, while empirical and oracle bounds can
plateau or decline when the finite sample lacks the required corner-aligned
individuals.
"""

from itertools import combinations

import numpy as np
import pandas as pd

from scipy.special import expit
from joblib import Parallel, delayed

from plotting.exp1_plotting import plot_oracle_vs_empirical, plot_recovery_ratio
from reporting.exp1_reporting import aggregate_results, export_compact_latex_table, make_reversal_summary
from utils import (
    fit_full_logits,
    fit_reduced_logits,
    make_logistic_regression,
    marginal_logit_from_treatment,
    setup_environment,
    theoretical_gamma_curve,
    validate_binary_treatment,
)


def generate_uniform_data(n, p, weights, seed):
    """Generate bounded uniform covariates and treatment from the fixed logistic DGP."""
    rng = np.random.default_rng(seed)

    X = rng.uniform(-1.0, 1.0, size=(n, p))
    true_propensity = expit(X @ weights)
    T = rng.binomial(1, true_propensity)

    validate_binary_treatment(T, context="Generated treatment")
    return X, T


def evaluate_lmo_for_m(X, T, weights, m, estimator, n_jobs=-1):
    """Evaluate every dropped subset of size m and return the strongest bounds."""
    p = X.shape[1]
    full_logits = fit_full_logits(X, T, estimator)
    marginal_logit = marginal_logit_from_treatment(T)

    def evaluate_subset(subset):
        idx = list(subset)

        # Oracle: use the known structural contribution removed by hiding subset S.
        dropped_logit_contribution = X[:, idx] @ weights[idx]
        oracle_gamma = np.exp(np.max(np.abs(dropped_logit_contribution)))

        # Empirical: refit the reduced propensity model as informal benchmarking
        # would in practice, then take the largest sample logit shift.
        reduced_logits = fit_reduced_logits(
            X=X,
            T=T,
            subset=subset,
            estimator=estimator,
            marginal_logit=marginal_logit,
        )

        empirical_gamma = np.exp(np.max(np.abs(full_logits - reduced_logits)))
        return empirical_gamma, oracle_gamma

    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_subset)(subset)
        for subset in combinations(range(p), m)
    )

    empirical_values = [r[0] for r in results]
    oracle_values = [r[1] for r in results]
    return max(empirical_values), max(oracle_values)


def evaluate_dataset(X, T, weights, m_values, estimator, n_jobs=-1):
    """Build the Gamma curve for one simulated dataset across all subset sizes."""
    rows = []

    for m in m_values:
        empirical_gamma, oracle_gamma = evaluate_lmo_for_m(
            X=X,
            T=T,
            weights=weights,
            m=m,
            estimator=estimator,
            n_jobs=n_jobs,
        )

        rows.append(
            {
                "m_Dropped": m,
                "Empirical_Gamma": empirical_gamma,
                "Oracle_Gamma": oracle_gamma,
            }
        )

    return pd.DataFrame(rows)


def run_experiment():
    """Run the sample-size comparison and export the thesis tables and figures."""
    setup_environment()

    num_iterations = 10
    p = 10
    n_values = [1000, 10000, 100000]
    m_values = list(range(1, p + 1))
    weights = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

    estimator = make_logistic_regression(random_state=42)
    theoretical = theoretical_gamma_curve(weights, m_values)

    all_runs = []

    for iteration in range(num_iterations):
        for n in n_values:
            seed = 1000 + iteration + n

            X, T = generate_uniform_data(n=n, p=p, weights=weights, seed=seed)
            result = evaluate_dataset(
                X=X,
                T=T,
                weights=weights,
                m_values=m_values,
                estimator=estimator,
                n_jobs=-1,
            )

            result["Iteration"] = iteration + 1
            result["N_Samples"] = n
            result["Theoretical"] = theoretical
            all_runs.append(result)

    raw_df = pd.concat(all_runs, ignore_index=True)
    raw_df.to_csv("outputs/experiment1_raw.csv", index=False)

    summary_df = aggregate_results(raw_df)
    summary_df.to_csv("outputs/experiment1_summary.csv", index=False)

    reversal_df = make_reversal_summary(summary_df)
    reversal_df.to_csv("outputs/reversal_hook_summary.csv", index=False)
    export_compact_latex_table(reversal_df)

    plot_oracle_vs_empirical(
        summary_df=summary_df,
        weights=weights,
        m_values=m_values,
    )
    plot_recovery_ratio(summary_df=summary_df, m_values=m_values)


if __name__ == "__main__":
    run_experiment()
