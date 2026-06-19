from itertools import combinations

import numpy as np
import pandas as pd

from scipy.special import expit
from joblib import Parallel, delayed

from plotting.exp4_plotting import plot_gamma_comparison
from reporting.exp4_reporting import aggregate_results, make_final_recovery_summary
from utils import (
    fit_full_logits,
    fit_reduced_logits,
    make_logistic_regression,
    marginal_logit_from_treatment,
    setup_environment,
    theoretical_gamma_curve,
    validate_binary_treatment,
)


"""Experiment 4: assess how covariate geometry changes empirical LMO recovery."""


def generate_data(n, p, weights, seed, dgp_type):
    rng = np.random.default_rng(seed)

    if dgp_type == "uniform":
        X = rng.uniform(-1.0, 1.0, size=(n, p))
    elif dgp_type == "beta_u":
        B = rng.beta(0.2, 0.2, size=(n, p))
        X = 2 * B - 1
    elif dgp_type == "beta_bell":
        B = rng.beta(5.0, 5.0, size=(n, p))
        X = 2 * B - 1
    else:
        raise ValueError("dgp_type must be 'uniform', 'beta_u', or 'beta_bell'.")

    true_propensity = expit(X @ weights)
    T = rng.binomial(1, true_propensity)

    validate_binary_treatment(T, context="Generated treatment")
    return X, T


def evaluate_oracle_for_m(X, weights, m):
    p = X.shape[1]
    subset_gammas = []

    for subset in combinations(range(p), m):
        dropped_logit = X[:, list(subset)] @ weights[list(subset)]
        subset_gammas.append(np.exp(np.max(np.abs(dropped_logit))))

    return np.max(subset_gammas)


def evaluate_lmo_for_m(X, T, m, estimator, n_jobs=-1):
    p = X.shape[1]
    full_logits = fit_full_logits(X, T, estimator)
    marginal_logit = marginal_logit_from_treatment(T)

    def evaluate_subset(subset):
        reduced_logits = fit_reduced_logits(
            X=X,
            T=T,
            subset=subset,
            estimator=estimator,
            marginal_logit=marginal_logit,
        )
        return np.exp(np.max(np.abs(full_logits - reduced_logits)))

    subset_gammas = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_subset)(subset)
        for subset in combinations(range(p), m)
    )

    return np.max(subset_gammas)


def evaluate_dataset(X, T, weights, m_values, estimator, n_jobs=-1):
    theoretical = theoretical_gamma_curve(weights, m_values)
    rows = []

    for idx, m in enumerate(m_values):
        gamma = evaluate_lmo_for_m(
            X=X,
            T=T,
            m=m,
            estimator=estimator,
            n_jobs=n_jobs,
        )
        oracle_gamma = evaluate_oracle_for_m(X=X, weights=weights, m=m)

        rows.append(
            {
                "m_Dropped": m,
                "Gamma": gamma,
                "Oracle_Gamma": oracle_gamma,
                "Theoretical": theoretical[idx],
                "Recovery": gamma / theoretical[idx],
                "Oracle_Recovery": oracle_gamma / theoretical[idx],
            }
        )

    return pd.DataFrame(rows)


def run_experiment_4():
    setup_environment()

    num_iterations = 10
    N = 10000
    p = 10
    m_values = list(range(1, p + 1))
    weights = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

    dgp_types = ["uniform", "beta_u", "beta_bell"]
    estimator = make_logistic_regression(random_state=42)
    all_runs = []

    for iteration in range(num_iterations):
        for dgp in dgp_types:
            seed = 3000 + iteration

            X, T = generate_data(
                n=N,
                p=p,
                weights=weights,
                seed=seed,
                dgp_type=dgp,
            )

            result = evaluate_dataset(
                X=X,
                T=T,
                weights=weights,
                m_values=m_values,
                estimator=estimator,
            )

            result["DGP"] = dgp
            result["Iteration"] = iteration + 1
            all_runs.append(result)

    raw_df = pd.concat(all_runs, ignore_index=True)
    raw_df.to_csv("outputs/experiment4_raw.csv", index=False)

    summary_df = aggregate_results(raw_df)
    summary_df.to_csv("outputs/experiment4_summary.csv", index=False)

    final_summary = make_final_recovery_summary(summary_df)
    final_summary.to_csv("outputs/experiment4_final_recovery.csv", index=False)

    plot_gamma_comparison(summary_df=summary_df, filename="figures/gamma_3way_dgp_comparison.png")


if __name__ == "__main__":
    run_experiment_4()
