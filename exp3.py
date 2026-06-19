from itertools import combinations

import numpy as np
import pandas as pd

from scipy.special import expit
from joblib import Parallel, delayed

from plotting.exp3_plotting import plot_corner_experiment
from reporting.exp3_reporting import aggregate_results, make_corner_summary
from utils import (
    fit_full_logits,
    fit_reduced_logits,
    make_logistic_regression,
    marginal_logit_from_treatment,
    setup_environment,
    theoretical_gamma_curve,
    validate_binary_treatment,
)


"""Experiment 3: compare standard vs corner-injected samples under matched randomness."""


def generate_matched_datasets(n, p, weights, seed):
    rng = np.random.default_rng(seed)

    X_std = rng.uniform(-1.0, 1.0, size=(n, p))

    X_corner = X_std.copy()
    X_corner[-2, :] = -1.0
    X_corner[-1, :] = 1.0

    treatment_uniforms = rng.uniform(0.0, 1.0, size=n)

    prop_std = expit(X_std @ weights)
    prop_corner = expit(X_corner @ weights)

    T_std = (treatment_uniforms < prop_std).astype(int)
    T_corner = (treatment_uniforms < prop_corner).astype(int)

    validate_binary_treatment(T_std, context="Standard treatment")
    validate_binary_treatment(T_corner, context="Corner treatment")

    return X_std, T_std, X_corner, T_corner


def evaluate_lmo_for_m(X, T, m, estimator, n_jobs=-1):
    p = X.shape[1]
    validate_binary_treatment(T)
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
        gamma_values = np.exp(np.abs(full_logits - reduced_logits))
        return np.max(gamma_values)

    gammas = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_subset)(subset)
        for subset in combinations(range(p), m)
    )

    return np.max(gammas)


def evaluate_dataset(X, T, weights, m_values, estimator, n_jobs=-1):
    rows = []
    theoretical = theoretical_gamma_curve(weights, m_values)

    for idx, m in enumerate(m_values):
        gamma = evaluate_lmo_for_m(
            X=X,
            T=T,
            m=m,
            estimator=estimator,
            n_jobs=n_jobs,
        )

        rows.append(
            {
                "m_Dropped": m,
                "Gamma": gamma,
                "Theoretical": theoretical[idx],
                "Recovery": gamma / theoretical[idx],
            }
        )

    return pd.DataFrame(rows)


def run_experiment_3():
    setup_environment()

    num_iterations = 10
    N = 10000
    p = 10
    m_values = list(range(1, p + 1))
    weights = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

    estimator = make_logistic_regression(random_state=42)
    all_runs = []

    for iteration in range(num_iterations):
        seed = 2000 + iteration

        X_std, T_std, X_corner, T_corner = generate_matched_datasets(
            n=N,
            p=p,
            weights=weights,
            seed=seed,
        )

        std_df = evaluate_dataset(
            X=X_std,
            T=T_std,
            weights=weights,
            m_values=m_values,
            estimator=estimator,
            n_jobs=-1,
        )
        std_df["Dataset"] = "standard"
        std_df["Iteration"] = iteration + 1

        corner_df = evaluate_dataset(
            X=X_corner,
            T=T_corner,
            weights=weights,
            m_values=m_values,
            estimator=estimator,
            n_jobs=-1,
        )
        corner_df["Dataset"] = "corners"
        corner_df["Iteration"] = iteration + 1

        all_runs.append(std_df)
        all_runs.append(corner_df)

    raw_df = pd.concat(all_runs, ignore_index=True)
    raw_df.to_csv("outputs/experiment3_raw.csv", index=False)

    summary_df = aggregate_results(raw_df)
    summary_df.to_csv("outputs/experiment3_summary.csv", index=False)

    corner_summary = make_corner_summary(summary_df)
    corner_summary.to_csv("outputs/experiment3_corner_summary.csv", index=False)

    plot_corner_experiment(summary_df=summary_df, filename="figures/gamma_corner_mc.png")


if __name__ == "__main__":
    run_experiment_3()
