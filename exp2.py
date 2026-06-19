from itertools import combinations

import numpy as np

from scipy.special import expit

from plotting.exp2_plotting import plot_distribution_summary
from reporting.exp2_reporting import build_long_table, build_summary_table, build_top_individual_table
from utils import (
    fit_full_logits,
    fit_reduced_logits,
    make_logistic_regression,
    marginal_logit_from_treatment,
    setup_environment,
    theoretical_gamma_curve,
    validate_binary_treatment,
)


"""Experiment 2: individual-level Gamma trajectories and envelope diagnostics."""


def generate_uniform_data(n, p, weights, seed):
    """Generate one dataset from the uniform logistic DGP."""
    rng = np.random.default_rng(seed)

    X = rng.uniform(-1.0, 1.0, size=(n, p))
    true_propensity = expit(X @ weights)
    T = rng.binomial(1, true_propensity)

    validate_binary_treatment(T, context="Generated treatment")
    return X, T


def calculate_individual_gammas_for_m(X, T, m, estimator, full_logits):
    n, p = X.shape
    max_gamma_per_individual = np.ones(n)
    marginal_logit = marginal_logit_from_treatment(T)

    for subset in combinations(range(p), m):
        reduced_logits = fit_reduced_logits(
            X=X,
            T=T,
            subset=subset,
            estimator=estimator,
            marginal_logit=marginal_logit,
        )

        gamma = np.exp(np.abs(full_logits - reduced_logits))
        max_gamma_per_individual = np.maximum(max_gamma_per_individual, gamma)

    return max_gamma_per_individual


def compute_individual_trajectories(X, T, m_values, estimator):
    full_logits = fit_full_logits(X, T, estimator)

    n = X.shape[0]
    gamma_matrix = np.zeros((len(m_values), n))

    for row_idx, m in enumerate(m_values):
        gamma_matrix[row_idx, :] = calculate_individual_gammas_for_m(
            X=X,
            T=T,
            m=m,
            estimator=estimator,
            full_logits=full_logits,
        )

    return gamma_matrix


def run_experiment_2():
    setup_environment()

    N = 10000
    p = 10
    seed = 42
    m_values = list(range(1, p + 1))
    weights = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

    theoretical = theoretical_gamma_curve(weights, m_values)
    estimator = make_logistic_regression(random_state=42)

    X, T = generate_uniform_data(n=N, p=p, weights=weights, seed=seed)

    gamma_matrix = compute_individual_trajectories(
        X=X,
        T=T,
        m_values=m_values,
        estimator=estimator,
    )

    summary_df = build_summary_table(
        gamma_matrix=gamma_matrix,
        theoretical=theoretical,
        m_values=m_values,
    )
    top_individuals_df = build_top_individual_table(
        gamma_matrix=gamma_matrix,
        X=X,
        weights=weights,
        m_values=m_values,
    )
    long_df = build_long_table(
        gamma_matrix=gamma_matrix,
        theoretical=theoretical,
        m_values=m_values,
    )

    summary_df.to_csv("outputs/experiment2_summary.csv", index=False)
    top_individuals_df.to_csv("outputs/experiment2_top_individuals.csv", index=False)
    long_df.to_csv("outputs/experiment2_individual_trajectories.csv", index=False)

    plot_distribution_summary(
        summary_df=summary_df,
        gamma_matrix=gamma_matrix,
        filename="figures/individual_gamma_distribution.png",
    )


if __name__ == "__main__":
    run_experiment_2()
