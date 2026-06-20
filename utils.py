import os
import warnings

import numpy as np
import seaborn as sns

from scipy.special import logit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression


"""Shared helpers for experiment setup, model fitting, and common math utilities."""


def setup_environment():
    """Configure plotting style and create standard output directories."""
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.filterwarnings("ignore")

    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
        },
    )

    os.makedirs("figures", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)


def make_logistic_regression(random_state=42):
    """Build a no-penalty logistic model across sklearn versions."""
    try:
        return LogisticRegression(
            penalty=None,
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        )
    except Exception:
        return LogisticRegression(
            penalty="none",
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        )


def theoretical_gamma_curve(weights, m_values):
    """Compute the structural maximum Gamma curve for each dropped size m."""
    ordered_weights = np.sort(np.abs(weights))[::-1]
    return np.array([np.exp(np.sum(ordered_weights[:m])) for m in m_values])


def validate_binary_treatment(T, context="Treatment"):
    """Ensure treatment contains both classes before fitting logistic models."""
    if len(np.unique(T)) < 2:
        raise ValueError(f"{context} has only one class. Try another seed or larger N.")


def marginal_logit_from_treatment(T):
    """Return marginal log-odds from treatment prevalence with safety checks."""
    marginal_prob = np.mean(T)
    if marginal_prob <= 0 or marginal_prob >= 1:
        raise ValueError("Marginal treatment probability is 0 or 1; logit is undefined.")
    return logit(marginal_prob)


def fit_full_logits(X, T, estimator):
    """Fit the full model and return decision-function logits on X."""
    model = clone(estimator)
    model.fit(X, T)
    return model.decision_function(X)


def fit_reduced_logits(X, T, subset, estimator, marginal_logit):
    """Fit reduced model dropping subset columns; fallback to marginal logit if empty."""
    n, p = X.shape
    mask = np.ones(p, dtype=bool)
    mask[list(subset)] = False

    X_reduced = X[:, mask]
    if X_reduced.shape[1] == 0:
        return np.full(n, marginal_logit)

    model = clone(estimator)
    model.fit(X_reduced, T)
    return model.decision_function(X_reduced)
