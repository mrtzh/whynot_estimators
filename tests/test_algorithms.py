"""Tests for causal inference methods."""
import numpy as np

import whynot
import whynot_estimators


def generate_data(num_samples, num_features):
    """Generate a synthetic causal inference dataset."""
    covariates = np.random.randn(num_samples, num_features)
    treatment = (np.random.rand(num_samples) < 0.5).astype(np.int64)
    outcome = np.random.randn(num_samples)
    return covariates, treatment, outcome


def test_bart_basic():
    """Ensure BART algorithm runs."""
    covariates, treatment, outcome = generate_data(100, 10)
    if whynot_estimators.causal_bart.is_installed():
        _ = whynot_estimators.causal_bart.estimate_treatment_effect(
            covariates, treatment, outcome
        )


def test_causal_forest_basic():
    """Ensure causal forest algorithm runs."""
    covariates, treatment, outcome = generate_data(100, 10)
    _ = whynot_estimators.causal_forest.estimate_treatment_effect(
        covariates, treatment, outcome
    )


def test_deconfounder_basic():
    """Ensure deconfounders runs."""
    covariates, treatment, outcome = generate_data(100, 10)
    if whynot_estimators.deconfounder.is_installed():
        _ = whynot_estimators.deconfounder.estimate_treatment_effect(
            covariates, treatment, outcome
        )


def test_doubleml_basic():
    """Ensure doubleml runs."""
    covariates, treatment, outcome = generate_data(100, 10)
    if whynot_estimators.doubleml.is_installed():
        _ = whynot_estimators.doubleml.estimate_treatment_effect(
            covariates, treatment, outcome
        )


def test_ipweighting_basic():
    """Ensure ip weighting runs."""
    covariates, treatment, outcome = generate_data(100, 10)
    if whynot_estimators.ip_weighting.is_installed():
        _ = whynot_estimators.ip_weighting.estimate_treatment_effect(
            covariates, treatment, outcome
        )


def test_matching_basic():
    """Ensure matching runs."""
    covariates, treatment, outcome = generate_data(100, 10)
    if whynot_estimators.matching.is_installed():
        _ = whynot_estimators.matching.estimate_treatment_effect(
            covariates, treatment, outcome
        )


def test_rlearner_basic():
    """Ensure rlearner methods run."""
    covariates, treatment, outcome = generate_data(100, 10)
    if whynot_estimators.rlearner_mlp.is_installed():
        _ = whynot_estimators.rlearner_mlp.estimate_treatment_effect(
            covariates, treatment, outcome
        )
    if whynot_estimators.rlearner_xbg.is_installed():
        _ = whynot_estimators.rlearner_xbg.estimate_treatment_effect(
            covariates, treatment, outcome
        )


def test_slearner_basic():
    """Ensure slearner methods run."""
    covariates, treatment, outcome = generate_data(100, 10)
    if whynot_estimators.slearner_linear.is_installed():
        _ = whynot_estimators.slearner_linear.estimate_treatment_effect(
            covariates, treatment, outcome
        )


def test_tmle_basic():
    """Ensure tmle runs."""
    covariates, treatment, outcome = generate_data(100, 10)
    _ = whynot_estimators.tmle.estimate_treatment_effect(covariates, treatment, outcome)


def test_causal_suite():
    """Ensure casual suite picks up everything."""
    covariates, treatment, outcome = generate_data(100, 10)
    results = whynot.causal_suite(covariates, treatment, outcome)
    if whynot_estimators.causal_forest.is_installed():
        assert "causal_forest" in results
    if whynot_estimators.ip_weighting.is_installed():
        assert "ip_weighting" in results
    if whynot_estimators.matching.is_installed():
        assert "matching" in results
    if whynot_estimators.tmle.is_installed():
        assert "tmle" in results
