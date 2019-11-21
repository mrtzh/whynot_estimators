"""Double machine learning methods for causal inference."""
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor
from whynot.framework import InferenceResult

import whynot_estimators


class DoubleML(whynot_estimators.Estimator):
    """Use double machine learning method to estimate treatment effects."""

    @property
    def name(self):
        """Estimator name."""
        return "double_ml"

    def import_estimator(self):
        """Import the econml package."""
        # pylint:disable-msg=unused-import
        import econml

    def estimate_treatment_effect(self, covariates, treatment, outcome):
        """Run a double machine learning method to estimate treatment effect.

        Parameters
        ----------
            covariates: `np.ndarray`
                Array of shape [num_samples, num_features] of features
            treatment:  `np.ndarray`
                Array of shape [num_samples]  indicating treatment status for each
                sample.
            outcome:  `np.ndarray`
                Array of shape [num_samples] containing the observed outcome for
                each sample.

        Returns
        -------
            result: `whynot.framework.InferenceResult`
                InferenceResult object for this procedure

        """
        if not self.is_installed():
            raise ValueError(f"Estimator {self.name} is not installed!")

        from econml.dml import DMLCateEstimator

        start_time = perf_counter()

        # TODO: There are many choices for these estimators we should expose.
        estimator = DMLCateEstimator(
            model_y=RandomForestRegressor(n_estimators=10),
            model_t=LogisticRegressionCV(cv=5, max_iter=200),
            discrete_treatment=True,
        )

        # X is features/axes of heterogeneity, W is controls. We assume
        # assume that we care about heterogeneity for every covariate.
        estimator.fit(outcome, treatment, X=covariates, W=None)

        # Compute treatment effects on the sample
        treatment_effects = estimator.effect(T0=0, T1=1, X=covariates)
        ate = np.mean(treatment_effects)

        stop_time = perf_counter()

        # TODO: econml will soon support bootstrap uncertainty estimation.
        return InferenceResult(
            ate=ate,
            stderr=None,
            ci=None,
            individual_effects=treatment_effects,
            elapsed_time=stop_time - start_time,
        )


DOUBLEML = DoubleML()
