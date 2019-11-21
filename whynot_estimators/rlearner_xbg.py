"""Estimate treatment effects with R-learning and XBG."""
from time import perf_counter

from whynot.framework import InferenceResult

import whynot_estimators


class RLearnerXBG(whynot_estimators.Estimator):
    """Estimate treatment effects with an R-learner and a decision tree regressor."""

    @property
    def name(self):
        """Estimator name."""
        return "rlearner_xbg"

    def import_estimator(self):
        """Import the causalml package."""
        # pylint:disable-msg=unused-import
        import causalml

    def estimate_treatment_effect(self, covariates, treatment, outcome):
        """Estimate treatment effect with R-learner and boosted decision trees.

        Wrapper around causalml.inference.meta.XGBTRegressor.

        Parameters
        ----------
            covariates: `np.ndarray`
                Array of shape [num_samples, num_features] of features
            treatment:  `np.ndarray`
                Binary array of shape [num_samples]  indicating treatment status for each
                sample.
            outcome:  `np.ndarray`
                Array of shape [num_samples] containing the observed outcome for each sample.

        Returns
        -------
            result: `whynot.framework.InferenceResult`
                InferenceResult object for this procedure

        """
        if not self.is_installed():
            raise ValueError(f"Estimator {self.name} is not installed!")

        from causalml.inference.meta import XGBTRegressor

        start_time = perf_counter()
        xbg_regressor = XGBTRegressor()
        ate, lower_bound, upper_bound = xbg_regressor.estimate_ate(
            covariates, treatment, outcome
        )
        stop_time = perf_counter()

        return InferenceResult(
            ate=ate[0],
            stderr=None,
            ci=(lower_bound[0], upper_bound[0]),
            individual_effects=None,
            elapsed_time=stop_time - start_time,
        )


RLEARNER_XBG = RLearnerXBG()
