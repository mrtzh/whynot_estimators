"""S-learner with linear regression."""
from time import perf_counter

from whynot.framework import InferenceResult

import whynot_estimators


class SLearnerLinear(whynot_estimators.Estimator):
    """Estimate treatment effects with an S-learner and a linear regressor."""

    @property
    def name(self):
        """Estimator name."""
        return "slearner_linear"

    def import_estimator(self):
        """Import the causalml package."""
        # pylint:disable-msg=unused-import
        import causalml

    def estimate_treatment_effect(self, covariates, treatment, outcome):
        """Estimate treatement effects with an S-learner with linear regression.

        Wrapper around causalml.inference.meta.LRSRegressor.

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

        from causalml.inference.meta import LRSRegressor

        start_time = perf_counter()
        lsr = LRSRegressor()
        ate, lower_bound, upper_bound = lsr.estimate_ate(covariates, treatment, outcome)
        stop_time = perf_counter()

        return InferenceResult(
            ate=ate[0],
            stderr=None,
            ci=(lower_bound[0], upper_bound[0]),
            individual_effects=None,
            elapsed_time=stop_time - start_time,
        )


SLEARNER_LINEAR = SLearnerLinear()
