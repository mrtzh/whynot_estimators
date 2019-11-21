"""Estimate treatment effects with R-learning."""
from time import perf_counter

from whynot.framework import InferenceResult

import whynot_estimators


class RLearnerMLP(whynot_estimators.Estimator):
    """Estimate treatment effects with an R-learner and MLP regressor."""

    @property
    def name(self):
        """Estimator name."""
        return "rlearner_mlp"

    def import_estimator(self):
        """Import the causalml package."""
        # pylint:disable-msg=unused-import
        import causalml

    def estimate_treatment_effect(
        self,
        covariates,
        treatment,
        outcome,
        hidden_layer_sizes=(10, 10),
        learning_rate_init=0.1,
        early_stopping=True,
    ):
        """Estimate treatment effect with an R-learner and a MLP regressor.

        Wrapper around causalml.inference.meta.MLPTRegressor.

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

        from causalml.inference.meta import MLPTRegressor

        start_time = perf_counter()
        nn_regressor = MLPTRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate_init,
            early_stopping=early_stopping,
        )
        ate, lower_bound, upper_bound = nn_regressor.estimate_ate(
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


RLEARNER_MLP = RLearnerMLP()
