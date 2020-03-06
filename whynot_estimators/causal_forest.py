"""Causal estimators based on random forests."""
from time import perf_counter

from whynot.framework import InferenceResult

import whynot_estimators


class CausalForest(whynot_estimators.Estimator):
    """Estimate average and individual treatment effects using causal forest.

    The causal forest implementation uses the `grf` package. For more details see:

        Susan Athey, Julie Tibshirani and Stefan Wager. Generalized Random Forests,
        Annals of Statistics 47.2 (2019): 1148-1178.

    """

    @property
    def name(self):
        """Estimator name."""
        return "causal_forest"

    def import_estimator(self):
        """Import the grf package."""
        # pylint:disable-msg=import-outside-toplevel
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects import numpy2ri

        numpy2ri.activate()
        if not rpackages.isinstalled("grf"):
            raise ImportError(f"Package {self.name} is not installed")
        # pylint:disable-msg=attribute-defined-outside-init
        self.grf = rpackages.importr("grf")

    def estimate_treatment_effect(self, covariates, treatment, outcome):
        """Estimate average and individual treatment effects using causal forest.

        The causal forest implementation uses the `grf` package. For more details see:

            Susan Athey, Julie Tibshirani and Stefan Wager. Generalized Random Forests,
            Annals of Statistics 47.2 (2019): 1148-1178.

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

        # Convert to matrices
        treatment = treatment.reshape(-1, 1)
        outcome = outcome.reshape(-1, 1)

        start_time = perf_counter()
        # rpy2 automatically handles the conversion of numpy
        # arrays into r objects. The returned forest object is an rpy2 object!
        # pylint:disable-msg=no-member
        forest = self.grf.causal_forest(
            covariates, outcome, treatment, tune_parameters="all"
        )

        # Estimate the conditional average treatment effect. rpy2 semantics
        # so target.sample in the R api becomes target_sample
        # pylint:disable-msg=no-member
        result = self.grf.average_treatment_effect(forest, target_sample="all")
        cate, stderr = result[0], result[1]

        predictions = self.grf.predict_causal_forest(forest, covariates)

        stop_time = perf_counter()

        individual_effects = [item for sublist in predictions for item in sublist]

        conf_int = [cate - 1.96 * stderr, cate + 1.96 * stderr]
        return InferenceResult(
            ate=cate,
            stderr=stderr,
            ci=conf_int,
            individual_effects=individual_effects,
            elapsed_time=stop_time - start_time,
        )


CAUSAL_FOREST = CausalForest()
