"""Causal estimators based on Bayesian Additive Regression Trees."""
from time import perf_counter

from whynot.framework import InferenceResult

import whynot_estimators
import whynot_estimators.utils as utils


class CausalBart(whynot_estimators.Estimator):
    """Estimate average and individual treatment effects using causal BART.

    The estimator reprises on the `bartCause` package. For more details see:

        Hill, J. L. (2011) Bayesian Nonparametric Modeling for Causal Inference. Journal
        of Computational and Graphical Statistics 20(1), 217–240. Taylor & Francis.
        https://doi.org/10.1198/jcgs.2010.08162.

    """

    @property
    def name(self):
        """Estimator name."""
        return "causal_bart"

    def import_estimator(self):
        """Import the BART package."""
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects import numpy2ri

        numpy2ri.activate()
        if not rpackages.isinstalled("bartCause"):
            raise ImportError(f"Package {self.name} is not installed!")
        # pylint:disable-msg=attribute-defined-outside-init
        self.bartcause = rpackages.importr("bartCause")

    def estimate_treatment_effect(self, covariates, treatment, outcome):
        """Estimate average and individual treatment effects using causal BART.

        The estimator reprises on the `bartCause` package. For more details see:

            Hill, J. L. (2011) Bayesian Nonparametric Modeling for Causal Inference. Journal
            of Computational and Graphical Statistics 20(1), 217–240. Taylor & Francis.
            https://doi.org/10.1198/jcgs.2010.08162.

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

        start_time = perf_counter()
        # pylint:disable-msg=no-member
        model = self.bartcause.bartc(outcome, treatment, covariates, verbose=False)
        results = self.bartcause.summary_bartcFit(model)
        stop_time = perf_counter()

        estimates = utils.extract(results, "estimates")
        ate = utils.extract(estimates, "estimate")[0]
        stderr = utils.extract(estimates, "sd")[0]
        ci_lower = utils.extract(estimates, "ci.lower")[0]
        ci_upper = utils.extract(estimates, "ci.upper")[0]

        return InferenceResult(
            ate=ate,
            stderr=stderr,
            ci=[ci_lower, ci_upper],
            individual_effects=None,
            elapsed_time=stop_time - start_time,
        )


CAUSAL_BART = CausalBart()
