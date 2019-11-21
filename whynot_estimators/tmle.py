"""Causal estimators based on Targeted Maximum Likelihood Estimation (TMLE)."""
from time import perf_counter

import numpy as np
from whynot.framework import InferenceResult

import whynot_estimators
import whynot_estimators.utils as utils


class Tmle(whynot_estimators.Estimator):
    """Estimates average and individual treatment effects using TMLE.

    This estimator uses the `tmle` package to perform targeted maximum likelihood
    estimation. For more details, see:

        https://cran.r-project.org/web/packages/tmle/tmle.pdf

    For a brief introduction to these techniques, see:

        Gruber, Susan and van der Laan, Mark J., "Targeted Maximum Likelihood
        Estimation: A Gentle Introduction" (August 2009). U.C. Berkeley Division of
        Biostatistics Working Paper Series. Working Paper 252.
        https://biostats.bepress.com/ucbbiostat/paper252

    """

    @property
    def name(self):
        """Estimator name."""
        return "tmle"

    def import_estimator(self):
        """Import the estimator and its dependencies."""
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects import numpy2ri

        numpy2ri.activate()
        if not rpackages.isinstalled("tmle"):
            raise ImportError(f"Package {self.name} is not installed!")
        self.tmle = rpackages.importr("tmle")

    def estimate_treatment_effect(self, covariates, treatment, outcome):
        """Estimates average and individual treatment effects using TMLE.

        This estimator uses the `tmle` package to perform targeted maximum likelihood
        estimation. For more details, see:
            https://cran.r-project.org/web/packages/tmle/tmle.pdf

        Parameters
        ----------
            covariates: `np.ndarray`
                Array of shape [num_samples, num_features] of features
            treatment:  `np.ndarray`
                Array of shape [num_samples]  indicating treatment status for each sample.
            outcome:  `np.ndarray`
                Array of shape [num_samples] containing the observed outcome for each sample.

        Returns
        -------
            result: `whynot.framework.InferenceResult`
                InferenceResult object for this procedure

        """
        if not self.is_installed():
            raise ValueError(f"Estimator {self.name} is not installed!")

        # Ensure arrays will be conformable.
        treatment = treatment.reshape(-1, 1)
        outcome = outcome.reshape(-1, 1)

        start_time = perf_counter()
        # pylint:disable-msg=no-member
        model = self.tmle.tmle(outcome, treatment, covariates)
        stop_time = perf_counter()

        # From the TMLE documentation
        # https://cran.r-project.org/web/packages/tmle/tmle.pdf
        estimates = utils.extract(model, "estimates")
        ate_estimate = utils.extract(estimates, "ATE")
        ate = utils.extract(ate_estimate, "psi")
        conf_int = utils.extract(ate_estimate, "CI")
        var = utils.extract(ate_estimate, "var.psi")

        # Unpack
        ate = float(ate[0])
        var = float(var[0])
        conf_int = (float(conf_int[0]), float(conf_int[1]))

        return InferenceResult(
            ate=ate,
            stderr=np.sqrt(var),
            ci=conf_int,
            individual_effects=None,
            elapsed_time=stop_time - start_time,
        )


TMLE = Tmle()
