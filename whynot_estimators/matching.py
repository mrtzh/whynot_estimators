"""Causal estimators based on matching techniques."""
from time import perf_counter

from whynot.framework import InferenceResult

import whynot_estimators
import whynot_estimators.utils as utils


class Matching(whynot_estimators.Estimator):
    """Estimate average treatment effect via matching.

    The estimator uses the `Matching` package. For more details, see
        https://cran.r-project.org/web/packages/Matching/Matching.pdf.

    """

    @property
    def name(self):
        """Estimator name."""
        return "matching"

    def import_estimator(self):
        """Import the matching package."""
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects import numpy2ri

        numpy2ri.activate()
        if not rpackages.isinstalled("Matching"):
            raise ImportError(f"'Matching' package not installed!")
        self.matching = rpackages.importr("Matching")

    def estimate_treatment_effect(
        self,
        covariates,
        treatment,
        outcome,
        distance="mahalanobis",
        errors="abadie_imbens",
    ):
        """Estimate treatment effects using matching.

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
            distance:
                One of "inverse_variance", "mahalanobis", or "multivariate".
            errors:
                One of "abadie-imbens" or the usual "neyman" standard errors.

        Returns
        -------
            result: `whynot.framework.InferenceResult`
                InferenceResult object for this procedure

        """
        if not self.is_installed():
            raise ValueError(f"Estimator {self.name} is not installed!")

        weight_matrix = None
        if distance == "inverse_variance":
            weight = 1
        elif distance == "mahalanobis":
            weight = 2
        elif distance == "multivariate":
            weight = 3
            # Use GenMatch to automatically find covariate balance
            # pylint:disable-msg=no-member
            genmatch_out = self.matching.GenMatch(
                treatment, covariates, estimand="ATE", verbose=False
            )
            weight_matrix = genmatch_out[2]
        else:
            raise ValueError(f"{distance} not a valid option for matching estimator.")

        start_time = perf_counter()
        # pylint: disable-msg=no-member
        if weight_matrix is None:
            results = self.matching.Match(
                outcome, treatment, covariates, estimand="ATE", Weight=weight
            )
        else:
            results = self.matching.Match(
                outcome,
                treatment,
                covariates,
                estimand="ATE",
                Weight=weight,
                Weight_matrix=weight_matrix,
            )

        stop_time = perf_counter()

        ate = float(utils.extract(results, "est")[0])
        if errors == "abadie_imbens":
            stderr = float(utils.extract(results, "se")[0])
        elif errors == "neyman":
            stderr = float(utils.extract(results, "se.standard")[0])
        else:
            raise ValueError(f"{errors} not a valid option.")

        conf_int = (ate - 1.96 * stderr, ate + 1.96 * stderr)
        return InferenceResult(
            ate=ate,
            individual_effects=None,
            stderr=stderr,
            ci=conf_int,
            elapsed_time=stop_time - start_time,
        )


MATCHING = Matching()
