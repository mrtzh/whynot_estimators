"""Causal estimators based on IP weighting techniques."""
from time import perf_counter

import pandas as pd
from whynot.framework import InferenceResult

import whynot_estimators
import whynot_estimators.utils as utils


class IPWeighting(whynot_estimators.Estimator):
    """Estimate average treatment effects via inverse propensity scores.

    Uses WeightIt to get propensity scores:
        https://cran.r-project.org/web/packages/WeightIt/WeightIt.pdf

    Uses Survey to fit a generalized linear model:
        https://cran.r-project.org/web/packages/survey/survey.pdf

    """

    @property
    def name(self):
        """Estimator name."""
        return "ip_weighting"

    def import_estimator(self):
        """Import WeightIt and survey packages."""
        import rpy2.robjects.packages as rpackages
        from rpy2.robjects import numpy2ri, pandas2ri

        numpy2ri.activate()
        pandas2ri.activate()
        if not rpackages.isinstalled("WeightIt") or not rpackages.isinstalled("survey"):
            raise ImportError(f"Estimator {self.name} is not installed!")
        self.weightit = rpackages.importr("WeightIt")
        self.survey = rpackages.importr("survey")

    def estimate_treatment_effect(self, covariates, treatment, outcome):
        """Estimate average treatment effects via inverse propensity scores.

        Uses WeightIt to get propensity scores:
            https://cran.r-project.org/web/packages/WeightIt/WeightIt.pdf

        Uses Survey to fit a generalized linear model:
            https://cran.r-project.org/web/packages/survey/survey.pdf

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

        from rpy2.robjects import Formula

        # Convert data into a dataframe
        column_names = ["x_{}".format(idx) for idx in range(covariates.shape[1])]
        data = pd.DataFrame(covariates, columns=column_names)
        data["treatment"] = treatment
        data["outcome"] = outcome

        propensity_formula = Formula(f"treatment ~ {' + '.join(column_names)}")

        start_time = perf_counter()

        # Note: The field ["ps"] contains the actual propensity scores we could use
        # elsewhere if needed. Stabilize=True not strictly needed for reasonable
        # performance.
        # pylint:disable-msg=no-member
        weight_obj = self.weightit.weightit(
            propensity_formula, data=data, method="ps", estimand="ATE", stabilize=True
        )
        weights = utils.extract(weight_obj, "weights")

        # Use survey to fit weighted glm and get robust confidence intervals.
        design = self.survey.svydesign(Formula("~ 1"), weights=weights, data=data)
        fit = self.survey.svyglm(Formula("outcome ~ treatment"), design=design)

        # coefs[0] is bias term, coefs[1] is the treatment
        ate = self.survey.coef_svyglm(fit)[1]

        # 95% confidence interval for the treatment
        # cis[0] is for the bias term, cis[1] is for treatment.
        robust_ci = self.survey.confint_svyglm(fit)[1]
        robust_ci = (robust_ci[0], robust_ci[1])

        stop_time = perf_counter()

        return InferenceResult(
            ate=ate,
            stderr=None,
            ci=robust_ci,
            individual_effects=None,
            elapsed_time=stop_time - start_time,
        )


IP_WEIGHTING = IPWeighting()
