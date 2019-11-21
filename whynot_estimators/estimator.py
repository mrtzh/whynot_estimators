"""Base class for whynot_estimators."""
from abc import abstractmethod, ABCMeta


class Estimator:
    """Generic estimator class."""

    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def name(self):
        """Name of the estimator."""
        return NotImplemented

    @abstractmethod
    def import_estimator(self):
        """Import the estimator."""
        return NotImplemented

    def is_installed(self):
        """Check if the estimator is installed."""
        try:
            self.import_estimator()
            return True
        except ImportError:
            return False

    @abstractmethod
    def estimate_treatment_effect(
        self, covariates, treatment, outcome, *args, **kwargs
    ):
        """Estimate average and individual treatment effects.

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
