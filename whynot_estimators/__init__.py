"""Algorithms for causal inference."""

__version__ = "0.11.0"
from whynot_estimators.estimator import Estimator
from whynot_estimators.bart import CAUSAL_BART as causal_bart
from whynot_estimators.causal_forest import CAUSAL_FOREST as causal_forest
from whynot_estimators.deconfounder import DECONFOUNDER as deconfounder
from whynot_estimators.doubleml import DOUBLEML as doubleml
from whynot_estimators.ip_weighting import IP_WEIGHTING as ip_weighting
from whynot_estimators.matching import MATCHING as matching
from whynot_estimators.rlearner_mlp import RLEARNER_MLP as rlearner_mlp
from whynot_estimators.rlearner_xbg import RLEARNER_XBG as rlearner_xbg
from whynot_estimators.slearner_linear import SLEARNER_LINEAR as slearner_linear
from whynot_estimators.tmle import TMLE as tmle

ESTIMATORS = [
    causal_bart,
    causal_forest,
    deconfounder,
    doubleml,
    ip_weighting,
    matching,
    rlearner_mlp,
    rlearner_xbg,
    slearner_linear,
    tmle,
]


def get_installed():
    """Return all of the currently installed estimators."""
    return [estimator for estimator in ESTIMATORS if estimator.is_installed()]
