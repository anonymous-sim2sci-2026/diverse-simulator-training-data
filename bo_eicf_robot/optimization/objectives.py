"""
Objective functions for the paper-facing Bayesian optimization runs.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional

from models import predict_magnitudes_single
from utils.detection import DetectionCalculator


def objective_function_probability(
    X: np.ndarray,
    gp_models: Dict,
    scaler_X: StandardScaler,
    flat_limits: Dict,
    detection_calc: Optional[DetectionCalculator] = None
) -> float:
    """
    Active-search detection probability (0-1) for a single candidate point.

    This is computed from the GP-predicted peak magnitudes and the known ZTF
    limiting-magnitude CDF:

        f(x) = g(h(x)) = P_detect.
    """
    if detection_calc is None:
        detection_calc = DetectionCalculator(flat_limits=flat_limits)

    magnitudes, uncertainties = predict_magnitudes_single(X, gp_models, scaler_X)
    prob, _ = detection_calc.compute_detection_probability(magnitudes, uncertainties)
    return prob


def objective_function_margin(
    X: np.ndarray,
    gp_models: Dict,
    scaler_X: StandardScaler,
    flat_limits: Dict,
    detection_calc: Optional[DetectionCalculator] = None
) -> float:
    """
    Unbounded detection-margin objective.
    Uses detection_calc.filter_errors for magnitude uncertainties.
    """
    if detection_calc is None:
        detection_calc = DetectionCalculator(flat_limits=flat_limits)
    
    magnitudes, uncertainties = predict_magnitudes_single(X, gp_models, scaler_X)
    
    # If uncertainties are not provided or empty, use filter_errors from detection_calc
    if not uncertainties:
        # Use filter_errors as uncertainties
        for filt in magnitudes.keys():
            if filt in detection_calc.filter_errors:
                uncertainties[filt] = detection_calc.filter_errors[filt]
    
    return detection_calc.compute_detection_margin(magnitudes, uncertainties)
