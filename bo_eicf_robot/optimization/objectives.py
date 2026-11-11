"""
Objective functions for Bayesian Optimization
"""

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional

from models import predict_magnitudes_batch, predict_magnitudes_single
from utils.detection import DetectionCalculator


def objective_function_binary(
    X: np.ndarray,
    gp_models: Dict,
    scaler_X: StandardScaler,
    flat_limits: Dict,
    detection_gp: Optional[GaussianProcessRegressor] = None,
    use_gp_detection: bool = False,
    detection_calc: Optional[DetectionCalculator] = None
) -> tuple:
    """
    Binary detection status (0/1) from the CDF probability for a single candidate point.
    """
    if detection_calc is None:
        detection_calc = DetectionCalculator(flat_limits=flat_limits)

    magnitudes, _ = predict_magnitudes_single(X, gp_models, scaler_X)
    prob_detected, _ = detection_calc.compute_detection_probability(magnitudes)
    return int(prob_detected > 0.5), prob_detected


def objective_function_probability(
    X: np.ndarray,
    gp_models: Dict,
    scaler_X: StandardScaler,
    flat_limits: Dict,
    detection_gp: Optional[GaussianProcessRegressor] = None,
    use_gp_detection: bool = False,
    detection_calc: Optional[DetectionCalculator] = None
) -> float:
    """
    Active-search detection probability (0-1) for a single candidate point.

    This is computed from the GP-predicted peak magnitudes and the known ZTF
    limiting-magnitude CDF, not from the detection GP. The detection GP is only
    retrained on sampled rollout labels after a query is evaluated.
    """
    if detection_calc is None:
        detection_calc = DetectionCalculator(flat_limits=flat_limits)

    magnitudes, uncertainties = predict_magnitudes_single(X, gp_models, scaler_X)
    prob, _ = detection_calc.compute_detection_probability(magnitudes, uncertainties)
    return prob


def objective_function_probability_batch(
    X_batch: np.ndarray,
    gp_models: Dict,
    scaler_X: StandardScaler,
    flat_limits: Dict,
    detection_calc: Optional[DetectionCalculator] = None,
) -> np.ndarray:
    """
    Vectorized active-search detection probability for many candidate points.

    This matches objective_function_probability: for each point, compute the
    per-filter limiting-magnitude CDF probability and return the maximum across
    available filters.
    """
    X_batch = np.atleast_2d(np.asarray(X_batch, dtype=float))
    if detection_calc is None:
        detection_calc = DetectionCalculator(flat_limits=flat_limits)

    magnitudes, _ = predict_magnitudes_batch(X_batch, gp_models, scaler_X)
    probs = np.zeros(len(X_batch), dtype=float)
    for filt in detection_calc.detection_filters:
        if filt not in magnitudes or filt not in flat_limits:
            continue
        mags = np.asarray(magnitudes[filt], dtype=float)
        valid = np.isfinite(mags)
        if not np.any(valid):
            continue
        sigma = detection_calc.filter_errors.get(filt, 0.1)
        filt_probs = np.zeros(len(X_batch), dtype=float)
        filt_probs[valid] = norm.cdf((flat_limits[filt] - mags[valid]) / sigma)
        probs = np.maximum(probs, filt_probs)
    return probs


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


def objective_function_margin_batch(
    X_batch: np.ndarray,
    gp_models: Dict,
    scaler_X: StandardScaler,
    flat_limits: Dict,
    detection_calc: Optional[DetectionCalculator] = None
) -> np.ndarray:
    """
    Vectorized detection-margin objective for multiple points.
    """
    if detection_calc is None:
        detection_calc = DetectionCalculator(flat_limits=flat_limits)

    # Get magnitudes and uncertainties for all points
    from models import predict_magnitudes_batch
    mags_batch, uncs_batch = predict_magnitudes_batch(X_batch, gp_models, scaler_X)

    # If uncertainties are not provided, use filter_errors
    if not uncs_batch:
        uncs_batch = {}
        for filt in mags_batch.keys():
            if filt in detection_calc.filter_errors:
                uncs_batch[filt] = np.ones(len(X_batch)) * detection_calc.filter_errors[filt]

    # Use vectorized margin calculation
    margins, _ = detection_calc.compute_detection_margin_batch(mags_batch, uncs_batch)
    return margins
