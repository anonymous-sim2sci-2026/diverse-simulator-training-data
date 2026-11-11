"""
Acquisition functions for Bayesian Optimization
"""

import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple

from utils.detection import DetectionCalculator
from utils.h_columns import H_OUTPUT_COLUMNS, MAGNITUDE_COLUMNS


DEFAULT_EICF_MC_SAMPLES = 256
MULTITASK_GP_KEY = "__multitask_magnitude_gp__"


def _available_h_outputs(gp_models: Dict) -> list:
    """h(x) outputs that are present in the surrogate models."""
    multitask_gp = gp_models.get(MULTITASK_GP_KEY)
    if multitask_gp is not None and getattr(multitask_gp, "available", False):
        return [name for name in H_OUTPUT_COLUMNS if name in multitask_gp.filters]
    return [name for name in H_OUTPUT_COLUMNS if name in gp_models and gp_models[name] is not None]


def _available_detection_filters(gp_models: Dict, flat_limits: Dict) -> list:
    """Magnitude filters that are present in both h(x) and the detection limits."""
    h_outputs = set(_available_h_outputs(gp_models))
    return [filt for filt in MAGNITUDE_COLUMNS if filt in flat_limits and filt in h_outputs]


def composite_detection_probability(
    h_samples: np.ndarray,
    filters: list,
    flat_limits: Dict,
    detection_calc: DetectionCalculator,
) -> np.ndarray:
    """
    Cheap composite function g(h): map sampled peak magnitudes to ZTF detection probability.

    h_samples has shape (n_samples, n_filters), with columns ordered like filters.
    """
    if h_samples.size == 0 or not filters:
        return np.zeros(h_samples.shape[0] if h_samples.ndim == 2 else 1)

    probs = []
    for j, filt in enumerate(filters):
        if filt not in flat_limits:
            continue
        limit = flat_limits[filt]
        sigma = detection_calc.filter_errors.get(filt, 0.1)
        probs.append(norm.cdf((limit - h_samples[:, j]) / sigma))

    if not probs:
        return np.zeros(h_samples.shape[0])

    return np.max(np.vstack(probs), axis=0)


def sample_h_posterior(
    X: np.ndarray,
    gp_models: Dict,
    scaler_X: StandardScaler,
    flat_limits: Dict,
    n_mc: int = DEFAULT_EICF_MC_SAMPLES,
    random_state: int = 12345,
) -> Tuple[np.ndarray, list, np.ndarray, np.ndarray]:
    """
    Sample the full h(x) output vector from the GP posterior.

    If a BoTorch MultiTaskGP is available, this uses its full posterior
    covariance across filters. Otherwise it falls back to the independent
    sklearn-GP diagonal approximation.
    """
    filters = _available_h_outputs(gp_models)
    if not filters:
        return np.empty((0, 0)), [], np.array([]), np.array([])

    multitask_gp = gp_models.get(MULTITASK_GP_KEY)
    if multitask_gp is not None and getattr(multitask_gp, "available", False):
        means, cov, filters = multitask_gp.posterior_h(X, filters=filters)
        cov = np.asarray(cov, dtype=float)
        cov = 0.5 * (cov + cov.T)
        cov = cov + 1e-9 * np.eye(len(filters))

        rng = np.random.default_rng(random_state)
        try:
            chol = np.linalg.cholesky(cov)
            z = rng.standard_normal(size=(n_mc, len(filters)))
            h_samples = means + z @ chol.T
        except np.linalg.LinAlgError:
            h_samples = rng.multivariate_normal(means, cov, size=n_mc, method="svd")

        stds = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        return h_samples, filters, means, stds

    X_scaled = scaler_X.transform(X.reshape(1, -1))
    means, stds = [], []
    for filt in filters:
        mean, std = gp_models[filt].predict(X_scaled, return_std=True)
        means.append(float(mean[0]))
        stds.append(max(float(std[0]), 1e-9))

    means = np.asarray(means)
    stds = np.asarray(stds)
    rng = np.random.default_rng(random_state)
    z = rng.standard_normal(size=(n_mc, len(filters)))
    h_samples = means + z * stds
    return h_samples, filters, means, stds


def expected_detection_probability_cf(
    X: np.ndarray,
    gp_models: Dict,
    scaler_X: StandardScaler,
    flat_limits: Dict,
    detection_calc: DetectionCalculator,
    n_mc: int = DEFAULT_EICF_MC_SAMPLES,
) -> float:
    """Monte Carlo estimate of E[g(h(x))] under the GP posterior on h."""
    h_samples, filters, _, _ = sample_h_posterior(
        X, gp_models, scaler_X, flat_limits, n_mc=n_mc
    )
    probs = composite_detection_probability(h_samples, filters, flat_limits, detection_calc)
    return float(np.mean(probs)) if len(probs) else 0.0


def expected_improvement_composite(
    X: np.ndarray,
    gp_models: Dict,
    scaler_X: StandardScaler,
    flat_limits: Dict,
    best_score: float,
    detection_calc: DetectionCalculator,
    n_mc: int = DEFAULT_EICF_MC_SAMPLES,
    xi: float = 0.0,
    verbose: bool = False,
) -> float:
    """
    Algorithm-1-style EI-CF for h(x) = peak magnitudes and g(h) = P_detect.

    EI-CF_n(x) = E_n[{g(h(x)) - f*_n - xi}+].
    """
    h_samples, filters, means, stds = sample_h_posterior(
        X, gp_models, scaler_X, flat_limits, n_mc=n_mc
    )
    if not filters:
        return 0.0

    sampled_probs = composite_detection_probability(
        h_samples, filters, flat_limits, detection_calc
    )
    improvements = np.maximum(sampled_probs - best_score - xi, 0.0)
    ei_cf = float(np.mean(improvements))

    if verbose:
        mean_prob = float(np.mean(sampled_probs))
        print(f"      EI-CF filters={filters}, mean_h={dict(zip(filters, means))}")
        print(f"      EI-CF std_h={dict(zip(filters, stds))}")
        print(f"      EI-CF: E[P_detect]={mean_prob:.4f}, best={best_score:.4f}, "
              f"xi={xi:.4f}, EI-CF={ei_cf:.6f}")

    return ei_cf


def objective_with_uncertainty(
    X: np.ndarray,
    gp_models: Dict,
    scaler_X: StandardScaler,
    flat_limits: Dict,
    detection_calc: DetectionCalculator,
    verbose: bool = False
) -> Tuple[float, float, str]:
    """
    Active-search utility from the ZTF limiting-magnitude CDF.

    Returns: (P_detect, limit_uncertainty_for_best_filter, best_filter)
    """
    X_scaled = scaler_X.transform(X.reshape(1, -1))

    best_prob = -np.inf
    best_filter = None
    best_uncertainty = 0.0

    filters = _available_detection_filters(gp_models, flat_limits)
    multitask_gp = gp_models.get(MULTITASK_GP_KEY)
    if multitask_gp is not None and getattr(multitask_gp, "available", False):
        means, cov, ordered_filters = multitask_gp.posterior_h(X, filters=filters)
        stds = np.sqrt(np.maximum(np.diag(cov), 1e-12))
        for filt, pred_mean, pred_std in zip(ordered_filters, means, stds):
            limit = flat_limits.get(filt)
            if limit is None:
                continue

            limit_sigma = detection_calc.filter_errors.get(filt, 0.1)
            detection_prob = norm.cdf((limit - pred_mean) / limit_sigma)

            if verbose:
                print(f"      Filter {filt}: mag={pred_mean:.3f}, limit={limit:.3f}, "
                      f"limit_sigma={limit_sigma:.4f}, P_detect={detection_prob:.4f}, "
                      f"MultiTaskGP_std={pred_std:.4f} mag")

            if detection_prob > best_prob:
                best_prob = detection_prob
                best_filter = filt
                best_uncertainty = limit_sigma

        if verbose:
            print(f"      Best filter: {best_filter}, P_detect={best_prob:.4f}, "
                  f"limit_sigma={best_uncertainty:.4f}")

        return best_prob, best_uncertainty, best_filter

    for filt in filters:
        gp = gp_models.get(filt)
        if gp is None:
            continue

        pred_mean, pred_std = gp.predict(X_scaled, return_std=True)
        pred_mean = pred_mean[0]
        pred_std = pred_std[0]

        limit = flat_limits.get(filt)
        if limit is None:
            continue

        limit_sigma = detection_calc.filter_errors.get(filt, 0.1)
        detection_prob = norm.cdf((limit - pred_mean) / limit_sigma)

        if verbose:
            print(f"      Filter {filt}: mag={pred_mean:.3f}, limit={limit:.3f}, "
                  f"limit_sigma={limit_sigma:.4f}, P_detect={detection_prob:.4f}, "
                  f"GP_std={pred_std:.4f} mag")

        if detection_prob > best_prob:
            best_prob = detection_prob
            best_filter = filt
            best_uncertainty = limit_sigma

    if verbose:
        print(f"      Best filter: {best_filter}, P_detect={best_prob:.4f}, "
              f"limit_sigma={best_uncertainty:.4f}")

    return best_prob, best_uncertainty, best_filter

def expected_improvement(
    X: np.ndarray,
    gp_models: Dict,
    scaler_X: StandardScaler,
    flat_limits: Dict,
    best_score: float,
    detection_gp: Optional = None,
    use_gp_detection: bool = False,
    xi: float = 0.01,
    detection_calc: Optional[DetectionCalculator] = None,
    n_mc: int = DEFAULT_EICF_MC_SAMPLES,
    active_search: bool = False,
    verbose: bool = False
) -> float:
    """
    Expected Improvement for Composite Functions (EI-CF).

    Here h(x) is the 3D vector of GP-predicted peak magnitudes in g/r/i, and
    g(h) is the ZTF CDF detection probability.
    """
    if detection_calc is None:
        detection_calc = DetectionCalculator(flat_limits=flat_limits)
    
    if active_search:
        score = expected_detection_probability_cf(
            X, gp_models, scaler_X, flat_limits,
            detection_calc=detection_calc, n_mc=n_mc,
        )
    else:
        score = expected_improvement_composite(
            X, gp_models, scaler_X, flat_limits, best_score,
            detection_calc=detection_calc, n_mc=n_mc,
            xi=xi, verbose=verbose
        )
    
    if score is None or np.isnan(score) or np.isinf(score):
        return 0.0
    
    if verbose:
        print(f"      Active-search utility: EI-CF={score:.6f}")

    return max(float(score), 0.0)


def upper_confidence_bound(
    X: np.ndarray,
    gp_models: Dict,
    scaler_X: StandardScaler,
    flat_limits: Dict,
    detection_gp: Optional = None,
    use_gp_detection: bool = False,
    kappa: float = 2.0,
    detection_calc: Optional[DetectionCalculator] = None,
    verbose: bool = False
) -> float:
    """
    Monte Carlo UCB on the composite detection probability.

    This is retained for compatibility, but the EI acquisition is the direct
    implementation of Algorithm 1 EI-CF.
    """
    if detection_calc is None:
        detection_calc = DetectionCalculator(flat_limits=flat_limits)
    
    h_samples, filters, _, _ = sample_h_posterior(
        X, gp_models, scaler_X, flat_limits, n_mc=DEFAULT_EICF_MC_SAMPLES
    )
    sampled_probs = composite_detection_probability(
        h_samples, filters, flat_limits, detection_calc
    )
    score = float(np.mean(sampled_probs) + kappa * np.std(sampled_probs)) if len(sampled_probs) else 0.0
    
    if score is None or np.isnan(score) or np.isinf(score):
        return -np.inf
    
    if verbose:
        print(f"      Composite UCB utility: {score:.6f}")
    
    return float(np.clip(score, 0.0, 1.0))


def sample_near_detectable(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    bounds: list,
    n_samples: int = 2000,
    spread_factor: float = 0.25,
    n_uniform: int = 300,
    iteration: int = 0,
    max_iterations: int = 10,
) -> np.ndarray:
    """
    Sample candidate points centered on the distribution of known-detectable
    events. Includes adaptive exploration: more exploration early, more exploitation late.
    
    Out-of-bounds jitter is REFLECTED back into the valid range rather than
    clipped, to avoid candidates piling up exactly on the boundary value.
    """
    lows = np.array([b[0] for b in bounds], dtype=float)
    highs = np.array([b[1] for b in bounds], dtype=float)
    ranges = highs - lows
    
    # Adaptive exploration: more exploration early, more exploitation late
    progress = iteration / max_iterations if max_iterations > 0 else 0
    adaptive_spread = spread_factor * (0.5 + 0.5 * (1 - progress))  # Starts at 1.0x, goes to 0.5x
    
    # Adjust uniform samples: more early, fewer late
    adaptive_uniform = int(n_uniform * (1 + (1 - progress)))  # More uniform early
    
    detectable_mask = y_pool == 1
    X_detectable = X_pool[detectable_mask]
    
    if len(X_detectable) == 0:
        # fallback: no detectable points yet, sample uniformly only
        print(f"  No detectable points yet - sampling uniformly")
        return lows + np.random.uniform(size=(n_samples, len(bounds))) * ranges
    
    # per-dimension spread of the detectable population
    stds = np.std(X_detectable, axis=0)
    stds = np.maximum(stds, adaptive_spread * ranges * 0.1)
    
    # Pick base points - biased toward BEST detectable points
    # Give higher weight to better detections (closer = brighter)
    if len(X_detectable) > 1:
        # Weight by inverse magnitude (brighter = higher weight)
        # Use distance as proxy for brightness (closer = brighter)
        weights = 1.0 / (np.abs(X_detectable[:, -1]) + 1)  # Distance column (last)
        weights = weights / weights.sum()
        idx = np.random.choice(len(X_detectable), size=n_samples, p=weights)
    else:
        idx = np.random.randint(0, len(X_detectable), size=n_samples)
    
    base_points = X_detectable[idx]
    jitter = np.random.normal(scale=stds * adaptive_spread, size=(n_samples, len(bounds)))
    
    X_near = base_points + jitter
    
    # Reflect out-of-bounds values instead of clipping, to avoid pileup at the edges
    for dim in range(X_near.shape[1]):
        lo, hi = lows[dim], highs[dim]
        col = X_near[:, dim]
        below = col < lo
        col[below] = lo + (lo - col[below])
        above = col > hi
        col[above] = hi - (col[above] - hi)
        # safety net in case reflection itself overshoots the opposite bound
        col = np.clip(col, lo, hi)
        X_near[:, dim] = col
    
    # Mix in uniform samples for exploration
    n_uniform_actual = min(adaptive_uniform, n_samples)
    X_uniform = lows + np.random.uniform(size=(n_uniform_actual, len(bounds))) * ranges
    
    # If we have too many uniform, trim; if too few, add more near samples
    if n_uniform_actual < n_uniform:
        extra_near = n_uniform - n_uniform_actual
        extra_idx = np.random.randint(0, len(X_detectable), size=extra_near)
        extra_base = X_detectable[extra_idx]
        extra_jitter = np.random.normal(scale=stds * adaptive_spread, size=(extra_near, len(bounds)))
        X_extra = extra_base + extra_jitter
        # Reflect
        for dim in range(X_extra.shape[1]):
            lo, hi = lows[dim], highs[dim]
            col = X_extra[:, dim]
            below = col < lo
            col[below] = lo + (lo - col[below])
            above = col > hi
            col[above] = hi - (col[above] - hi)
            col = np.clip(col, lo, hi)
            X_extra[:, dim] = col
        X_candidates = np.vstack([X_near, X_uniform, X_extra])
    else:
        X_candidates = np.vstack([X_near, X_uniform[:n_uniform_actual]])
    
    # Ensure we have exactly n_samples (or close)
    if len(X_candidates) > n_samples:
        X_candidates = X_candidates[:n_samples]
    
    print(f"  Sampling: {len(X_near)} near detections, {n_uniform_actual} uniform, "
          f"spread_factor={adaptive_spread:.3f}")
    
    return X_candidates
