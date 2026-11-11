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


def expected_improvement_composite_batch(
    X: np.ndarray,
    gp_models: Dict,
    scaler_X: StandardScaler,
    flat_limits: Dict,
    best_score: float,
    detection_calc: DetectionCalculator,
    n_mc: int = 16,
    xi: float = 0.0,
    batch_size: int = 128,
    random_state: int = 12345,
    progress_every: int = 1,
    active_search: bool = False,
) -> np.ndarray:
    """
    Batch EI-CF for many candidates.

    This samples only the h dimensions used by g(h), i.e. ZTF g/r/i peak
    magnitudes. ROBOT still uses the larger h-vector for diversity, but EI-CF
    does not need to sample SNR, area90, ejecta masses, or class flags.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    filters = _available_detection_filters(gp_models, flat_limits)
    if len(X) == 0 or not filters:
        return np.zeros(len(X), dtype=float)

    scores = np.zeros(len(X), dtype=float)
    rng = np.random.default_rng(random_state)
    multitask_gp = gp_models.get(MULTITASK_GP_KEY)

    if multitask_gp is not None and getattr(multitask_gp, "available", False):
        filters = [f for f in filters if f in multitask_gp.filter_to_task]
        n_filters = len(filters)
        if n_filters == 0:
            return scores

        import torch
        import gpytorch

        n_chunks = int(np.ceil(len(X) / batch_size))
        for chunk_idx, start in enumerate(range(0, len(X), batch_size), start=1):
            stop = min(start + batch_size, len(X))
            chunk = X[start:stop]
            test_X, _ = multitask_gp._make_task_test_X(chunk, filters)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                posterior = multitask_gp.model.posterior(test_X)

            means = posterior.mean.detach().cpu().numpy().reshape(-1)
            cov = posterior.mvn.covariance_matrix.detach().cpu().numpy()
            cov = np.asarray(cov, dtype=float)
            cov = 0.5 * (cov + cov.T)

            for local_i in range(len(chunk)):
                a = local_i * n_filters
                b = a + n_filters
                mean_i = means[a:b]
                cov_i = cov[a:b, a:b] + 1e-9 * np.eye(n_filters)
                try:
                    chol = np.linalg.cholesky(cov_i)
                    z = rng.standard_normal(size=(n_mc, n_filters))
                    h_samples = mean_i + z @ chol.T
                except np.linalg.LinAlgError:
                    h_samples = rng.multivariate_normal(mean_i, cov_i, size=n_mc, method="svd")
                sampled_probs = composite_detection_probability(
                    h_samples, filters, flat_limits, detection_calc
                )
                if active_search:
                    scores[start + local_i] = float(np.mean(sampled_probs))
                else:
                    improvements = np.maximum(sampled_probs - best_score - xi, 0.0)
                    scores[start + local_i] = float(np.mean(improvements))

            if progress_every and (chunk_idx == 1 or chunk_idx % progress_every == 0 or stop == len(X)):
                label = "active-search composite" if active_search else "EI-CF"
                print(f"        {label} batch scored {stop}/{len(X)} candidates", flush=True)

        return scores

    X_scaled = scaler_X.transform(X)
    means = []
    stds = []
    used_filters = []
    for filt in filters:
        gp = gp_models.get(filt)
        if gp is None:
            continue
        mean, std = gp.predict(X_scaled, return_std=True)
        means.append(np.asarray(mean, dtype=float))
        stds.append(np.maximum(np.asarray(std, dtype=float), 1e-9))
        used_filters.append(filt)

    if not means:
        return scores

    means = np.column_stack(means)
    stds = np.column_stack(stds)
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        z = rng.standard_normal(size=(stop - start, n_mc, len(used_filters)))
        h_samples = means[start:stop, None, :] + z * stds[start:stop, None, :]

        prob_cols = []
        for j, filt in enumerate(used_filters):
            limit = flat_limits.get(filt)
            if limit is None:
                continue
            sigma = detection_calc.filter_errors.get(filt, 0.1)
            prob_cols.append(norm.cdf((limit - h_samples[:, :, j]) / sigma))
        if prob_cols:
            sampled_probs = np.max(np.stack(prob_cols, axis=2), axis=2)
            if active_search:
                scores[start:stop] = np.mean(sampled_probs, axis=1)
            else:
                improvements = np.maximum(sampled_probs - best_score - xi, 0.0)
                scores[start:stop] = np.mean(improvements, axis=1)

        if progress_every:
            label = "active-search composite" if active_search else "EI-CF"
            print(f"        {label} batch scored {stop}/{len(X)} candidates", flush=True)

    return scores


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
    n_mc: int = DEFAULT_EICF_MC_SAMPLES,
    detection_calc: Optional[DetectionCalculator] = None,
    verbose: bool = False
) -> float:
    """
    Expected Improvement for Composite Functions (EI-CF).

    Here h(x) is the 3D vector of GP-predicted peak magnitudes in g/r/i, and
    g(h) is the ZTF CDF detection probability.
    """
    if detection_calc is None:
        detection_calc = DetectionCalculator(flat_limits=flat_limits)
    
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
    
    # Use more global exploration early, then taper to the requested uniform
    # subset. n_samples is the total candidate-pool size, not the number of
    # nearby candidates.
    adaptive_uniform = int(n_uniform * (1 + (1 - progress)))
    n_uniform_actual = min(max(adaptive_uniform, 0), n_samples)
    n_near = n_samples - n_uniform_actual
    
    detectable_mask = y_pool == 1
    X_detectable = X_pool[detectable_mask]
    
    if len(X_detectable) == 0:
        # fallback: no detectable points yet, sample uniformly only
        print(f"  No detectable points yet - sampling uniformly")
        return lows + np.random.uniform(size=(n_samples, len(bounds))) * ranges
    
    # per-dimension spread of the detectable population
    stds = np.std(X_detectable, axis=0)
    stds = np.maximum(stds, adaptive_spread * ranges * 0.1)
    
    # Pick nearby base points evenly across occupied distance bins. This avoids
    # letting the much larger nearby population dominate candidate generation.
    if n_near > 0:
        if len(X_detectable) > 1:
            n_distance_bins = min(10, len(X_detectable))
            distance_edges = np.linspace(lows[-1], highs[-1], n_distance_bins + 1)
            distance_bins = np.clip(
                np.digitize(X_detectable[:, -1], distance_edges[1:-1], right=False),
                0,
                n_distance_bins - 1,
            )
            occupied_bins = np.unique(distance_bins)
            sampled_bins = np.random.choice(occupied_bins, size=n_near, replace=True)
            idx = np.empty(n_near, dtype=int)
            for distance_bin in occupied_bins:
                target = np.flatnonzero(sampled_bins == distance_bin)
                members = np.flatnonzero(distance_bins == distance_bin)
                if len(target):
                    idx[target] = np.random.choice(members, size=len(target), replace=True)
        else:
            idx = np.random.randint(0, len(X_detectable), size=n_near)

        base_points = X_detectable[idx]
        jitter = np.random.normal(
            scale=stds * adaptive_spread,
            size=(n_near, len(bounds)),
        )
        X_near = base_points + jitter
    else:
        X_near = np.empty((0, len(bounds)), dtype=float)
    
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
    
    # Mix in uniform samples for global exploration.
    X_uniform = lows + np.random.uniform(size=(n_uniform_actual, len(bounds))) * ranges

    X_candidates = np.vstack([X_near, X_uniform])
    if len(X_candidates) != n_samples:
        raise RuntimeError(
            f"Candidate sampler created {len(X_candidates)} points; expected {n_samples}."
        )

    # Shuffle so candidate ordering cannot favor either sampling component.
    np.random.shuffle(X_candidates)

    print(
        f"  Sampling: {n_near} near detections + {n_uniform_actual} uniform "
        f"= {len(X_candidates)} total, spread_factor={adaptive_spread:.3f}, "
        "near_sampling=balanced_distance_bins"
    )
    
    return X_candidates


def sample_around_robot_center(
    center: np.ndarray,
    bounds: list,
    n_samples: int,
    spread_factor: float = 0.12,
) -> np.ndarray:
    """Sample inputs around the evaluated x associated with an h-space center."""
    center = np.asarray(center, dtype=float)
    lows = np.asarray([bound[0] for bound in bounds], dtype=float)
    highs = np.asarray([bound[1] for bound in bounds], dtype=float)
    ranges = highs - lows
    samples = center + np.random.normal(
        scale=np.maximum(spread_factor * ranges, 1e-12),
        size=(int(n_samples), len(bounds)),
    )
    for dim, (low, high) in enumerate(zip(lows, highs)):
        width = high - low
        if width <= 0:
            samples[:, dim] = low
            continue
        samples[:, dim] = low + np.mod(samples[:, dim] - low, 2.0 * width)
        over = samples[:, dim] > high
        samples[over, dim] = high - (samples[over, dim] - high)
    return np.clip(samples, lows, highs)
