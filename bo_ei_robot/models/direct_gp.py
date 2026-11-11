"""
Direct scalar GP baseline for g(x) = P_detect.

This is the non-gray-box comparison model: instead of fitting a GP to the
intermediate magnitudes h(x), it fits one GP directly to the scalar detection
probability g(x).
"""

import numpy as np
import time
import warnings
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


def compute_g_targets_from_magnitudes(y_train_mags, detection_calc, detection_filters):
    """
    Convert training peak magnitudes into scalar g(x)=P_detect targets.

    P_detect is computed with the same ZTF limiting-magnitude CDF used by the
    gray-box code:

        P_detect = max_f Phi((L_f - m_f) / sigma_f)
    """
    mag_arrays = {}
    n_rows = None

    for filt in detection_filters:
        if filt not in y_train_mags:
            continue

        values = y_train_mags[filt]
        arr = values.to_numpy(dtype=float) if hasattr(values, "to_numpy") else np.asarray(values, dtype=float)
        mag_arrays[filt] = arr
        n_rows = len(arr) if n_rows is None else min(n_rows, len(arr))

    if n_rows is None:
        raise ValueError("No magnitude columns are available to compute direct g(x) targets.")

    g_targets = np.zeros(n_rows, dtype=float)
    for filt, arr in mag_arrays.items():
        if filt not in detection_calc.flat_limits:
            continue
        vals = np.asarray(arr[:n_rows], dtype=float)
        valid = np.isfinite(vals)
        if not np.any(valid):
            continue
        limit = detection_calc.flat_limits[filt]
        sigma = detection_calc.filter_errors.get(filt, 0.1)
        filt_probs = np.zeros(n_rows, dtype=float)
        filt_probs[valid] = norm.cdf((limit - vals[valid]) / sigma)
        g_targets = np.maximum(g_targets, filt_probs)

    return np.clip(g_targets, 0.0, 1.0)


def _make_direct_gp(n_restarts_optimizer=5):
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-6, 1e0))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=int(n_restarts_optimizer),
        alpha=1e-6,
        normalize_y=True,
    )


def _subsample_for_fit(X_scaled, g_arr, max_fit_points=None, random_state=12345):
    """Optionally cap exact GP training size for long exploratory runs."""
    if max_fit_points is None or max_fit_points <= 0 or len(X_scaled) <= max_fit_points:
        return X_scaled, g_arr

    max_fit_points = int(max_fit_points)
    rng = np.random.default_rng(random_state)
    n_keep_best = min(max_fit_points // 2, len(X_scaled))
    best_idx = np.argsort(-g_arr)[:n_keep_best]
    remaining = np.setdiff1d(np.arange(len(X_scaled)), best_idx, assume_unique=False)
    n_random = max_fit_points - len(best_idx)
    if n_random > 0 and len(remaining) > 0:
        random_idx = rng.choice(remaining, size=min(n_random, len(remaining)), replace=False)
        idx = np.concatenate([best_idx, random_idx])
    else:
        idx = best_idx
    return X_scaled[idx], g_arr[idx]


def build_direct_gp_model(
    X_train,
    g_train,
    n_restarts_optimizer=5,
    max_fit_points=None,
    random_state=12345,
):
    """Fit the initial scalar GP on g(x)=P_detect."""
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(np.asarray(X_train, dtype=float))
    g_arr = np.clip(np.asarray(g_train, dtype=float), 0.0, 1.0)
    X_fit, g_fit = _subsample_for_fit(
        X_scaled, g_arr, max_fit_points=max_fit_points, random_state=random_state
    )

    direct_gp = _make_direct_gp(n_restarts_optimizer=n_restarts_optimizer)
    t0 = time.time()
    print(
        f"  Fitting direct scalar GP on {len(X_fit)}/{len(X_scaled)} "
        "g(x)=P_detect targets...",
        flush=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        direct_gp.fit(X_fit, g_fit)
    print(f"  Direct scalar GP fit finished in {time.time() - t0:.1f} s", flush=True)
    return direct_gp, scaler_X


def fit_direct_gp_with_scaler(
    X_train,
    g_train,
    scaler_X,
    n_restarts_optimizer=5,
    max_fit_points=None,
    random_state=12345,
):
    """
    Refit the scalar GP after adding a new NMMA point.

    The scaler is kept fixed so every prediction uses the same feature
    transform as the original direct-GP training.
    """
    direct_gp = _make_direct_gp(n_restarts_optimizer=n_restarts_optimizer)
    X_scaled = scaler_X.transform(np.asarray(X_train, dtype=float))
    g_arr = np.clip(np.asarray(g_train, dtype=float), 0.0, 1.0)
    X_fit, g_fit = _subsample_for_fit(
        X_scaled, g_arr, max_fit_points=max_fit_points, random_state=random_state
    )
    t0 = time.time()
    print(
        f"      Fitting direct scalar GP on {len(X_fit)}/{len(X_scaled)} "
        "evaluated g(x) values...",
        flush=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        direct_gp.fit(X_fit, g_fit)
    print(f"      Direct scalar GP fit finished in {time.time() - t0:.1f} s", flush=True)
    return direct_gp


def predict_direct_gp(X, direct_gp, scaler_X):
    """Return posterior mean/std for scalar g(x), clipped to probability range."""
    X_arr = np.asarray(X, dtype=float).reshape(1, -1)
    X_scaled = scaler_X.transform(X_arr)
    mean, std = direct_gp.predict(X_scaled, return_std=True)
    return float(np.clip(mean[0], 0.0, 1.0)), float(max(std[0], 1e-12))


def predict_direct_gp_batch(X, direct_gp, scaler_X):
    """Vectorized posterior mean/std for scalar g(x)."""
    X_arr = np.asarray(X, dtype=float)
    X_scaled = scaler_X.transform(np.atleast_2d(X_arr))
    mean, std = direct_gp.predict(X_scaled, return_std=True)
    return np.clip(mean, 0.0, 1.0), np.maximum(std, 1e-12)
