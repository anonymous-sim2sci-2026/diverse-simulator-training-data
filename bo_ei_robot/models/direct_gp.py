"""
Direct scalar GP baseline for g(x) = P_detect.

This is the non-gray-box comparison model: instead of fitting a GP to the
intermediate magnitudes h(x), it fits one GP directly to the scalar detection
probability g(x).
"""

import numpy as np
import time
import warnings
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
    for i in range(n_rows):
        peak_mags = {
            filt: float(arr[i])
            for filt, arr in mag_arrays.items()
            if np.isfinite(arr[i])
        }
        prob_detect, _ = detection_calc.compute_detection_probability(peak_mags)
        g_targets[i] = prob_detect

    return np.clip(g_targets, 0.0, 1.0)


def _make_direct_gp():
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-6, 1e0))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=5,
        alpha=1e-6,
        normalize_y=True,
    )


def build_direct_gp_model(X_train, g_train):
    """Fit the initial scalar GP on g(x)=P_detect."""
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(np.asarray(X_train, dtype=float))
    g_arr = np.clip(np.asarray(g_train, dtype=float), 0.0, 1.0)

    direct_gp = _make_direct_gp()
    t0 = time.time()
    print(
        f"  Fitting full direct scalar GP on {len(X_scaled)} "
        "g(x)=P_detect targets...",
        flush=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        direct_gp.fit(X_scaled, g_arr)
    print(f"  Direct scalar GP fit finished in {time.time() - t0:.1f} s", flush=True)
    return direct_gp, scaler_X


def fit_direct_gp_with_scaler(X_train, g_train, scaler_X):
    """
    Refit the scalar GP after adding a new NMMA point.

    The scaler is kept fixed so every prediction uses the same feature
    transform as the original direct-GP training.
    """
    direct_gp = _make_direct_gp()
    X_scaled = scaler_X.transform(np.asarray(X_train, dtype=float))
    g_arr = np.clip(np.asarray(g_train, dtype=float), 0.0, 1.0)
    t0 = time.time()
    print(
        f"      Fitting full direct scalar GP on {len(X_scaled)} "
        "evaluated g(x) values...",
        flush=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        direct_gp.fit(X_scaled, g_arr)
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
