"""
Gaussian Process surrogate models
"""

import warnings
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, Matern
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score
from typing import Dict, Tuple, Optional

from .multitask_gp import build_multitask_magnitude_gp
from utils.h_columns import H_OUTPUT_COLUMNS


MULTITASK_GP_KEY = "__multitask_magnitude_gp__"


def build_fast_h_diagnostic_models(
    X_train: np.ndarray,
    y_train_mags: Dict,
    max_fit_points: int = 350,
    random_state: int = 12345,
) -> Tuple[Dict, StandardScaler, None]:
    """
    Build lightweight independent h(x) surrogates for DIRECT-GP diagnostics.

    These models are used only to fill new-event columns like snr, area90,
    HasNS, HasMassGap, and HasRemnant. They are not used by DIRECT_EI itself.
    """
    scaler_X = StandardScaler()
    X_scaled_all = scaler_X.fit_transform(np.asarray(X_train, dtype=float))
    rng = np.random.default_rng(random_state)

    kernel = (
        ConstantKernel(1.0, constant_value_bounds="fixed")
        * Matern(length_scale=1.0, nu=2.5, length_scale_bounds="fixed")
        + WhiteKernel(noise_level=0.05, noise_level_bounds="fixed")
    )

    gp_models = {}
    t0 = time.time()
    print(
        f"  Building fast diagnostic h(x) surrogates "
        f"(max {max_fit_points} points/output)...",
        flush=True,
    )

    for output_name in H_OUTPUT_COLUMNS:
        if output_name not in y_train_mags:
            gp_models[output_name] = None
            continue

        values = y_train_mags[output_name]
        y_output = values.to_numpy(dtype=float) if hasattr(values, "to_numpy") else np.asarray(values, dtype=float)
        mask = np.isfinite(y_output)
        valid_idx = np.where(mask)[0]

        if len(valid_idx) <= 10:
            print(f"  Warning: not enough data for diagnostic h output {output_name}")
            gp_models[output_name] = None
            continue

        if len(valid_idx) > max_fit_points:
            fit_idx = rng.choice(valid_idx, size=max_fit_points, replace=False)
        else:
            fit_idx = valid_idx
        fit_idx = np.sort(fit_idx)

        gp = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=None,
            n_restarts_optimizer=0,
            alpha=1e-5,
            normalize_y=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gp.fit(X_scaled_all[fit_idx], y_output[fit_idx])

        gp_models[output_name] = gp
        print(f"    h[{output_name}] fit on {len(fit_idx)}/{len(valid_idx)} points", flush=True)

    print(f"  Fast diagnostic h(x) surrogates finished in {time.time() - t0:.1f} s", flush=True)
    return gp_models, scaler_X, None


def build_gp_models(
    X_train: np.ndarray, 
    y_train_mags: Dict, 
    y_train_detection: Optional[np.ndarray] = None
) -> Tuple[Dict, StandardScaler, Optional[GaussianProcessRegressor]]:
    """
    Build one GP per h(x) output + one GP for binary detection.
    
    Args:
        X_train: Training features
        y_train_mags: Dictionary of magnitude arrays per filter
        y_train_detection: Detection labels (optional)
        
    Returns:
        gp_models: Dictionary of GP models per filter
        scaler_X: StandardScaler fitted to X
        detection_gp: GP for binary detection (or None)
    """
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_train)

    gp_models = {}
    multitask_outputs = [col for col in H_OUTPUT_COLUMNS if col in y_train_mags]
    if len(multitask_outputs) >= 2:
        print(f"\nBuilding BoTorch MultiTaskGP for 11D h(x): {multitask_outputs}")
        multitask_gp = build_multitask_magnitude_gp(
            X_train, y_train_mags, filters=multitask_outputs
        )
        if multitask_gp.available:
            gp_models[MULTITASK_GP_KEY] = multitask_gp
            print(f"MultiTaskGP built successfully for h outputs: {multitask_outputs}")
        else:
            print("Warning: MultiTaskGP unavailable; using independent sklearn GPs.")
            if multitask_gp.error_message:
                print(f"  Reason: {multitask_gp.error_message}")

    if MULTITASK_GP_KEY not in gp_models:
        kernel = (
            ConstantKernel(1.0, (1e-4, 1e4))
            * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-3, 1e4))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1))
        )

        for filter_name in y_train_mags:
            y_filter = y_train_mags[filter_name].values
            mask = ~np.isnan(y_filter)
            X_filter, y_filter = X_scaled[mask], y_filter[mask]

            if len(X_filter) <= 10:
                print(f"Warning: Not enough data for h output {filter_name}")
                gp_models[filter_name] = None
                continue

            gp = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=1, alpha=1e-6, normalize_y=True
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                gp.fit(X_filter, y_filter)
            gp_models[filter_name] = gp

            r2 = r2_score(y_filter, gp.predict(X_filter))
            print(f"Built fallback sklearn GP for h output {filter_name} with {len(X_filter)} training points")
            print(f"  R² score: {r2:.4f}")

    detection_gp = None
    if y_train_detection is not None:
        print("\nBuilding GP for binary detection...")
        detection_gp = GaussianProcessRegressor(
            kernel=(
                ConstantKernel(1.0, (1e-4, 1e4))
                * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e4), nu=2.5)
                + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1))
            ),
            n_restarts_optimizer=1,
            alpha=1e-6,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            detection_gp.fit(X_scaled, y_train_detection)
        print("Detection GP built successfully")

    return gp_models, scaler_X, detection_gp


def predict_magnitudes_batch(
    X_new: np.ndarray, 
    gp_models: Dict, 
    scaler_X: StandardScaler
) -> Tuple[Dict, Dict]:
    """
    Vectorized magnitude prediction for a batch of points.
    
    Args:
        X_new: New points to predict
        gp_models: Dictionary of GP models
        scaler_X: Fitted StandardScaler
        
    Returns:
        magnitudes: Dictionary of magnitude arrays per filter
        uncertainties: Dictionary of uncertainty arrays per filter
    """
    X_scaled = scaler_X.transform(np.atleast_2d(X_new))
    magnitudes, uncertainties = {}, {}

    multitask_gp = gp_models.get(MULTITASK_GP_KEY)
    if multitask_gp is not None and getattr(multitask_gp, "available", False):
        mt_mags, mt_uncs = multitask_gp.predict_batch(
            np.atleast_2d(X_new),
            filters=[f for f in H_OUTPUT_COLUMNS if f in multitask_gp.filters],
        )
        magnitudes.update(mt_mags)
        uncertainties.update(mt_uncs)

    for filter_name, gp in gp_models.items():
        if filter_name == MULTITASK_GP_KEY or filter_name in magnitudes:
            continue
        if gp is not None:
            mean, std = gp.predict(X_scaled, return_std=True)
            magnitudes[filter_name] = mean
            uncertainties[filter_name] = std
    return magnitudes, uncertainties


def predict_magnitudes_single(
    X_new: np.ndarray, 
    gp_models: Dict, 
    scaler_X: StandardScaler
) -> Tuple[Dict, Dict]:
    """
    Single-point wrapper for magnitude prediction.
    
    Args:
        X_new: Single point to predict
        gp_models: Dictionary of GP models
        scaler_X: Fitted StandardScaler
        
    Returns:
        magnitudes: Dictionary of magnitudes per filter
        uncertainties: Dictionary of uncertainties per filter
    """
    mags, uncs = predict_magnitudes_batch(X_new, gp_models, scaler_X)
    return ({f: v[0] for f, v in mags.items()},
            {f: v[0] for f, v in uncs.items()})
