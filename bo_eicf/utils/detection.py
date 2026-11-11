"""
Detection calculation utilities
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple, Optional


class DetectionCalculator:
    """Handles detection probability calculations"""
    
    def __init__(self, detection_filters=None, flat_limits=None):
        self.detection_filters = detection_filters or ['g', 'i', 'r', 'u', 'y', 'z']
        
        # FIX: Initialize flat_limits properly with default ZTF limits
        if flat_limits is not None:
            self.flat_limits = flat_limits
        else:
            # Default ZTF + Rubin limits
            self.flat_limits = {
                'g': 21.067,
                'r': 21.012,
                'i': 20.510,
                'u': 23.9,
                'y': 22.1,
                'z': 23.3,
            }
        
        # Initialize filter_errors with default values
        self.filter_errors = {
            'g': 0.003,
            'r': 0.002,
            'i': 0.010,
            'u': 0.01,
            'y': 0.01,
            'z': 0.01,
        }
        
        # Override with any errors from flat_limits if they contain error info
        if flat_limits is not None:
            for filt, val in flat_limits.items():
                if isinstance(val, dict) and 'error' in val:
                    self.filter_errors[filt] = val['error']
                elif isinstance(val, (int, float)):
                    # If just a number, keep default error
                    pass
    
    def define_detection_limits(self) -> Dict:
        """Raw, per-survey magnitude limits (ZTF + Rubin) with actual ZTF errors"""
        return {
            'ZTF': {
                'g': {'limit': 21.067, 'error': 0.003},
                'r': {'limit': 21.012, 'error': 0.002},
                'i': {'limit': 20.510, 'error': 0.010},
                'errors': {'g': 0.003, 'r': 0.002, 'i': 0.010},
            },
            'Rubin': {
                'u': {'limit': 23.9, 'error': 0.01},
                'y': {'limit': 22.1, 'error': 0.01},
                'z': {'limit': 23.3, 'error': 0.01},
            },
        }
    
    def flatten_limits(self, limits: Dict) -> Dict:
        """Flatten nested survey limits into a {filter: value} dict.

        Only filters requested by ``self.detection_filters`` are included. This
        keeps the BO objective aligned with the filters that NMMA actually
        simulates, e.g. ztfg/ztfr/ztfi -> g/r/i.
        """
        flat = {}
        self.filter_errors = {}
        requested_filters = set(self.detection_filters)
        for survey_limits in limits.values():
            if isinstance(survey_limits, dict):
                for filt, val in survey_limits.items():
                    if filt == 'errors' or filt not in requested_filters:
                        continue
                    if isinstance(val, dict):
                        flat[filt] = val['limit']
                        if 'error' in val:
                            self.filter_errors[filt] = val['error']
                    else:
                        flat[filt] = val
        self.flat_limits = flat
        return flat
    
    def get_filter_errors(self, limits: Dict) -> Dict:
        """Get filter errors from the limits dictionary"""
        errors = {}
        for survey_limits in limits.values():
            if isinstance(survey_limits, dict):
                if 'errors' in survey_limits:
                    errors.update(survey_limits['errors'])
                for filt, val in survey_limits.items():
                    if filt != 'errors' and isinstance(val, dict) and 'error' in val:
                        errors[filt] = val['error']
        return errors
    
    def compute_detection_status(self, magnitudes: Dict) -> Tuple[int, Dict]:
        """
        Binary detection status (1 or 0) for an event.
        An event is detected if it is brighter than the limit in ANY filter.
        Brighter = smaller magnitude number.
        """
        detection_details = {}
        
        # If no magnitudes, not detected
        if not magnitudes:
            return 0, {}
        
        for filter_name in self.detection_filters:
            mag = magnitudes.get(filter_name)
            limit = self.flat_limits.get(filter_name)
            
            # Check if we have a valid magnitude and limit
            if mag is not None and limit is not None and not np.isnan(mag):
                # IMPORTANT: mag < limit means the object is BRIGHTER than the limit
                # (e.g., mag=20.5 is brighter than limit=21.0)
                if mag < limit:
                    detection_details[filter_name] = 1
                else:
                    detection_details[filter_name] = 0
            else:
                # If no magnitude for this filter, mark as no detection
                detection_details[filter_name] = 0

        # Detected if ANY filter has detection
        detected = 1 if any(detection_details.values()) else 0
        return detected, detection_details
    
    def compute_detection_probability(self, magnitudes: Dict, uncertainties: Optional[Dict] = None) -> Tuple[float, Dict]:
        """
        Probability of detection per filter from the limiting-magnitude CDF.

        The limiting magnitude is treated as the random variable:
            L_f ~ Normal(limit_f, limit_error_f)

        Detection in filter f means L_f >= mag_f, so:
            P_detect_f = 1 - CDF_L(mag_f)
                       = CDF((limit_f - mag_f) / limit_error_f)

        The event-level active-search utility is max(P_detect_f), because
        detection in any one requested filter is enough.
        """
        filter_probs = {}
        
        # If no magnitudes, probability is 0
        if not magnitudes:
            return 0.0, {}
        
        for filter_name in self.detection_filters:
            mag = magnitudes.get(filter_name)
            limit = self.flat_limits.get(filter_name)
            
            if mag is not None and limit is not None and not np.isnan(mag):
                sigma = self.filter_errors.get(filter_name, 0.1)
                
                # Positive z_score means the source is brighter than the limit.
                z_score = (limit - mag) / sigma
                filter_probs[filter_name] = norm.cdf(z_score)
            else:
                filter_probs[filter_name] = 0.0

        if filter_probs:
            detection_prob = max(filter_probs.values())
        else:
            detection_prob = 0.0

        return detection_prob, filter_probs
    
    def compute_detection_margin(self, magnitudes: Dict, uncertainties: Dict) -> float:
        """Best-filter detection margin in standard deviations"""
        z_scores = []
        for filter_name in self.detection_filters:
            mag = magnitudes.get(filter_name)
            limit = self.flat_limits.get(filter_name)
            if mag is not None and limit is not None and not np.isnan(mag):
                # Use provided uncertainty, or use the filter error if available
                if filter_name in uncertainties and uncertainties[filter_name] is not None:
                    sigma = uncertainties[filter_name]
                elif filter_name in self.filter_errors:
                    sigma = self.filter_errors[filter_name]
                else:
                    sigma = 0.1
                z_scores.append((limit - mag) / sigma)
        return float(np.max(z_scores)) if z_scores else -np.inf
    
    def compute_detection_margin_batch(self, mags_batch: Dict, uncs_batch: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized version of compute_detection_margin"""
        z_cols, std_cols = [], []
        for filter_name in self.detection_filters:
            if filter_name in mags_batch and filter_name in self.flat_limits:
                # Use provided uncertainty, if available
                if filter_name in uncs_batch and uncs_batch[filter_name] is not None:
                    sigma = uncs_batch[filter_name]
                elif filter_name in self.filter_errors:
                    sigma = np.ones_like(mags_batch[filter_name]) * self.filter_errors[filter_name]
                else:
                    sigma = np.ones_like(mags_batch[filter_name]) * 0.1
                z_cols.append((self.flat_limits[filter_name] - mags_batch[filter_name]) / sigma)
                std_cols.append(sigma)
        if not z_cols:
            n = len(next(iter(mags_batch.values()))) if mags_batch else 0
            return np.full(n, -np.inf), np.full(n, 0.1)
        z_matrix = np.vstack(z_cols)
        std_matrix = np.vstack(std_cols)
        return z_matrix.max(axis=0), std_matrix.mean(axis=0)
