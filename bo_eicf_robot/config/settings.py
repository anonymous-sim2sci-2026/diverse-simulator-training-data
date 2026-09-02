"""
Configuration settings for BO CDF Simulator
"""

import os
from datetime import datetime


class Config:
    """Configuration for NMMA simulation."""
    
    # Path to SVD models for kilonova lightcurve generation
    SVD_PATH = os.environ.get("NMMA_SVD_PATH", "./svdmodels")
    
    # Prior file for NMMA
    PRIOR_FILE = os.environ.get("NMMA_PRIOR_FILE", "./priors/Bu2019lm.prior")
    
    # Output directory
    OUTPUT_BASE = os.environ.get("BO_OUTPUT_BASE", "./out")
    
    # ZTF Detection limits with errors
    ZTF_LIMITS = {
        'g': {'limit': 21.067, 'error': 0.003},
        'r': {'limit': 21.012, 'error': 0.002},
        'i': {'limit': 20.510, 'error': 0.010},
    }
    
    # Detection thresholds
    DETECTION_THRESHOLD_MAG = 22.0  # ZTF-like limit
    ERROR_THRESHOLD = 1.0  # Magnitude error threshold
    
    # Exposure times to test
    EXPOSURE_TIMES = [180, 300]  # 3min and 5min exposures
    
    # ZTF filters for NMMA
    FILTERS = ['ztfg', 'ztfr', 'ztfi']
    
    # Feature columns for optimization
    FEATURE_COLUMNS = ['mass1', 'mass2', 'spin1z', 'spin2z', 'inclination', 'distance']
    
    # Parameter bounds for optimization
    PBOUNDS = {
        'mass1': (1.0, 4.0),
        'mass2': (1.0, 3.0),
        'spin1z': (0.0, 0.95),
        'spin2z': (0.0, 0.4), 
        'inclination': (0.0, 3.141592653589793),  # np.pi
        'distance': (55.0, 1500.0),
    }
    
    # Detection filters used by the optimizer and detection scoring.
    # NMMA is configured to simulate only ztfg, ztfr, ztfi, so the surrogate
    # objective should score only the corresponding CSV columns: g, i, r.
    DETECTION_FILTERS = ['g', 'i', 'r']
    
    def __init__(self):
        self._verify_paths()
        
    def _verify_paths(self):
        """Verify required paths exist"""
        if not os.path.exists(self.SVD_PATH):
            print(f"⚠️ Warning: SVD_PATH '{self.SVD_PATH}' does not exist")
        else:
            print(f"✅ SVD_PATH found: {self.SVD_PATH}")
            
        if not os.path.exists(self.PRIOR_FILE):
            print(f"⚠️ Warning: PRIOR_FILE '{self.PRIOR_FILE}' does not exist")
        else:
            print(f"✅ PRIOR_FILE found: {self.PRIOR_FILE}")
    
    @property
    def bounds(self):
        """Get bounds as list of tuples"""
        return [self.PBOUNDS[col] for col in self.FEATURE_COLUMNS]
    
    @property
    def feature_columns(self):
        """Get feature columns"""
        return self.FEATURE_COLUMNS
    
    @property
    def detection_filters(self):
        """Get detection filters"""
        return self.DETECTION_FILTERS
    
    @property
    def output_dir(self):
        """Get output directory with short date+run-number naming (e.g. 072126_1)"""
        date_str = datetime.now().strftime('%m%d%y')
        n = 1
        while True:
            candidate = os.path.join(self.OUTPUT_BASE, f"{date_str}_{n}")
            if not os.path.exists(candidate):
                os.makedirs(candidate, exist_ok=True)
                return candidate
            n += 1
    
    def get_ztf_limit(self, filter_name):
        """Get ZTF limit for a specific filter"""
        if filter_name in self.ZTF_LIMITS:
            return self.ZTF_LIMITS[filter_name]['limit']
        return None
    
    def get_ztf_error(self, filter_name):
        """Get ZTF error for a specific filter"""
        if filter_name in self.ZTF_LIMITS:
            return self.ZTF_LIMITS[filter_name]['error']
        return 0.1  # Default if not found
