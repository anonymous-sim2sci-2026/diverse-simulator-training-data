"""
Data loading and preparation functions
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

from utils.h_columns import H_OUTPUT_COLUMNS, prepare_h_columns


def load_and_prepare_data(base_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the O5a, O5b, O5c datasets and combine them.
    
    Args:
        base_path: Path to data directory
        
    Returns:
        Combined DataFrame
    """
    if base_path is None:
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    files = {
        'O5a': [
            f'{base_path}/ZTF_BNS_COMBINED_O5a_WITH_PEAK_MAGS.csv',
        ],
        'O5b': [
            f'{base_path}/ZTF_BNS_COMBINED_O5b_WITH_PEAK_MAGS.csv',
        ],
        'O5c': [
            f'{base_path}/ZTF_BNS_COMBINED_O5c_WITH_PEAK_MAGS.csv',
        ],
    }

    dfs = []
    for name, candidate_paths in files.items():
        path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if path is None:
            print(f"Warning: {name} file not found. Tried:")
            for candidate_path in candidate_paths:
                print(f"  - {candidate_path}")
            continue

        df = pd.read_csv(path)
        df['dataset'] = name
        dfs.append(df)
        print(f"Loaded {name}: {len(df)} events from {path}")

    if not dfs:
        raise ValueError("No datasets loaded!")

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = prepare_h_columns(combined_df)
    print(f"\nTotal combined dataset: {len(combined_df)} events")
    return combined_df

def prepare_training_data(
    df: pd.DataFrame, 
    feature_columns: list, 
    detection_filters: list,
    flat_limits: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    df = prepare_h_columns(df)

    numeric_columns = list(dict.fromkeys(
        feature_columns
        + [col for col in detection_filters if col in df.columns]
        + [col for col in H_OUTPUT_COLUMNS if col in df.columns]
    ))
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    finite_mask = np.isfinite(df[numeric_columns].to_numpy(dtype=float)).all(axis=1)
    if not finite_mask.all():
        dropped = int((~finite_mask).sum())
        print(f"  Dropping {dropped} rows with missing/non-finite required x or h values")
        if "dataset" in df.columns:
            print(df.loc[~finite_mask, "dataset"].value_counts().to_string())
        df = df.loc[finite_mask].reset_index(drop=True)

    X_train = df[feature_columns].to_numpy(dtype=float)
    y_train_mags = {}
    for column_name in H_OUTPUT_COLUMNS:
        if column_name in df.columns:
            y_train_mags[column_name] = df[column_name].astype(float)
        else:
            print(f"Warning: h-column {column_name} not found in dataset")
            y_train_mags[column_name] = pd.Series([np.nan] * len(df))

    from utils.detection import DetectionCalculator
    
    det_calc = DetectionCalculator(
        detection_filters=detection_filters,
        flat_limits=flat_limits,   # <-- now actually populated
    )
    y_train_detection = []
    for i in range(len(df)):
        mags_i = {
            f: df[f].iloc[i]
            for f in detection_filters
            if f in df.columns and not np.isnan(df[f].iloc[i])
        }
        prob_i, _ = det_calc.compute_detection_probability(mags_i)
        y_train_detection.append(float(np.random.random() < prob_i))
    y_train_detection = np.array(y_train_detection)

    return X_train, y_train_mags, y_train_detection
