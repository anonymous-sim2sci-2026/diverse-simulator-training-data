"""Definitions and preparation helpers for the gray-box h(x) output vector."""

import numpy as np


H_OUTPUT_COLUMNS = [
    "snr",
    "area90",
    "log10_mej_dyn",
    "log10_mej_wind",
    "g",
    "i",
    "r",
    "HasNS",
    "HasMassGap",
    "HasRemnant",
    "distance",
]

H_PLOT_COLUMNS = [
    "snr",
    "area90",
    "log10_mej_dyn",
    "log10_mej_wind",
    "g",
    "r",
    "i",
    "HasNS",
    "HasRemnant",
    "HasMassGap",
    "distance",
]

H_COLUMN_ALIASES = {
    "snr": ["snr", "SNR"],
    "area90": ["area90", "area(90)", "Area90", "area_90", "localization_area90", "localization_area_90"],
    "log10_mej_dyn": ["log10_mej_dyn", "log_mej_dyn", "log10_dynamical_ejecta_mass"],
    "log10_mej_wind": ["log10_mej_wind", "log_mej_wind", "log10_wind_ejecta_mass"],
    "mej_dyn": ["mej_dyn", "dynamical_ejecta_mass", "dyn_ejecta_mass", "ejecta_mass_dyn", "mej_dynamical"],
    "mej_wind": ["mej_wind", "wind_ejecta_mass", "ejecta_mass_wind", "mej_wind_ejecta"],
    "HasNS": ["HasNS", "HasnS", "has_ns"],
    "HasMassGap": ["HasMassGap", "has_mass_gap"],
    "HasRemnant": ["HasRemnant", "HasRemnan", "has_remnant"],
}

MAGNITUDE_COLUMNS = ["g", "i", "r"]
H_PLOT_MAGNITUDE_COLUMNS = ["g", "i", "r", "z"]


def prepare_h_columns(df):
    """
    Add/standardize the 11 h(x) columns used by the gray-box models.

    The current catalog does not contain ejecta masses directly, so we derive
    them from the same simple HasRemnant mapping used by the NMMA candidate
    setup when explicit ejecta columns are absent.
    """
    df = df.copy()

    for target, aliases in H_COLUMN_ALIASES.items():
        if target in df.columns:
            continue
        for source in aliases:
            if source in df.columns:
                df[target] = df[source]
                break

    if "log10_mej_dyn" not in df.columns:
        if "mej_dyn" in df.columns:
            vals = df["mej_dyn"].astype(float)
            df["log10_mej_dyn"] = np.log10(np.clip(vals, 1e-12, None))
        elif "HasRemnant" in df.columns:
            df["log10_mej_dyn"] = -2.0 + 0.5 * df["HasRemnant"].astype(float)
        else:
            df["log10_mej_dyn"] = np.nan

    if "log10_mej_wind" not in df.columns:
        if "mej_wind" in df.columns:
            vals = df["mej_wind"].astype(float)
            df["log10_mej_wind"] = np.log10(np.clip(vals, 1e-12, None))
        elif "HasRemnant" in df.columns:
            df["log10_mej_wind"] = -1.5 + 0.5 * df["HasRemnant"].astype(float)
        else:
            df["log10_mej_wind"] = np.nan

    if "mej_dyn" not in df.columns and "log10_mej_dyn" in df.columns:
        df["mej_dyn"] = np.power(10.0, df["log10_mej_dyn"].astype(float))

    if "mej_wind" not in df.columns and "log10_mej_wind" in df.columns:
        df["mej_wind"] = np.power(10.0, df["log10_mej_wind"].astype(float))

    return df


def available_h_columns(columns):
    """Return h columns available after prepare_h_columns has been applied."""
    column_set = set(columns)
    return [col for col in H_OUTPUT_COLUMNS if col in column_set]
