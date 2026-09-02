"""EM-bright source-classification helpers."""

import math


_WARNED_EMBRIGHT_IMPORT = False


def compute_embright_classification(mass1, mass2, spin1z, spin2z, snr):
    """
    Compute HasNS, HasRemnant, and HasMassGap using ligo.em_bright.

    Returns probabilities from ``em_bright.source_classification`` when the
    package is available. If it is unavailable or inputs are invalid, the
    values are left as NaN and the caller can decide whether to use a fallback.
    """
    global _WARNED_EMBRIGHT_IMPORT

    values = [mass1, mass2, spin1z, spin2z, snr]
    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError):
        return {
            "HasNS": math.nan,
            "HasRemnant": math.nan,
            "HasMassGap": math.nan,
            "embright_status": "invalid_input",
        }

    if not all(math.isfinite(v) for v in values):
        return {
            "HasNS": math.nan,
            "HasRemnant": math.nan,
            "HasMassGap": math.nan,
            "embright_status": "invalid_input",
        }

    try:
        from ligo.em_bright import em_bright
    except Exception:
        if not _WARNED_EMBRIGHT_IMPORT:
            print("  Warning: ligo.em_bright is unavailable; EM-bright columns will be NaN.")
            _WARNED_EMBRIGHT_IMPORT = True
        return {
            "HasNS": math.nan,
            "HasRemnant": math.nan,
            "HasMassGap": math.nan,
            "embright_status": "ligo_em_bright_unavailable",
        }

    try:
        has_ns, has_remnant, has_mass_gap = em_bright.source_classification(*values)
    except Exception as exc:
        return {
            "HasNS": math.nan,
            "HasRemnant": math.nan,
            "HasMassGap": math.nan,
            "embright_status": f"em_bright_failed:{type(exc).__name__}",
        }

    return {
        "HasNS": float(has_ns),
        "HasRemnant": float(has_remnant),
        "HasMassGap": float(has_mass_gap),
        "embright_status": "computed",
    }
