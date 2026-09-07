from __future__ import annotations
import math

UNIT_TO_MOLAR = {
    'M': 1.0, 'mM': 1e-3, 'uM': 1e-6, 'µM': 1e-6, 'μM': 1e-6,
    'nM': 1e-9, 'pM': 1e-12, 'fM': 1e-15,
}


def to_molar(value, units):
    try:
        x = float(value)
    except Exception:
        return None
    if not math.isfinite(x) or x <= 0:
        return None
    factor = UNIT_TO_MOLAR.get(str(units).strip())
    return x * factor if factor else None


def pactivity_from_molar(value_molar):
    if value_molar is None or value_molar <= 0:
        return None
    return -math.log10(value_molar)
