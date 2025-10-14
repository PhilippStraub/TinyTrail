from typing import Tuple, Any


def get_qparams(details: Any) -> Tuple[float, int]:
    """Extract quantization (scale, zero_point) from TFLite tensor details.

    Supports both 'quantization' and 'quantization_parameters' formats and
    returns sensible defaults (1.0, 0) when not present.
    """
    if details is None:
        return 1.0, 0

    # Try direct 'quantization'
    q = details.get('quantization', None)
    if q and isinstance(q, (list, tuple)) and len(q) >= 2:
        try:
            scale = float(q[0] or 1.0)
            zero = int(q[1] or 0)
            return scale, zero
        except Exception:
            pass

    # Fallback 'quantization_parameters'
    qp = details.get('quantization_parameters', {}) or {}
    scales = qp.get('scales', [])
    zeros = qp.get('zero_points', [])
    try:
        scale = float(scales[0]) if scales else 1.0
    except Exception:
        scale = 1.0
    try:
        zero = int(zeros[0]) if zeros else 0
    except Exception:
        zero = 0

    return scale, zero

