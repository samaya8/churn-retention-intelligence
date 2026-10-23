"""Pytest configuration for Bank Customer Churn Intelligence."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import numpy.dtypes

# Ensure numpy.dtypes is fully populated even when reloaded under pytest-cov
_dt_mod = sys.modules.get("numpy.dtypes", numpy.dtypes)
_type_samples = [
    bool, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64,
    np.complex64, np.complex128,
    object, str, bytes, "datetime64[s]", "timedelta64[s]", "void",
]
for _t in _type_samples:
    try:
        _dt_cls = type(np.dtype(_t))
        if not hasattr(_dt_mod, _dt_cls.__name__):
            setattr(_dt_mod, _dt_cls.__name__, _dt_cls)
        if hasattr(numpy, "dtypes") and not hasattr(numpy.dtypes, _dt_cls.__name__):
            setattr(numpy.dtypes, _dt_cls.__name__, _dt_cls)
    except Exception:
        pass
