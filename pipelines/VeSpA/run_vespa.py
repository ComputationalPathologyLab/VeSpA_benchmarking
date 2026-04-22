from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


_MODULE_PATH = Path(__file__).with_name("Segmentation + Measurements.py")
_SPEC = spec_from_file_location("vespa_segmentation", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load VeSpA module from {_MODULE_PATH}")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def run_model(image_path: str) -> np.ndarray:
    """Return the VeSpA binary segmentation mask for a single image."""
    return _MODULE.run_model(image_path)
