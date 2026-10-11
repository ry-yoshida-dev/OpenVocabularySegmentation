"""
Shared NumPy array type aliases for segmentation inputs and outputs.
"""

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type Float32Array = NDArray[np.float32]
type IntArray = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]
