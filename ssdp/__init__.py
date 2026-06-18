import numpy as np
import torch

from . import engines, fields, models, renderers, utils

torch.serialization.add_safe_globals(
    [
        np.dtype,
        np.dtypes.Float64DType,
        np._core.multiarray.scalar,
    ]
)
