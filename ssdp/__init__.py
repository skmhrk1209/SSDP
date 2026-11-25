import numpy as np
import torch

torch.serialization.add_safe_globals(
    [
        np.dtype,
        np.dtypes.Float64DType,
        np.core.multiarray.scalar,
    ]
)
