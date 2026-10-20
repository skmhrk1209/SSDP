from collections.abc import Callable

import jaxtyping as jt
import torch

type SDFType = Callable[[jt.Float[torch.Tensor, " *B 3 "]], jt.Float[torch.Tensor, " *B 1 "]]
