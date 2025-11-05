import torch
from typing import Tuple

def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)

# Type definition
KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]