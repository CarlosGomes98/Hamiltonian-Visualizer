from dataclasses import dataclass
from typing import Literal, overload

import torch
from torch import Tensor

Scalar = float | int | bool


def _as_1d_tensor(value: Tensor | Scalar) -> Tensor:
    with torch.no_grad():
        tensor = torch.as_tensor(value)
        return tensor.unsqueeze(0) if tensor.ndim == 0 else tensor
        
@dataclass
class PhaseVector:
    p: Tensor
    q: Tensor

    def __init__(self, p: Tensor | Scalar, q: Tensor | Scalar) -> None:
        self.p = _as_1d_tensor(p)
        self.q = _as_1d_tensor(q)

    def detach(self) -> 'PhaseVector':
        return PhaseVector(self.p.detach(), self.q.detach())

    def __str__(self) -> str:
        return f"PhaseVector(p={self.p}, q={self.q})"
    
    @overload
    def __add__(self, other: Literal[0]) -> 'PhaseVector': ...
    @overload
    def __add__(self, other: 'PhaseVector') -> 'PhaseVector': ...
    def __add__(self, other: 'PhaseVector' | Literal[0]) -> 'PhaseVector':
        if other == 0:
            return self
        return PhaseVector(self.p + other.p, self.q + other.q)

    def __radd__(self, other: Literal[0]) -> 'PhaseVector':
        # so sum(...) works: it starts from 0, i.e. 0 + PhaseVector
        return self
    
    def __mul__(self, other: Scalar) -> 'PhaseVector':
        return PhaseVector(self.p * other, self.q * other)
