from dataclasses import dataclass
from typing import Protocol

import torch

from state import PhaseVector


@dataclass
class Field(Protocol):
    def __call__(self, state: PhaseVector) -> PhaseVector:
        ...

class Hamiltonian(Protocol):
    def __call__(self, state: PhaseVector) -> torch.Tensor:
        ...

def hamiltonian_field(H: Hamiltonian) -> Field:
    # torch.func.grad can be traced by torch.compile into a single graph, but runs ~2x slower than autograd.grad
    # when not compiled, so use each where it's best
    dH = torch.func.grad(lambda p, q: H(PhaseVector(p, q)), argnums=(0, 1))
    def field(state: PhaseVector) -> PhaseVector:
        if torch.compiler.is_compiling():
            dH_dp, dH_dq = dH(state.p.detach(), state.q.detach())
        else:
            p = state.p.detach().requires_grad_(True)
            q = state.q.detach().requires_grad_(True)
            dH_dp, dH_dq = torch.autograd.grad(H(PhaseVector(p, q)), (p, q))
        return PhaseVector(-dH_dq, dH_dp)
    return field

def damped_field(gamma: float) -> PhaseVector:
    def field(state: PhaseVector) -> PhaseVector:
        return PhaseVector(-gamma * state.p, 0)
    return field
