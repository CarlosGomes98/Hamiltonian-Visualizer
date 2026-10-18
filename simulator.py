from typing import Optional

import torch
from torch import Tensor

from engine import Engine
from field import Field
from state import PhaseVector

Scalar = float | int | bool

class Simulator:
    """A simulator evolves a phase space state according to a list of vector fields.

    With `compile_chunk` set, `run` advances through torch.compile'd chunks of that many steps. The math is
    unchanged; compiling just removes the per-step Python and dispatch overhead, which dominates for small states.
    The first `run` pays a one-off compile of a few seconds.
    """
    def __init__(
        self,
        fields: list[Field],
        engine: Engine,
        initial_state: PhaseVector,
        dt: float,
        compile_chunk: Optional[int] = None,
    ):
        self.fields: list[Field] = fields
        self.engine: Engine = engine
        self.state: PhaseVector = initial_state
        self.dt: float = dt
        self.compile_chunk = compile_chunk
        if compile_chunk is not None:
            # compiled on plain tensors so each chunk is one graph; the remainder of a run goes through `_one`
            self._chunk = torch.compile(lambda p, q: self._advance(p, q, compile_chunk), fullgraph=True)
            self._one = torch.compile(lambda p, q: self._advance(p, q, 1), fullgraph=True)

    @property
    def vector(self) -> PhaseVector:
        return sum(field(self.state) for field in self.fields)

    def _advance(self, p: Tensor, q: Tensor, steps: int) -> tuple[Tensor, Tensor]:
        state = PhaseVector(p, q)
        for _ in range(steps):
            vector = sum(field(state) for field in self.fields)
            state = self.engine(state, vector, self.dt)
        return state.p, state.q

    def step(self) -> PhaseVector:
        self.state = self.engine(self.state, self.vector, self.dt)
        return self.state

    def run(self, steps: int) -> PhaseVector:
        if self.compile_chunk is None:
            for _ in range(steps):
                self.step()
            return self.state
        p, q = self.state.p, self.state.q
        for _ in range(steps // self.compile_chunk):
            p, q = self._chunk(p, q)
        for _ in range(steps % self.compile_chunk):
            p, q = self._one(p, q)
        self.state = PhaseVector(p, q)
        return self.state
