

from typing import Protocol

from state import PhaseVector


class Engine(Protocol):
    """An engine evolves a phase space state according to a vector.
    """
    def __call__(state: PhaseVector, vector: PhaseVector, dt: float) -> PhaseVector:
        ...



def euler_engine(state: PhaseVector, vector: PhaseVector, dt: float) -> PhaseVector:
    return (state + vector * dt).detach()
