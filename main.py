import math
import time

import torch

from engine import euler_engine
from field import damped_field, hamiltonian_field
from state import PhaseVector
from simulator import Simulator
from visualizer import PlotVisualizer


def main() -> None:
    torch.set_default_dtype(torch.float64)  # before any tensors are made, so the state and embeddings pick it up

    def harmonic(state: PhaseVector) -> torch.Tensor:
        omega = 2.0 * torch.pi
        return (0.5 * omega * (state.q**2 + state.p**2)).sum()

    def quartic(state: PhaseVector) -> torch.Tensor:
        omega = 2.0 * torch.pi
        return (0.5 * omega * state.p**2 + 0.25 * state.q**4).sum()

    def double_pendulum(state: PhaseVector) -> torch.Tensor:
        t1, t2 = state.q[0], state.q[1]
        p1, p2 = state.p[0], state.p[1]
        d = t1 - t2
        T = (p1**2 + 2 * p2**2 - 2 * p1 * p2 * torch.cos(d)) / (2 * (1 + torch.sin(d)**2))
        V = -2 * torch.cos(t1) - torch.cos(t2)
        return T + V

    def double_pendulum_embed(q: torch.Tensor) -> torch.Tensor:
        t1, t2 = q[0], q[1]
        pivot = torch.zeros(2)
        bob1 = pivot + torch.stack([torch.sin(t1), -torch.cos(t1)])
        bob2 = bob1 + torch.stack([torch.sin(t2), -torch.cos(t2)])
        return torch.stack([pivot, bob1, bob2])

    initial = PhaseVector(p=torch.tensor([2.0, 0.0]), q=torch.tensor([0.0, 2.0]))
    # harmonic_sim = Simulator(
    #     [hamiltonian_field(harmonic)],
    #     euler_engine,
    #     initial,
    #     dt=0.0001,
    # )
    # quartic_sim = Simulator(
    #     [hamiltonian_field(quartic)],
    #     euler_engine,
    #     initial,
    #     dt=0.0001,
    # )
    pendulum_sim = Simulator(
        [hamiltonian_field(double_pendulum)],
        euler_engine,
        initial,
        dt=0.0001,
        compile_chunk=10,  # ~2-5 s compile on the first frame, then ~2 us/step instead of ~100
    )
    visualizer = PlotVisualizer(embeddings={"pendulum": double_pendulum_embed}, windows={"pendulum": ((-math.pi, math.pi), (-math.pi, math.pi))})
    # pace by the wall clock: each frame simulates however much time has really passed, times `speed`
    speed = 1.0  # simulated seconds per wall second (1.0 = real time)
    max_steps = 2000  # per frame, so a simulation that can't keep up degrades to slow motion instead of spiralling
    last = time.perf_counter()
    while True:
        now = time.perf_counter()
        steps = min(max_steps, round((now - last) * speed / pendulum_sim.dt))
        last = now
        # visualizer.step(harmonic_sim.run(steps), vector=harmonic_sim.vector, name="harmonic")
        state = pendulum_sim.run(steps)
        visualizer.step(state, vector=pendulum_sim.vector, name="pendulum", energy=float(double_pendulum(state)))


if __name__ == "__main__":
    main()
