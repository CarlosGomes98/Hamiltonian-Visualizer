"""Records the README's GIFs by driving the real PlotVisualizer offscreen.

    MPLBACKEND=Agg uv run python docs/record.py
"""
import atexit
import math
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import euler_engine  # noqa: E402
from field import damped_field, hamiltonian_field  # noqa: E402
from simulator import Simulator  # noqa: E402
from state import PhaseVector  # noqa: E402
from visualizer import PlotVisualizer  # noqa: E402

OUT = Path(__file__).resolve().parent
FPS = 20
DT = 1e-4
DPI = 80


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


def record(
    filename: str, seconds: float, series: list[tuple[str, Simulator, callable]], speed: float = 0.4, **visualizer_kwargs
) -> None:
    """Plays `seconds` of simulated time at `speed` (simulated seconds per GIF second) and saves it as a looping GIF."""
    visualizer = PlotVisualizer(**visualizer_kwargs)
    atexit.unregister(visualizer._block_until_closed)
    # several visualizer points per GIF frame, so phase-space curves come out smooth instead of polygonal
    points_per_frame = 5
    steps_per_point = round(speed / (FPS * points_per_frame * DT))
    frames = []
    for _ in range(round(seconds / speed * FPS)):
        for _ in range(points_per_frame):
            for name, sim, H in series:
                state = sim.run(steps_per_point)
                visualizer.step(state, vector=sim.vector, name=name, energy=float(H(state)))
        visualizer.fig.canvas.draw()
        rgba = Image.frombuffer("RGBA", visualizer.fig.canvas.get_width_height(), visualizer.fig.canvas.buffer_rgba())
        frames.append(rgba.convert("RGB").quantize(colors=64, dither=Image.Dither.NONE))
    frames[0].save(OUT / filename, save_all=True, append_images=frames[1:], duration=1000 // FPS, loop=0, optimize=True)
    print(f"{filename}: {len(frames)} frames, {(OUT / filename).stat().st_size / 1e6:.1f} MB")


def sim(fields, p, q) -> Simulator:
    return Simulator(fields, euler_engine, PhaseVector(p=p, q=q), dt=DT, compile_chunk=10)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    import matplotlib
    matplotlib.rcParams["figure.dpi"] = DPI

    record("harmonic.gif", 3, [("harmonic", sim([hamiltonian_field(harmonic)], 2.0, 1.0), harmonic)])
    record("damped.gif", 3, [
        ("undamped", sim([hamiltonian_field(harmonic)], 2.0, 1.0), harmonic),
        ("damped (γ = 1.5)", sim([hamiltonian_field(harmonic), damped_field(gamma=1.5)], 2.0, 1.0), harmonic),
    ])
    record("quartic.gif", 3, [("quartic", sim([hamiltonian_field(quartic)], 2.0, 1.0), quartic)])
    record(
        "double_pendulum.gif", 10,
        [("double pendulum", sim([hamiltonian_field(double_pendulum)], [2.0, 0.0], [0.0, 2.0]), double_pendulum)],
        embeddings={"double pendulum": double_pendulum_embed},
        windows={"double pendulum": ((-math.pi, math.pi), (-math.pi, math.pi))},
        trail_length=40,
        speed=0.6,
    )


if __name__ == "__main__":
    main()
