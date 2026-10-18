import atexit
import math
from typing import Callable, Optional, Protocol

import matplotlib.pyplot as plt
import torch
from matplotlib.lines import Line2D
from matplotlib.text import Annotation, Text
from torch import Tensor

from state import PhaseVector


class Visualizer(Protocol):
    def step(self, state: PhaseVector, vector: Optional[PhaseVector] = None, name: str = "trajectory") -> None:
        ...


class PrintVisualizer:
    def step(self, state: PhaseVector, vector: Optional[PhaseVector] = None, name: str = "trajectory") -> None:
        print(f"{name}: {state}")


Embedding = Callable[[Tensor], Tensor]
"""Maps q to the points of the physical system in the plane, as a (n_points, 2) tensor.

With more than one point, the first is a fixed anchor (e.g. a pendulum pivot), the rest are bodies,
and consecutive points are joined by rods. A single point is drawn as a lone body.
"""


Window = Optional[tuple[float, float]]
"""The (low, high) range a periodic coordinate is wrapped into, e.g. (-pi, pi) for an angle; None means no wrapping."""


class PlotVisualizer:
    def __init__(
        self,
        embeddings: Optional[dict[str, Embedding]] = None,
        windows: Optional[dict[str, tuple[Window, ...]]] = None,
        trail_length: int = 15,
    ) -> None:
        self.embeddings: dict[str, Embedding] = embeddings or {}
        self.windows: dict[str, tuple[Window, ...]] = windows or {}  # one window per coordinate of q
        self.trail_length = trail_length
        self.q: dict[str, list[float]] = {}
        self.p: dict[str, list[float]] = {}
        # self.t: dict[str, list[int]] = {}
        self.phase_trajectories: dict[str, Line2D] = {}
        self.phase_markers: dict[str, Line2D] = {}
        self.phase_arrows: dict[str, Annotation] = {}
        self.phase_turns: dict[str, int] = {}  # which copy of the window q[0] was last in
        self.energy_texts: dict[str, Text] = {}
        self.initial_energies: dict[str, float] = {}
        # self.q_trajectories: dict[str, Line2D] = {}
        # self.q_markers: dict[str, Line2D] = {}
        self.config_x: dict[str, list[float]] = {}
        self.config_y: dict[str, list[float]] = {}
        self.config_embeddings: dict[str, Embedding] = {}  # resolved per series: user-given or default
        self.config_rods: dict[str, Line2D] = {}
        self.config_anchors: dict[str, Line2D] = {}
        self.config_particles: dict[str, Line2D] = {}
        self.config_trails: dict[str, Line2D] = {}
        self.config_arrows: dict[str, list[Annotation]] = {}
        self.config_lanes: dict[str, int] = {}  # 1-d series get a dummy constant y, one lane each
        self.config_is_2d = False
        self.config_extent = 1.0
        self.config_max_speed = 0.0
        # position over time is disabled for now; restore ax_q here and uncomment its lines to bring it back
        self.fig, (self.ax_phase, self.ax_config) = plt.subplots(1, 2, figsize=(10, 4.5))

        self.ax_phase.set_xlabel("q")
        self.ax_phase.set_ylabel("p")
        self.ax_phase.set_title("Phase space")
        self.ax_phase.set_aspect("equal", adjustable="datalim")
        self.ax_phase.axhline(0.0, color="0.8", lw=0.8)
        self.ax_phase.axvline(0.0, color="0.8", lw=0.8)
        self.ax_phase.grid(True, alpha=0.3)

        # self.ax_q.set_xlabel("t")
        # self.ax_q.set_ylabel("q")
        # self.ax_q.set_title("Position over time")
        # self.ax_q.axhline(0.0, color="0.8", lw=0.8)
        # self.ax_q.grid(True, alpha=0.3)

        self.ax_config.set_xlabel("q[0]")
        self.ax_config.set_title("Configuration space")
        self.ax_config.axvline(0.0, color="0.6", lw=0.8, ls="--")
        self.ax_config.grid(True, alpha=0.3)
        self.ax_config.set_yticks([])

        self.fig.tight_layout()
        plt.ion()
        plt.show()
        atexit.register(self._block_until_closed)

    def _ensure_series(self, name: str) -> None:
        if name in self.phase_trajectories:
            return
        (phase_trajectory,) = self.ax_phase.plot([], [], "-", lw=1.5, label=name)
        color = phase_trajectory.get_color()
        (phase_marker,) = self.ax_phase.plot([], [], "o", ms=6, color=color)
        # (q_trajectory,) = self.ax_q.plot([], [], "-", lw=1.5, color=color, label=name)
        # (q_marker,) = self.ax_q.plot([], [], "o", ms=6, color=color)
        self.phase_trajectories[name] = phase_trajectory
        self.phase_markers[name] = phase_marker
        # self.q_trajectories[name] = q_trajectory
        # self.q_markers[name] = q_marker
        (config_trail,) = self.ax_config.plot([], [], "-", lw=4, alpha=0.25, color=color, solid_capstyle="round")
        (config_rod,) = self.ax_config.plot([], [], "-", lw=2, color="0.3")
        (config_anchor,) = self.ax_config.plot([], [], "s", ms=6, color="0.3")
        (config_particle,) = self.ax_config.plot([], [], "o", ms=16, color=color, mec="black", mew=0.8)
        self.config_trails[name] = config_trail
        self.config_rods[name] = config_rod
        self.config_anchors[name] = config_anchor
        self.config_particles[name] = config_particle
        self.config_arrows[name] = []
        self.config_x[name] = []
        self.config_y[name] = []
        self.q[name] = []
        self.p[name] = []
        # self.t[name] = []
        self.ax_phase.legend(loc="upper right")
        # self.ax_q.legend(loc="upper right")

    def step(
        self,
        state: PhaseVector,
        vector: Optional[PhaseVector] = None,
        name: str = "trajectory",
        energy: Optional[float] = None,
    ) -> None:
        self._ensure_series(name)
        if energy is not None:
            self._draw_energy(name, energy)
        q_all, turns = self._wrap(name, state.q.detach().reshape(-1))
        q = float(q_all[0]) # projection into that dim
        p = float(state.p.detach().reshape(-1)[0])
        
        # crossing a window edge would draw a line across the plot, so break the line there with a NaN
        turn = int(turns[0])
        if self.phase_turns.get(name, turn) != turn:
            self.q[name].append(math.nan)
            self.p[name].append(math.nan)
        self.phase_turns[name] = turn
        self.q[name].append(q)
        self.p[name].append(p)
        # self.t[name].append(len(self.t[name]))
        # t = self.t[name][-1]
        self.phase_trajectories[name].set_data(self.q[name], self.p[name])
        self.phase_markers[name].set_data([q], [p])
        # self.q_trajectories[name].set_data(self.t[name], self.q[name])
        # self.q_markers[name].set_data([t], [q])
        self.ax_phase.relim()
        self.ax_phase.autoscale_view()
        if vector is not None:
            q_vector = float(vector.q.detach().reshape(-1)[0])
            p_vector = float(vector.p.detach().reshape(-1)[0])
            self._draw_arrow(name, q, p, q_vector, p_vector)
        # wrapping is only for the phase plot; the embedding sees the real q
        self._draw_particle(name, state.q.detach().reshape(-1), None if vector is None else vector.q.detach().reshape(-1))
        # self.ax_q.relim()
        # self.ax_q.autoscale_view()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def _draw_energy(self, name: str, energy: float) -> None:
        # a conservative system should keep H constant, so the drift measures the integrator's error
        initial = self.initial_energies.setdefault(name, energy)
        drift = (energy - initial) / abs(initial) if initial != 0.0 else energy - initial
        if name not in self.energy_texts:
            self.energy_texts[name] = self.ax_phase.text(
                0.02, 0.97 - 0.06 * len(self.energy_texts), "", transform=self.ax_phase.transAxes,
                va="top", fontsize=9, family="monospace", color=self.phase_trajectories[name].get_color(),
            )
        self.energy_texts[name].set_text(f"H = {energy:+.5f}  drift {drift:+.1e}")

    def _wrap(self, name: str, q: Tensor) -> tuple[Tensor, Tensor]:
        """q wrapped into each coordinate's window, and how many periods each coordinate was shifted by."""
        wrapped = q.clone()
        turns = torch.zeros_like(q, dtype=torch.long)
        for i, window in enumerate(self.windows.get(name, ())):
            if window is None:
                continue
            low, high = window
            # the period is high - low, and the cut sits at the window's edges
            turns[i] = int(math.floor((float(q[i]) - low) / (high - low)))
            wrapped[i] = q[i] - turns[i] * (high - low)
        return wrapped, turns

    def _draw_arrow(self, name: str, q: float, p: float, q_vector: float, p_vector: float) -> None:
        previous = self.phase_arrows.pop(name, None)
        if previous is not None:
            previous.remove()
        magnitude = math.hypot(q_vector, p_vector)
        if magnitude == 0.0:
            return
        # the field has units of 1/time, so scale it to a fixed fraction of the visible axes
        x_min, x_max = self.ax_phase.get_xlim()
        y_min, y_max = self.ax_phase.get_ylim()
        scale = 0.15 * max(x_max - x_min, y_max - y_min) / magnitude
        self.phase_arrows[name] = self.ax_phase.annotate(
            "",
            xytext=(q, p),
            xy=(q + scale * q_vector, p + scale * p_vector),
            arrowprops=dict(arrowstyle="->", color=self.phase_trajectories[name].get_color()),
        )

    def _config_lane(self, name: str) -> int:
        if name not in self.config_lanes:
            lane = len(self.config_lanes)
            self.config_lanes[name] = lane
            color = self.phase_trajectories[name].get_color()
            self.ax_config.axhline(lane, color="0.85", lw=1.0, zorder=0)
            self.ax_config.text(0.02, lane - 0.3, name, color=color, transform=self.ax_config.get_yaxis_transform())
        return self.config_lanes[name]

    def _enable_config_2d(self, xlabel: str, ylabel: str) -> None:
        if self.config_is_2d:
            return
        self.config_is_2d = True
        self.ax_config.set_xlabel(xlabel)
        self.ax_config.set_ylabel(ylabel)
        self.ax_config.yaxis.set_major_locator(plt.AutoLocator())
        self.ax_config.axhline(0.0, color="0.6", lw=0.8, ls="--")
        self.ax_config.set_aspect("equal", adjustable="box")

    def _default_embedding(self, name: str, q: Tensor) -> Embedding:
        # project onto the first two dims of q (the identity in 2-d); a 1-d q gets a dummy constant second dim (its lane)
        if q.numel() >= 2:
            self._enable_config_2d("q[0]", "q[1]")
            return lambda q: q[:2].reshape(1, 2)
        lane = float(self._config_lane(name))
        return lambda q: torch.stack([q[0], torch.full_like(q[0], lane)]).reshape(1, 2)

    def _config_points(self, name: str, q: Tensor, velocity: Optional[Tensor]) -> tuple[Tensor, Optional[Tensor]]:
        """Points of the system in the plane, (n_points, 2), and their velocities if known."""
        embed = self.config_embeddings.get(name)
        if embed is None:
            embed = self.embeddings.get(name)
            if embed is not None:
                self._enable_config_2d("x", "y")
            else:
                embed = self._default_embedding(name, q)
            self.config_embeddings[name] = embed
        if velocity is None:
            return embed(q), None
        # chain rule: each point moves at d(embed)/dq . dq/dt
        return torch.func.jvp(embed, (q,), (velocity,))

    def _draw_particle(self, name: str, q: Tensor, velocity: Optional[Tensor]) -> None:
        points, velocities = self._config_points(name, q, velocity)
        points = points.detach()
        has_anchor = len(points) > 1
        bodies = points[1:] if has_anchor else points
        x, y = float(points[-1, 0]), float(points[-1, 1])
        self.config_x[name].append(x)
        self.config_y[name].append(y)

        # keep the range fixed (only ever growing) so the motion isn't hidden by rescaling
        on_lane = name in self.config_lanes
        reach = points[:, 0].abs().max() if on_lane else points.abs().max()
        self.config_extent = max(self.config_extent, 1.1 * float(reach))
        self.ax_config.set_xlim(-self.config_extent, self.config_extent)
        y_min, y_max = math.inf, -math.inf
        if self.config_is_2d:
            y_min, y_max = -self.config_extent, self.config_extent
        if self.config_lanes:
            y_min, y_max = min(y_min, -0.75), max(y_max, len(self.config_lanes) - 0.25)
        self.ax_config.set_ylim(y_min, y_max)

        if has_anchor:
            self.config_rods[name].set_data(points[:, 0].tolist(), points[:, 1].tolist())
            self.config_anchors[name].set_data([float(points[0, 0])], [float(points[0, 1])])
        self.config_particles[name].set_data(bodies[:, 0].tolist(), bodies[:, 1].tolist())
        # trail follows the last body, e.g. the tip of a pendulum
        trail = slice(-self.trail_length, None)
        self.config_trails[name].set_data(self.config_x[name][trail], self.config_y[name][trail])

        for arrow in self.config_arrows[name]:
            arrow.remove()
        self.config_arrows[name] = []
        if velocities is None:
            return
        body_velocities = (velocities[1:] if has_anchor else velocities).detach()
        speeds = body_velocities.norm(dim=1)
        if float(speeds.max()) == 0.0:
            return
        # dq/dt has units of length/time, so scale it relative to the fastest speed seen so far
        self.config_max_speed = max(self.config_max_speed, float(speeds.max()))
        scale = 0.3 * self.config_extent / self.config_max_speed
        color = self.phase_trajectories[name].get_color()
        for (bx, by), (vx, vy) in zip(bodies.tolist(), body_velocities.tolist()):
            self.config_arrows[name].append(self.ax_config.annotate(
                "",
                xytext=(bx, by),
                xy=(bx + scale * vx, by + scale * vy),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            ))

    def _block_until_closed(self) -> None:
        plt.ioff()
        plt.show()
