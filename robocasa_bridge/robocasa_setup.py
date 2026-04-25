"""Robocasa environment factory for the MuJoCo bridge.

Builds a robosuite/robocasa env with a Franka-class robot and exposes:

* the underlying ``mujoco.MjModel`` / ``mujoco.MjData`` so the bridge can
  read raw qpos / qvel / ctrl,
* ``action_low`` / ``action_high`` from ``env.action_spec``,
* ``action_layout`` — the composite controller's per-part decomposition
  (which action indices correspond to arm OSC vs gripper vs mobile base),
* ``initial_obs`` — the full robosuite observation dict captured from the
  reset that ``robosuite.make`` runs internally, so the bridge has a
  populated ``last_obs`` before the first ``env.step``.

Install (one-time): see README ("Robocasa kitchen flow").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np


DEFAULT_ENV = "PickPlaceCounterToCabinet"
DEFAULT_ROBOT = "PandaOmron"
DEFAULT_CONTROL_FREQ = 20


@dataclass
class RobocasaEnv:
    """Holds a robosuite env plus quick handles to its raw MuJoCo state."""

    env: Any
    model: mujoco.MjModel
    data: mujoco.MjData
    action_dim: int
    control_freq: int
    action_low: np.ndarray
    action_high: np.ndarray
    action_layout: list[dict[str, Any]]
    initial_obs: dict[str, Any]
    env_name: str
    robot_name: str

    def reset(self) -> dict[str, Any]:
        return self.env.reset()

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        return self.env.step(action)

    def render(self) -> None:
        self.env.render()


def _probe_action_layout(env: Any) -> list[dict[str, Any]]:
    """Best-effort introspection of the composite controller's parts.

    Returns a list of ``{"part": name, "indices": [start, stop],
    "controller": classname, "dim": int}`` covering the full action
    vector. Falls back to a single "composite" entry if robosuite's
    internal attributes don't match what we expect on this version.
    """
    try:
        robot = env.robots[0]
    except Exception:
        return []

    cc = (
        getattr(robot, "composite_controller", None)
        or getattr(robot, "controller", None)
    )
    if cc is None:
        return []

    parts = (
        getattr(cc, "part_controllers", None)
        or getattr(cc, "controllers", None)
    )
    if not parts:
        return [{
            "part": "composite",
            "indices": [0, int(env.action_dim)],
            "controller": type(cc).__name__,
            "dim": int(env.action_dim),
        }]

    layout: list[dict[str, Any]] = []
    idx = 0
    for name, part_ctrl in parts.items():
        dim = int(getattr(part_ctrl, "control_dim", None)
                  or getattr(part_ctrl, "action_dim", 0))
        if dim <= 0:
            continue
        layout.append({
            "part": str(name),
            "indices": [idx, idx + dim],
            "controller": type(part_ctrl).__name__,
            "dim": dim,
        })
        idx += dim

    # If we under-counted (e.g. mode channel not in part_controllers), pad
    # the remainder so the layout always covers the full action vector.
    if idx < int(env.action_dim):
        layout.append({
            "part": "extra",
            "indices": [idx, int(env.action_dim)],
            "controller": "unknown",
            "dim": int(env.action_dim) - idx,
        })

    return layout


def _force_arm_osc_absolute(config: dict) -> None:
    """Switch any OSC_POSE arm controller to absolute world-frame targets.

    Default OSC delta mode integrates each tick's delta into the EEF target.
    When the target ends up beyond the arm's reach, the controller piles
    force into the saturated joints and that force gets transferred to the
    mobile base — the arm "drags" the base along, which makes precise
    grasping near the reach limit impossible.

    With absolute mode + world ref frame, the client sends a fixed world
    target every tick; the controller stops integrating, force never
    builds past what's needed to hold position, and the base stays put
    during arm motion."""
    body_parts = config.get("body_parts") or {}
    for part in body_parts.values():
        if not isinstance(part, dict):
            continue
        if part.get("type") in ("OSC_POSE", "OSC_POSITION"):
            part["input_type"] = "absolute"
            part["input_ref_frame"] = "world"


def make_env(
    env_name: str = DEFAULT_ENV,
    robot: str = DEFAULT_ROBOT,
    control_freq: int = DEFAULT_CONTROL_FREQ,
    seed: int | None = 0,
    has_renderer: bool = False,
) -> RobocasaEnv:
    """Build a robocasa env and expose its raw mujoco handles.

    The first import of ``robocasa`` registers all kitchen tasks with
    robosuite's env registry; that import has to happen before
    ``robosuite.make`` is called.

    When ``has_renderer`` is True, robosuite's native ``mjviewer`` is wired
    up — call ``env.render()`` from the main thread to drive the window.
    Use ``mujoco.viewer.launch_passive`` only on light scenes (e.g. the
    Panda+cube sandbox); the kitchen scenes are too heavy for it on WSLg.
    """
    import robosuite
    import robocasa  # noqa: F401  (registration side-effect)
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(robot=robot)
    _force_arm_osc_absolute(controller_config)

    env = robosuite.make(
        env_name=env_name,
        robots=robot,
        controller_configs=controller_config,
        has_renderer=has_renderer,
        renderer="mjviewer" if has_renderer else "mujoco",
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        ignore_done=True,
        control_freq=control_freq,
        translucent_robot=False,
        seed=seed,
    )
    initial_obs = env.reset()

    raw_model = env.sim.model._model
    raw_data = env.sim.data._data

    low, high = env.action_spec
    action_layout = _probe_action_layout(env)

    return RobocasaEnv(
        env=env,
        model=raw_model,
        data=raw_data,
        action_dim=int(env.action_dim),
        control_freq=control_freq,
        action_low=np.asarray(low, dtype=float),
        action_high=np.asarray(high, dtype=float),
        action_layout=action_layout,
        initial_obs=initial_obs,
        env_name=env_name,
        robot_name=robot,
    )
