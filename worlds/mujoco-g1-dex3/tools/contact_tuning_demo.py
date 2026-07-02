from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import threading
import time
from typing import Any

import mujoco
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from PIL import Image
import uvicorn


ROOT = Path(__file__).resolve().parents[3]
WORLD = ROOT / "worlds" / "mujoco-g1-dex3"
SCENE = WORLD / "models" / "g1" / "scene_hands_modified.xml"
SNAPSHOT_DIR = (
    WORLD
    / "results"
    / "genexp-python-20260612T151414Z-fable"
    / "runs"
    / "genexp-python-20260612T151414Z-g010"
    / "agents"
    / "Eko-rgenexp-python-20260612T151414Z-g010-w0-a0"
    / "workspace"
    / "g1"
)

SCENARIOS = {
    "hover": SNAPSHOT_DIR / "hover_snap.npz",
    "slide": SNAPSHOT_DIR / "slide_snap.npz",
    "rake": SNAPSHOT_DIR / "rake_snap.npz",
}


class ContactParams(BaseModel):
    scenario: str = "hover"
    brick_mass: float = Field(default=0.08, gt=0)
    solref_time: float = Field(default=0.02, gt=0)
    solref_damping: float = Field(default=1.0, gt=0)
    solimp0: float = Field(default=0.9, ge=0, le=1)
    solimp1: float = Field(default=0.95, ge=0, le=1)
    solimp2: float = Field(default=0.001, ge=0)
    speed: int = Field(default=8, ge=1, le=100)
    arm_kp: float = Field(default=80.0, ge=0)
    arm_kd: float = Field(default=8.0, ge=0)


class StepRequest(BaseModel):
    steps: int = Field(default=8, ge=1, le=500)


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, idx: int) -> str:
    return mujoco.mj_id2name(model, obj, idx) or f"{obj.name}:{idx}"


class DemoSim:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.params = ContactParams()
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.camera = mujoco.MjvCamera()
        self.qtarget = np.zeros(self.model.nq)
        self.motor_actuators: list[int] = []
        self.position_actuators: list[int] = []
        self.hand_geoms: set[int] = set()
        self.brick_geoms: set[int] = set()
        self.table_geoms: set[int] = set()
        self.target_geoms: list[int] = []
        self.tick = 0
        self.reset(self.params)

    def close(self) -> None:
        return None

    def reset(self, params: ContactParams) -> dict[str, Any]:
        with self.lock:
            self.params = params
            self._collect_ids()
            self._apply_params()
            mujoco.mj_resetData(self.model, self.data)
            self._load_snapshot(params.scenario)
            self._sync_targets_from_qpos()
            mujoco.mj_forward(self.model, self.data)
            self.tick = 0
            return self.metrics()

    def step(self, steps: int) -> dict[str, Any]:
        with self.lock:
            for _ in range(steps):
                self._apply_pose_hold_control()
                mujoco.mj_step(self.model, self.data)
                self.tick += 1
            return self.metrics()

    def render_png(self) -> bytes:
        with self.lock:
            brick = self.data.xpos[self.model.body("brick_1").id]
            self.camera.lookat[:] = [brick[0], brick[1], max(0.75, brick[2])]
            self.camera.distance = 0.75
            self.camera.azimuth = 135
            self.camera.elevation = -18
            renderer = mujoco.Renderer(self.model, height=420, width=640)
            try:
                renderer.update_scene(self.data, camera=self.camera)
                image = renderer.render()
            finally:
                renderer.close()
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        return buf.getvalue()

    def metrics(self) -> dict[str, Any]:
        hand_brick = []
        table_brick = []
        brick_any = []
        deepest = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            dist = float(contact.dist)
            if g1 in self.brick_geoms or g2 in self.brick_geoms:
                brick_any.append(dist)
                deepest.append(
                    {
                        "dist_mm": dist * 1000,
                        "a": self._geom_label(g1),
                        "b": self._geom_label(g2),
                    }
                )
            if (g1 in self.hand_geoms and g2 in self.brick_geoms) or (
                g2 in self.hand_geoms and g1 in self.brick_geoms
            ):
                hand_brick.append(dist)
            if (g1 in self.table_geoms and g2 in self.brick_geoms) or (
                g2 in self.table_geoms and g1 in self.brick_geoms
            ):
                table_brick.append(dist)

        deepest.sort(key=lambda item: item["dist_mm"])
        brick1 = self.data.xpos[self.model.body("brick_1").id].copy()
        return {
            "time": float(self.data.time),
            "tick": self.tick,
            "params": self.params.model_dump(),
            "hand_brick_count": len(hand_brick),
            "table_brick_count": len(table_brick),
            "brick_contact_count": len(brick_any),
            "min_hand_brick_mm": min(hand_brick) * 1000 if hand_brick else None,
            "min_table_brick_mm": min(table_brick) * 1000 if table_brick else None,
            "min_brick_any_mm": min(brick_any) * 1000 if brick_any else None,
            "brick1_pos": brick1.tolist(),
            "deepest": deepest[:8],
        }

    def _collect_ids(self) -> None:
        self.hand_geoms.clear()
        self.brick_geoms.clear()
        self.table_geoms.clear()
        for geom_id in range(self.model.ngeom):
            geom_name = _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            body_name = _name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.geom_bodyid[geom_id]),
            )
            if "hand" in body_name or "wrist_yaw" in body_name:
                self.hand_geoms.add(geom_id)
            if geom_name.startswith("brick_"):
                self.brick_geoms.add(geom_id)
            if "table" in body_name or "table" in geom_name:
                self.table_geoms.add(geom_id)
        self.target_geoms = sorted(self.hand_geoms | self.brick_geoms | self.table_geoms)

        self.motor_actuators.clear()
        self.position_actuators.clear()
        for actuator_id in range(self.model.nu):
            name = _name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            if "hand" in name:
                self.position_actuators.append(actuator_id)
            else:
                self.motor_actuators.append(actuator_id)

    def _apply_params(self) -> None:
        params = self.params
        for body_name in [f"brick_{idx}" for idx in range(5)]:
            body_id = self.model.body(body_name).id
            scale = params.brick_mass / float(self.model.body_mass[body_id])
            self.model.body_mass[body_id] = params.brick_mass
            self.model.body_inertia[body_id] *= scale

        ids = np.array(self.target_geoms, dtype=np.int32)
        self.model.geom_solref[ids, 0] = params.solref_time
        self.model.geom_solref[ids, 1] = params.solref_damping
        self.model.geom_solimp[ids, 0] = params.solimp0
        self.model.geom_solimp[ids, 1] = params.solimp1
        self.model.geom_solimp[ids, 2] = params.solimp2
        mujoco.mj_setConst(self.model, self.data)

    def _load_snapshot(self, scenario: str) -> None:
        path = SCENARIOS.get(scenario, SCENARIOS["hover"])
        state = np.load(path)
        self.data.qpos[:] = state["qpos"]
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        self.data.time = 0

    def _sync_targets_from_qpos(self) -> None:
        self.qtarget = self.data.qpos.copy()
        for actuator_id in self.position_actuators:
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            qadr = int(self.model.jnt_qposadr[joint_id])
            value = float(self.data.qpos[qadr])
            if self.model.actuator_ctrllimited[actuator_id]:
                lo, hi = self.model.actuator_ctrlrange[actuator_id]
                value = max(float(lo), min(float(hi), value))
            self.data.ctrl[actuator_id] = value

    def _apply_pose_hold_control(self) -> None:
        for actuator_id in self.position_actuators:
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            qadr = int(self.model.jnt_qposadr[joint_id])
            value = float(self.qtarget[qadr])
            if self.model.actuator_ctrllimited[actuator_id]:
                lo, hi = self.model.actuator_ctrlrange[actuator_id]
                value = max(float(lo), min(float(hi), value))
            self.data.ctrl[actuator_id] = value

        for actuator_id in self.motor_actuators:
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            qadr = int(self.model.jnt_qposadr[joint_id])
            vadr = int(self.model.jnt_dofadr[joint_id])
            torque = self.params.arm_kp * (self.qtarget[qadr] - self.data.qpos[qadr])
            torque -= self.params.arm_kd * self.data.qvel[vadr]
            if self.model.actuator_ctrllimited[actuator_id]:
                lo, hi = self.model.actuator_ctrlrange[actuator_id]
                torque = max(float(lo), min(float(hi), torque))
            self.data.ctrl[actuator_id] = torque

    def _geom_label(self, geom_id: int) -> str:
        geom = _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body = _name(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            int(self.model.geom_bodyid[geom_id]),
        )
        return f"{geom}/{body}"


class ContactViewer:
    def __init__(self, sim: DemoSim, *, host: str, port: int) -> None:
        import viser
        from mjviser.scene import ViserMujocoScene

        self.sim = sim
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/"
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.viewer_model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.viewer_data = mujoco.MjData(self.viewer_model)
        self.server = viser.ViserServer(host=host, port=port, label="G1 Dex3 contact tuning")
        self.scene = ViserMujocoScene(self.server, self.viewer_model, num_envs=1)
        self.status = self.server.gui.add_html("")
        with self.server.gui.add_folder("Scene", expand_by_default=False):
            self.scene.create_scene_gui()
        with self.server.gui.add_folder("Visualization", expand_by_default=False):
            self.scene.create_overlay_gui()
        with self.server.gui.add_folder("Groups", expand_by_default=False):
            self.scene.create_groups_gui()

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, name="contact-tuning-viewer", daemon=True)
        self.thread.start()
        print(f"Interactive 3D viewer: {self.url}", flush=True)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.server.stop()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            loop_start = time.perf_counter()
            with self.sim.lock:
                self.viewer_data.qpos[:] = self.sim.data.qpos
                self.viewer_data.qvel[:] = self.sim.data.qvel
                self.viewer_data.ctrl[:] = self.sim.data.ctrl
                self.viewer_data.mocap_pos[:] = self.sim.data.mocap_pos
                self.viewer_data.mocap_quat[:] = self.sim.data.mocap_quat
                tick = self.sim.tick
                sim_time = float(self.sim.data.time)
                scenario = self.sim.params.scenario
                solref_time = self.sim.params.solref_time
            mujoco.mj_forward(self.viewer_model, self.viewer_data)
            self.scene.update_from_mjdata(self.viewer_data)
            self.status.content = (
                "<div style='font-size:0.9em; line-height:1.35'>"
                "<strong>G1 Dex3 contact tuning</strong><br/>"
                f"Scenario: {scenario}<br/>"
                f"Tick: {tick}<br/>"
                f"Time: {sim_time:.3f}s<br/>"
                f"solref[0]: {solref_time:g}"
                "</div>"
            )
            sleep_s = max(0.0, (1.0 / 30.0) - (time.perf_counter() - loop_start))
            self.stop_event.wait(sleep_s)


class ContactTuningViserApp:
    def __init__(self, sim: DemoSim, *, host: str, port: int) -> None:
        import viser
        from mjviser.scene import ViserMujocoScene

        self.sim = sim
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/"
        self.stop_event = threading.Event()
        self.playing = False
        self.thread: threading.Thread | None = None
        self.viewer_model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.viewer_data = mujoco.MjData(self.viewer_model)
        self.server = viser.ViserServer(host=host, port=port, label="G1 Dex3 contact tuning")
        self.scene = ViserMujocoScene(self.server, self.viewer_model, num_envs=1)
        self._building_gui = False
        self._viewer_update_lock = threading.Lock()

        self.server.gui.add_markdown(
            "**G1 Dex3 contact tuning**\n\n"
            "Same `scene_hands_modified.xml`, same saved robot snapshots. "
            "Use mouse controls in the viewport to orbit, pan, and zoom."
        )
        self._build_controls()

        with self.server.gui.add_folder("Viewer", expand_by_default=False):
            self.scene.create_scene_gui()
        with self.server.gui.add_folder("Visualization", expand_by_default=False):
            self.scene.create_overlay_gui()
        with self.server.gui.add_folder("Groups", expand_by_default=False):
            self.scene.create_groups_gui()

    def _build_controls(self) -> None:
        self._building_gui = True
        with self.server.gui.add_folder("Scenario", expand_by_default=True):
            self.scenario = self.server.gui.add_dropdown(
                "Snapshot",
                options=tuple(SCENARIOS.keys()),
                initial_value=self.sim.params.scenario,
            )
            self.reset_button = self.server.gui.add_button("Reset")
            self.step_button = self.server.gui.add_button("Step")
            self.play_button = self.server.gui.add_button("Play")
        with self.server.gui.add_folder("Contact parameters", expand_by_default=True):
            self.brick_mass = self.server.gui.add_slider("Brick mass kg", 0.005, 0.5, 0.005, self.sim.params.brick_mass)
            self.solref_time = self.server.gui.add_slider(
                "solref response time", 0.001, 0.08, 0.001, self.sim.params.solref_time
            )
            self.solref_damping = self.server.gui.add_slider(
                "solref damping", 0.2, 3.0, 0.05, self.sim.params.solref_damping
            )
            self.solimp0 = self.server.gui.add_slider("solimp 0", 0.1, 1.0, 0.01, self.sim.params.solimp0)
            self.solimp1 = self.server.gui.add_slider("solimp 1", 0.1, 1.0, 0.01, self.sim.params.solimp1)
            self.solimp2 = self.server.gui.add_slider("solimp 2", 0.0001, 0.05, 0.0001, self.sim.params.solimp2)
            self.default_button = self.server.gui.add_button("Default contact")
            self.hard_button = self.server.gui.add_button("Hard contact")
        with self.server.gui.add_folder("Pose hold", expand_by_default=False):
            self.arm_kp = self.server.gui.add_slider("Arm hold kp", 0, 300, 5, self.sim.params.arm_kp)
            self.arm_kd = self.server.gui.add_slider("Arm hold kd", 0, 40, 1, self.sim.params.arm_kd)
            self.speed = self.server.gui.add_slider("Steps per tick", 1, 80, 1, self.sim.params.speed)

        @self.scenario.on_update
        def _(_) -> None:
            self.reset()

        for handle in (
            self.brick_mass,
            self.solref_time,
            self.solref_damping,
            self.solimp0,
            self.solimp1,
            self.solimp2,
            self.arm_kp,
            self.arm_kd,
            self.speed,
        ):
            handle.on_update(lambda _: self.reset())

        @self.reset_button.on_click
        def _(_) -> None:
            self.reset()

        @self.step_button.on_click
        def _(_) -> None:
            self.sim.step(int(self.speed.value))

        @self.play_button.on_click
        def _(_) -> None:
            self.playing = not self.playing
            self.play_button.label = "Pause" if self.playing else "Play"

        @self.default_button.on_click
        def _(_) -> None:
            self._set_contact_values(0.08, 0.02, 1.0, 0.9, 0.95, 0.001)
            self.reset()

        @self.hard_button.on_click
        def _(_) -> None:
            self._set_contact_values(float(self.brick_mass.value), 0.004, 1.0, 0.99, 0.995, 0.0001)
            self.reset()

        self._building_gui = False

    def _set_contact_values(
        self,
        brick_mass: float,
        solref_time: float,
        solref_damping: float,
        solimp0: float,
        solimp1: float,
        solimp2: float,
    ) -> None:
        self.brick_mass.value = brick_mass
        self.solref_time.value = solref_time
        self.solref_damping.value = solref_damping
        self.solimp0.value = solimp0
        self.solimp1.value = solimp1
        self.solimp2.value = solimp2

    def _params_from_gui(self) -> ContactParams:
        return ContactParams(
            scenario=str(self.scenario.value),
            brick_mass=float(self.brick_mass.value),
            solref_time=float(self.solref_time.value),
            solref_damping=float(self.solref_damping.value),
            solimp0=float(self.solimp0.value),
            solimp1=float(self.solimp1.value),
            solimp2=float(self.solimp2.value),
            speed=int(self.speed.value),
            arm_kp=float(self.arm_kp.value),
            arm_kd=float(self.arm_kd.value),
        )

    def reset(self) -> None:
        if self._building_gui:
            return
        self.sim.reset(self._params_from_gui())

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, name="contact-tuning-viser", daemon=True)
        self.thread.start()
        print(f"Contact tuning app: {self.url}", flush=True)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.server.stop()

    def _run(self) -> None:
        self._update_viewer_and_status()
        while not self.stop_event.is_set():
            loop_start = time.perf_counter()
            if self.playing:
                self.sim.step(int(self.speed.value))
            self._update_viewer_and_status()
            sleep_s = max(0.0, (1.0 / 30.0) - (time.perf_counter() - loop_start))
            self.stop_event.wait(sleep_s)

    def _update_viewer_and_status(self) -> None:
        with self._viewer_update_lock:
            with self.sim.lock:
                self.viewer_data.qpos[:] = self.sim.data.qpos
                self.viewer_data.qvel[:] = self.sim.data.qvel
                self.viewer_data.ctrl[:] = self.sim.data.ctrl
                self.viewer_data.mocap_pos[:] = self.sim.data.mocap_pos
                self.viewer_data.mocap_quat[:] = self.sim.data.mocap_quat
            mujoco.mj_forward(self.viewer_model, self.viewer_data)
            self.scene.update_from_mjdata(self.viewer_data)


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>G1 Dex3 Contact Tuning</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f2ee;
      color: #191919;
    }
    * { box-sizing: border-box; }
    body { margin: 0; }
    header {
      padding: 14px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid #d8d2c8;
      background: #ffffff;
    }
    h1 { font-size: 18px; margin: 0; font-weight: 700; }
    a { color: #1d6f68; font-weight: 700; text-decoration: none; }
    main {
      display: grid;
      grid-template-columns: minmax(360px, 1fr) 360px;
      gap: 16px;
      padding: 16px;
      min-height: calc(100vh - 58px);
    }
    .viewer, .panel, .plot {
      background: #fff;
      border: 1px solid #d8d2c8;
      border-radius: 8px;
      overflow: hidden;
    }
    .viewer {
      display: grid;
      grid-template-rows: minmax(260px, 1fr) 180px;
      min-height: 600px;
    }
    #frame {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #e8e4dc;
      display: block;
    }
    .plot { padding: 12px; border-left: 0; border-right: 0; border-bottom: 0; border-radius: 0; }
    canvas { width: 100%; height: 150px; display: block; }
    .panel { padding: 14px; overflow: auto; }
    .row { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center; margin: 12px 0; }
    label { font-size: 13px; font-weight: 650; color: #333; }
    input[type="range"] { width: 100%; }
    select, button {
      height: 34px;
      border: 1px solid #bbb4aa;
      background: #fff;
      border-radius: 6px;
      padding: 0 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    button.primary { background: #1d6f68; border-color: #1d6f68; color: #fff; }
    .buttons { display: flex; gap: 8px; margin: 10px 0 16px; }
    .value { font-variant-numeric: tabular-nums; color: #555; font-size: 12px; min-width: 76px; text-align: right; }
    .metrics {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 12px 0;
    }
    .metric {
      border: 1px solid #e0dbd2;
      border-radius: 6px;
      padding: 8px;
      min-height: 58px;
    }
    .metric span { display: block; font-size: 12px; color: #6a625a; }
    .metric strong { display: block; margin-top: 5px; font-size: 18px; font-variant-numeric: tabular-nums; }
    .section-title { margin: 18px 0 8px; font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: 0; color: #6a625a; }
    pre {
      background: #f7f5f1;
      border-radius: 6px;
      padding: 10px;
      overflow: auto;
      max-height: 180px;
      font-size: 12px;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      .viewer { min-height: 420px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>G1 Dex3 Contact Tuning</h1>
    <div><a id="viewerLink" href="http://127.0.0.1:9098/" target="_blank" rel="noreferrer">Interactive 3D viewer</a> · <span id="status">paused</span></div>
  </header>
  <main>
    <section class="viewer">
      <img id="frame" alt="MuJoCo rendered frame" />
      <div class="plot"><canvas id="plot" width="900" height="150"></canvas></div>
    </section>
    <aside class="panel">
      <div class="row">
        <label for="scenario">Scenario</label>
        <select id="scenario">
          <option value="hover">hover_snap</option>
          <option value="slide">slide_snap</option>
          <option value="rake">rake_snap</option>
        </select>
      </div>
      <div class="buttons">
        <button class="primary" id="play">Play</button>
        <button id="step">Step</button>
        <button id="reset">Reset</button>
        <button id="default">Default</button>
        <button id="hard">Hard</button>
      </div>

      <div class="metrics">
        <div class="metric"><span>Hand-brick penetration</span><strong id="hb">--</strong></div>
        <div class="metric"><span>Table-brick penetration</span><strong id="tb">--</strong></div>
        <div class="metric"><span>Brick height</span><strong id="bz">--</strong></div>
        <div class="metric"><span>Contacts</span><strong id="contacts">--</strong></div>
      </div>

      <div class="section-title">Contact</div>
      <div id="controls"></div>

      <div class="section-title">Deepest Brick Contacts</div>
      <pre id="deepest"></pre>
    </aside>
  </main>
  <script>
    const controls = [
      ["brick_mass", "Brick mass kg", 0.005, 0.5, 0.005],
      ["solref_time", "solref response time", 0.001, 0.08, 0.001],
      ["solref_damping", "solref damping", 0.2, 3.0, 0.05],
      ["solimp0", "solimp 0", 0.1, 1.0, 0.01],
      ["solimp1", "solimp 1", 0.1, 1.0, 0.01],
      ["solimp2", "solimp 2", 0.0001, 0.05, 0.0001],
      ["arm_kp", "pose hold kp", 0, 300, 5],
      ["arm_kd", "pose hold kd", 0, 40, 1],
      ["speed", "steps per tick", 1, 80, 1],
    ];
    const params = {
      scenario: "hover",
      brick_mass: 0.08,
      solref_time: 0.02,
      solref_damping: 1.0,
      solimp0: 0.9,
      solimp1: 0.95,
      solimp2: 0.001,
      arm_kp: 80,
      arm_kd: 8,
      speed: 8,
    };
    const history = [];
    let playing = false;
    let lastMetrics = null;
    const controlsEl = document.getElementById("controls");

    function fmtMm(value) {
      if (value === null || value === undefined) return "--";
      return `${value.toFixed(2)} mm`;
    }
    function fmtM(value) {
      if (value === null || value === undefined) return "--";
      return `${value.toFixed(3)} m`;
    }
    function makeControls() {
      controlsEl.innerHTML = "";
      for (const [key, label, min, max, step] of controls) {
        const wrap = document.createElement("div");
        wrap.className = "row";
        const left = document.createElement("div");
        const lab = document.createElement("label");
        lab.textContent = label;
        const range = document.createElement("input");
        range.type = "range";
        range.min = min;
        range.max = max;
        range.step = step;
        range.value = params[key];
        range.id = key;
        left.append(lab, range);
        const val = document.createElement("div");
        val.className = "value";
        val.id = `${key}_value`;
        val.textContent = params[key];
        range.addEventListener("input", () => {
          params[key] = Number(range.value);
          val.textContent = Number(range.value).toPrecision(4);
        });
        range.addEventListener("change", reset);
        wrap.append(left, val);
        controlsEl.append(wrap);
      }
    }
    function syncControls() {
      for (const [key] of controls) {
        const input = document.getElementById(key);
        const val = document.getElementById(`${key}_value`);
        if (input) input.value = params[key];
        if (val) val.textContent = Number(params[key]).toPrecision(4);
      }
      document.getElementById("scenario").value = params.scenario;
    }
    async function postJson(url, body) {
      const res = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    }
    async function reset() {
      params.scenario = document.getElementById("scenario").value;
      history.length = 0;
      lastMetrics = await postJson("/api/reset", params);
      updateMetrics(lastMetrics);
      await updateFrame();
    }
    async function loadInfo() {
      const res = await fetch("/api/info");
      const info = await res.json();
      if (info.viewer_url) document.getElementById("viewerLink").href = info.viewer_url;
    }
    async function stepOnce() {
      lastMetrics = await postJson("/api/step", {steps: params.speed});
      updateMetrics(lastMetrics);
      await updateFrame();
    }
    async function updateFrame() {
      const img = document.getElementById("frame");
      img.src = `/api/frame?t=${Date.now()}`;
    }
    function updateMetrics(m) {
      const hb = m.min_hand_brick_mm;
      const tb = m.min_table_brick_mm;
      document.getElementById("hb").textContent = fmtMm(hb);
      document.getElementById("tb").textContent = fmtMm(tb);
      document.getElementById("bz").textContent = fmtM(m.brick1_pos[2]);
      document.getElementById("contacts").textContent = `${m.hand_brick_count}/${m.table_brick_count}/${m.brick_contact_count}`;
      document.getElementById("status").textContent = `${playing ? "playing" : "paused"} · t=${m.time.toFixed(3)}s`;
      document.getElementById("deepest").textContent = m.deepest
        .map(c => `${c.dist_mm.toFixed(2)} mm  ${c.a}  <->  ${c.b}`)
        .join("\n");
      history.push({t: m.time, hb, tb});
      if (history.length > 300) history.shift();
      drawPlot();
    }
    function drawPlot() {
      const canvas = document.getElementById("plot");
      const ctx = canvas.getContext("2d");
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#fbfaf7";
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = "#ddd6cc";
      ctx.beginPath();
      for (let y = 20; y < h; y += 32) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
      ctx.stroke();
      const values = history.flatMap(p => [p.hb, p.tb]).filter(v => v !== null && v !== undefined);
      const min = Math.min(-1, ...values);
      const max = Math.max(1, ...values);
      function line(key, color) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        let started = false;
        history.forEach((p, i) => {
          const v = p[key];
          if (v === null || v === undefined) { started = false; return; }
          const x = history.length <= 1 ? 0 : i / (history.length - 1) * w;
          const y = h - ((v - min) / (max - min)) * (h - 20) - 10;
          if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
        });
        ctx.stroke();
      }
      line("hb", "#b6403a");
      line("tb", "#1d6f68");
      ctx.fillStyle = "#333";
      ctx.font = "12px system-ui";
      ctx.fillText("red: hand-brick mm, green: table-brick mm", 10, 18);
    }
    async function loop() {
      if (playing) {
        try { await stepOnce(); } catch (err) { console.error(err); playing = false; }
      }
      requestAnimationFrame(loop);
    }
    document.getElementById("play").addEventListener("click", () => {
      playing = !playing;
      document.getElementById("play").textContent = playing ? "Pause" : "Play";
    });
    document.getElementById("step").addEventListener("click", stepOnce);
    document.getElementById("reset").addEventListener("click", reset);
    document.getElementById("scenario").addEventListener("change", reset);
    document.getElementById("default").addEventListener("click", () => {
      Object.assign(params, {brick_mass: 0.08, solref_time: 0.02, solref_damping: 1, solimp0: 0.9, solimp1: 0.95, solimp2: 0.001});
      syncControls(); reset();
    });
    document.getElementById("hard").addEventListener("click", () => {
      Object.assign(params, {solref_time: 0.004, solref_damping: 1, solimp0: 0.99, solimp1: 0.995, solimp2: 0.0001});
      syncControls(); reset();
    });
    makeControls();
    loadInfo();
    reset();
    loop();
  </script>
</body>
</html>
"""


app = FastAPI(title="G1 Dex3 Contact Tuning Demo")
sim = DemoSim()
viewer_url: str | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


@app.get("/api/info")
def info() -> dict[str, Any]:
    return {
        "scene": str(SCENE),
        "scenarios": {name: str(path) for name, path in SCENARIOS.items()},
        "viewer_url": viewer_url,
    }


@app.post("/api/reset")
def reset(params: ContactParams) -> dict[str, Any]:
    return sim.reset(params)


@app.post("/api/step")
def step(req: StepRequest) -> dict[str, Any]:
    return sim.step(req.steps)


@app.get("/api/frame")
def frame() -> Response:
    return Response(sim.render_png(), media_type="image/png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive contact tuning demo for the G1 Dex3 brick scene.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8098)
    args = parser.parse_args()
    tuning_app = ContactTuningViserApp(sim, host=args.host, port=args.port)
    tuning_app.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        tuning_app.stop()


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
