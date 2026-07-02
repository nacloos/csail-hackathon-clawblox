from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import mujoco
import viser
from mjviser.scene import ViserMujocoScene

from mujoco_recording import RecordingReader


ROOT = Path(__file__).resolve().parent


class WebReplay:
    def __init__(
        self,
        recording: Path,
        *,
        scene: Path | None = None,
        port: int = 8081,
        speed: float = 1.0,
        paused: bool = False,
        loop: bool = True,
    ) -> None:
        self.reader = RecordingReader(recording)
        self.meta = self.reader.meta
        self.scene_path = (scene or self.meta.scene_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.server = viser.ViserServer(port=port)
        self.scene = ViserMujocoScene(self.server, self.model, num_envs=1)

        self.recording = recording
        self.current_tick = 0
        self.total_tick = self.reader.total_tick()
        self.tick_rate = 1.0 / self.meta.timestep
        self.speed = max(0.05, speed)
        self.paused = paused
        self.loop = loop
        self._last_wall_time = time.perf_counter()
        self._last_ui_update = 0.0
        self._setting_slider = False

        self._setup_gui()
        self._apply_current_frame()

    def close(self) -> None:
        self.reader.close()
        self.server.stop()

    def run(self) -> None:
        previous_handler = signal.getsignal(signal.SIGINT)
        interrupted = False

        def on_sigint(signum, frame):
            nonlocal interrupted
            interrupted = True
            signal.signal(signal.SIGINT, signal.SIG_DFL)

        signal.signal(signal.SIGINT, on_sigint)
        try:
            while not interrupted:
                self._tick()
                time.sleep(1.0 / 60.0)
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            self.close()

    def _setup_gui(self) -> None:
        self.server.gui.add_markdown(
            f"**Replay**\n\n"
            f"`{self.recording}`\n\n"
            f"Scene: `{self.scene_path.relative_to(ROOT) if self.scene_path.is_relative_to(ROOT) else self.scene_path}`"
        )
        self.status = self.server.gui.add_html("")
        self.play_button = self.server.gui.add_button(
            "Play" if self.paused else "Pause",
            icon=viser.Icon.PLAYER_PLAY if self.paused else viser.Icon.PLAYER_PAUSE,
        )
        self.tick_slider = self.server.gui.add_slider(
            "Tick",
            min=0,
            max=max(1, self.total_tick),
            step=1,
            initial_value=0,
        )
        self.speed_buttons = self.server.gui.add_button_group(
            "Speed",
            options=["0.25x", "0.5x", "1x", "2x", "4x", "8x"],
        )
        self.loop_checkbox = self.server.gui.add_checkbox("Loop", initial_value=self.loop)

        with self.server.gui.add_folder("Scene", expand_by_default=False):
            self.scene.create_scene_gui()
        with self.server.gui.add_folder("Visualization", expand_by_default=False):
            self.scene.create_overlay_gui()
        with self.server.gui.add_folder("Groups", expand_by_default=False):
            self.scene.create_groups_gui()

        @self.play_button.on_click
        def _(_) -> None:
            self.paused = not self.paused
            self._last_wall_time = time.perf_counter()
            self._sync_play_button()
            self._update_status(force=True)

        @self.tick_slider.on_update
        def _(_) -> None:
            if self._setting_slider:
                return
            self.seek(int(self.tick_slider.value))

        @self.speed_buttons.on_click
        def _(event) -> None:
            self.speed = float(str(event.target.value).removesuffix("x"))
            self._update_status(force=True)

        @self.loop_checkbox.on_update
        def _(_) -> None:
            self.loop = bool(self.loop_checkbox.value)
            self._update_status(force=True)

    def seek(self, tick: int) -> None:
        self.current_tick = min(max(0, tick), self.total_tick)
        self._last_wall_time = time.perf_counter()
        self._apply_current_frame()
        self._update_status(force=True)

    def _tick(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_wall_time
        self._last_wall_time = now

        if not self.paused and self.total_tick > 0:
            if self.current_tick >= self.total_tick:
                if self.loop:
                    self.current_tick = 0
                else:
                    self.paused = True
                    self._sync_play_button()
            else:
                self.current_tick = min(
                    self.total_tick,
                    self.current_tick + round(elapsed * self.tick_rate * self.speed),
                )
            self._apply_current_frame()

        self._update_status()

    def _apply_current_frame(self) -> None:
        frame = self.reader.preview_at_tick(self.current_tick)
        self.data.qpos[:] = frame["qpos"]
        self.data.qvel[:] = frame["qvel"]
        self.data.ctrl[:] = frame["ctrl"]
        self.data.time = float(frame["time"])
        mujoco.mj_forward(self.model, self.data)
        self.scene.update_from_mjdata(self.data)

    def _sync_play_button(self) -> None:
        self.play_button.label = "Play" if self.paused else "Pause"
        self.play_button.icon = viser.Icon.PLAYER_PLAY if self.paused else viser.Icon.PLAYER_PAUSE

    def _update_status(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_ui_update < 0.15:
            return
        self._last_ui_update = now
        self._setting_slider = True
        self.tick_slider.value = int(self.current_tick)
        self._setting_slider = False
        current_seconds = self.current_tick / self.tick_rate
        total_seconds = self.total_tick / self.tick_rate
        self.status.content = (
            "<div style='font-size:0.9em; line-height:1.35'>"
            f"<strong>Status:</strong> {'Paused' if self.paused else 'Playing'}<br/>"
            f"<strong>Tick:</strong> {self.current_tick} / {self.total_tick}<br/>"
            f"<strong>Time:</strong> {current_seconds:.2f}s / {total_seconds:.2f}s<br/>"
            f"<strong>Speed:</strong> {self.speed:g}x"
            "</div>"
        )


def check_replay(recording: Path, scene: Path | None) -> None:
    reader = RecordingReader(recording)
    try:
        meta = reader.meta
        scene_path = (scene or meta.scene_path).resolve()
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        for tick in (0, reader.total_tick() // 2, reader.total_tick()):
            frame = reader.preview_at_tick(tick)
            data.qpos[:] = frame["qpos"]
            data.qvel[:] = frame["qvel"]
            data.ctrl[:] = frame["ctrl"]
            data.time = float(frame["time"])
            mujoco.mj_forward(model, data)
        print(
            "ok: "
            f"recording={recording} "
            f"scene={scene_path} "
            f"preview_frames={reader.preview_count()} "
            f"total_tick={reader.total_tick()}"
        )
    finally:
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a MuJoCo recording in a browser with Viser.")
    parser.add_argument("recording", type=Path)
    parser.add_argument("--scene", type=Path, help="Override scene XML path.")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--check", action="store_true", help="Validate without starting the browser viewer.")
    args = parser.parse_args()

    recording = args.recording.resolve()
    scene = args.scene.resolve() if args.scene else None
    if args.check:
        check_replay(recording, scene)
        return

    replay = WebReplay(
        recording,
        scene=scene,
        port=args.port,
        speed=args.speed,
        paused=args.paused,
        loop=not args.no_loop,
    )
    replay.run()


if __name__ == "__main__":
    main()
