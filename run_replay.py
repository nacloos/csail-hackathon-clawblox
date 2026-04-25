from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer

from mujoco_recording import RecordingReader


LEFT_ARROW = 263
RIGHT_ARROW = 262
HOME = 268
END = 269


class ReplayController:
    def __init__(
        self,
        reader: RecordingReader,
        *,
        speed: float = 1.0,
        paused: bool = False,
        loop: bool = True,
    ) -> None:
        self.reader = reader
        self.speed = max(0.1, speed)
        self.paused = paused
        self.loop = loop
        self.current_tick = 0
        self.total_tick = reader.total_tick()
        self.last_wall_time = time.perf_counter()
        self.tick_rate = round(1.0 / reader.meta.timestep)

    def key_callback(self, key: int) -> None:
        if key == ord(" "):
            self.paused = not self.paused
        elif key == LEFT_ARROW:
            self.seek(self.current_tick - self.tick_rate)
        elif key == RIGHT_ARROW:
            self.seek(self.current_tick + self.tick_rate)
        elif key == HOME:
            self.seek(0)
        elif key == END:
            self.seek(self.total_tick)
        elif key in (ord("["), ord(",")):
            self.speed = max(0.1, self.speed / 2.0)
        elif key in (ord("]"), ord(".")):
            self.speed = min(100.0, self.speed * 2.0)

    def seek(self, tick: int) -> None:
        self.current_tick = min(max(0, tick), self.total_tick)
        self.last_wall_time = time.perf_counter()

    def advance(self) -> None:
        now = time.perf_counter()
        elapsed = now - self.last_wall_time
        self.last_wall_time = now
        if self.paused:
            return
        if self.current_tick >= self.total_tick:
            if self.loop:
                self.current_tick = 0
            return
        self.current_tick = min(
            self.total_tick,
            self.current_tick + round(elapsed * self.tick_rate * self.speed),
        )

    def apply_preview(self, data: mujoco.MjData) -> int:
        frame = self.reader.preview_at_tick(self.current_tick)
        data.qpos[:] = frame["qpos"]
        data.qvel[:] = frame["qvel"]
        data.ctrl[:] = frame["ctrl"]
        data.time = float(frame["time"])
        return int(frame["tick"])


def check_recording(recording: Path, scene: Path | None) -> None:
    reader = RecordingReader(recording)
    try:
        meta = reader.meta
        scene_path = scene or meta.scene_path
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        first = reader.preview_at_tick(0)
        middle = reader.preview_at_tick(reader.total_tick() // 2)
        last = reader.preview_at_tick(reader.total_tick())
        for frame in (first, middle, last):
            data.qpos[:] = frame["qpos"]
            data.qvel[:] = frame["qvel"]
            data.ctrl[:] = frame["ctrl"]
            data.time = float(frame["time"])
            mujoco.mj_forward(model, data)
        checkpoint = reader.checkpoint_at_or_before(reader.total_tick())
        print(
            "ok: "
            f"preview_frames={reader.preview_count()} "
            f"checkpoints={reader.checkpoint_count()} "
            f"total_tick={reader.total_tick()} "
            f"checkpoint_tick={checkpoint['tick']}"
        )
    finally:
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a recorded MuJoCo Panda session.")
    parser.add_argument("recording", type=Path)
    parser.add_argument("--scene", type=Path, help="Override scene XML path.")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--no-loop", action="store_true", help="Stop on the final frame instead of looping.")
    parser.add_argument("--check", action="store_true", help="Validate the recording without opening a viewer.")
    args = parser.parse_args()

    if args.check:
        check_recording(args.recording, args.scene)
        return

    reader = RecordingReader(args.recording)
    try:
        scene = args.scene or reader.meta.scene_path
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        controller = ReplayController(reader, speed=args.speed, paused=args.paused, loop=not args.no_loop)

        print(f"Replaying: {args.recording}")
        print("Keys: space play/pause, arrows seek, home/end jump, [/] speed")
        with mujoco.viewer.launch_passive(model, data, key_callback=controller.key_callback) as viewer:
            controller.last_wall_time = time.perf_counter()
            while viewer.is_running():
                frame_tick = controller.apply_preview(data)
                mujoco.mj_forward(model, data)
                viewer.sync()
                print(
                    f"\rtick={frame_tick}/{controller.total_tick} "
                    f"speed={controller.speed:.2f} "
                    f"{'paused' if controller.paused else 'playing'}",
                    end="",
                    flush=True,
                )
                time.sleep(1.0 / 60.0)
                controller.advance()
    finally:
        reader.close()


if __name__ == "__main__":
    main()
