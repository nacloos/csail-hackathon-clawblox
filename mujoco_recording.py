from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


SCHEMA_VERSION = 1
STATE_DTYPE = np.float64
PREVIEW_DTYPE = np.float32


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "recording requires h5py; run with `uv run --with h5py ...`"
        ) from exc
    return h5py


def scene_hash(scene: Path) -> str:
    return hashlib.sha256(scene.read_bytes()).hexdigest()


def default_state_sig() -> int:
    return int(mujoco.mjtState.mjSTATE_INTEGRATION)


def capture_state(model: mujoco.MjModel, data: mujoco.MjData, state_sig: int) -> np.ndarray:
    state = np.empty(mujoco.mj_stateSize(model, state_sig), dtype=STATE_DTYPE)
    mujoco.mj_getState(model, data, state, state_sig)
    return state


def restore_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state: np.ndarray,
    state_sig: int,
) -> None:
    mujoco.mj_setState(model, data, np.asarray(state, dtype=STATE_DTYPE), state_sig)
    mujoco.mj_forward(model, data)


def timestamped_recording_path(record_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return record_dir / f"{stamp}.h5"


@dataclass(frozen=True)
class RecordingConfig:
    preview_hz: float = 30.0
    checkpoint_seconds: float = 1.0
    compression: str = "lzf"


@dataclass(frozen=True)
class RecordingMeta:
    path: Path
    events_path: Path
    scene_path: Path
    scene_sha256: str
    mujoco_version: str
    timestep: float
    nq: int
    nv: int
    nu: int
    state_sig: int
    state_size: int
    preview_count: int
    checkpoint_count: int


class RecordingWriter:
    def __init__(
        self,
        path: Path,
        *,
        scene: Path,
        model: mujoco.MjModel,
        config: RecordingConfig | None = None,
    ) -> None:
        h5py = _require_h5py()
        self.path = path
        self.events_path = path.with_suffix(".events.jsonl")
        self.scene = scene
        self.config = config or RecordingConfig()
        self.state_sig = default_state_sig()
        self.state_size = mujoco.mj_stateSize(model, self.state_sig)
        self.preview_interval_ticks = max(
            1, round(1.0 / (float(model.opt.timestep) * self.config.preview_hz))
        )
        self.checkpoint_interval_ticks = max(
            1, round(self.config.checkpoint_seconds / float(model.opt.timestep))
        )
        self.preview_count = 0
        self.checkpoint_count = 0
        self.last_tick = 0

        path.parent.mkdir(parents=True, exist_ok=True)
        self.h5 = h5py.File(path, "w")
        self.events_file = self.events_path.open("a", encoding="utf-8")
        self._write_attrs(model)
        self._create_datasets(model)

    def record_initial(self, tick: int, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.record_step(tick, model, data, force=True)

    def record_step(
        self,
        tick: int,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        force: bool = False,
    ) -> None:
        self.last_tick = max(self.last_tick, tick)
        if force or tick % self.preview_interval_ticks == 0:
            self._append_preview(tick, data)
        if force or tick % self.checkpoint_interval_ticks == 0:
            self._append_checkpoint(tick, model, data)

    def record_event(self, event: dict[str, Any]) -> None:
        payload = {"wall_time": time.time(), **event}
        self.events_file.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.events_file.flush()

    def flush(self) -> None:
        self.h5.attrs["last_tick"] = self.last_tick
        self.h5.attrs["preview_count"] = self.preview_count
        self.h5.attrs["checkpoint_count"] = self.checkpoint_count
        self.h5.flush()
        self.events_file.flush()

    def close(self) -> None:
        self.record_event({"type": "RecordingClosed", "tick": self.last_tick})
        self.flush()
        self.events_file.close()
        self.h5.close()

    def status(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "events_path": str(self.events_path),
            "preview_count": self.preview_count,
            "checkpoint_count": self.checkpoint_count,
            "last_tick": self.last_tick,
            "preview_hz": self.config.preview_hz,
            "checkpoint_seconds": self.config.checkpoint_seconds,
        }

    def _write_attrs(self, model: mujoco.MjModel) -> None:
        self.h5.attrs["schema_version"] = SCHEMA_VERSION
        self.h5.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.h5.attrs["scene_path"] = str(self.scene)
        self.h5.attrs["scene_sha256"] = scene_hash(self.scene)
        self.h5.attrs["mujoco_version"] = mujoco.__version__
        self.h5.attrs["timestep"] = float(model.opt.timestep)
        self.h5.attrs["nq"] = int(model.nq)
        self.h5.attrs["nv"] = int(model.nv)
        self.h5.attrs["nu"] = int(model.nu)
        self.h5.attrs["state_sig"] = self.state_sig
        self.h5.attrs["state_size"] = self.state_size
        self.h5.attrs["preview_hz"] = float(self.config.preview_hz)
        self.h5.attrs["checkpoint_seconds"] = float(self.config.checkpoint_seconds)
        self.h5.attrs["preview_interval_ticks"] = self.preview_interval_ticks
        self.h5.attrs["checkpoint_interval_ticks"] = self.checkpoint_interval_ticks
        self.h5.attrs["last_tick"] = 0
        self.h5.attrs["preview_count"] = 0
        self.h5.attrs["checkpoint_count"] = 0

    def _create_datasets(self, model: mujoco.MjModel) -> None:
        preview = self.h5.create_group("preview")
        checkpoints = self.h5.create_group("checkpoints")
        compression = self.config.compression

        preview.create_dataset("tick", shape=(0,), maxshape=(None,), dtype="i8", chunks=True)
        preview.create_dataset("time", shape=(0,), maxshape=(None,), dtype="f8", chunks=True)
        preview.create_dataset(
            "qpos",
            shape=(0, model.nq),
            maxshape=(None, model.nq),
            dtype=PREVIEW_DTYPE,
            chunks=(min(1024, max(1, round(self.config.preview_hz * 10))), model.nq),
            compression=compression,
        )
        preview.create_dataset(
            "qvel",
            shape=(0, model.nv),
            maxshape=(None, model.nv),
            dtype=PREVIEW_DTYPE,
            chunks=(min(1024, max(1, round(self.config.preview_hz * 10))), model.nv),
            compression=compression,
        )
        preview.create_dataset(
            "ctrl",
            shape=(0, model.nu),
            maxshape=(None, model.nu),
            dtype=PREVIEW_DTYPE,
            chunks=(min(1024, max(1, round(self.config.preview_hz * 10))), model.nu),
            compression=compression,
        )

        checkpoints.create_dataset("tick", shape=(0,), maxshape=(None,), dtype="i8", chunks=True)
        checkpoints.create_dataset("time", shape=(0,), maxshape=(None,), dtype="f8", chunks=True)
        checkpoints.create_dataset(
            "state",
            shape=(0, self.state_size),
            maxshape=(None, self.state_size),
            dtype=STATE_DTYPE,
            chunks=(1, self.state_size),
            compression=compression,
        )

    def _append_preview(self, tick: int, data: mujoco.MjData) -> None:
        idx = self.preview_count
        group = self.h5["preview"]
        for name in ("tick", "time", "qpos", "qvel", "ctrl"):
            group[name].resize((idx + 1, *group[name].shape[1:]))
        group["tick"][idx] = tick
        group["time"][idx] = float(data.time)
        group["qpos"][idx] = np.asarray(data.qpos, dtype=PREVIEW_DTYPE)
        group["qvel"][idx] = np.asarray(data.qvel, dtype=PREVIEW_DTYPE)
        group["ctrl"][idx] = np.asarray(data.ctrl, dtype=PREVIEW_DTYPE)
        self.preview_count += 1

    def _append_checkpoint(
        self,
        tick: int,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        idx = self.checkpoint_count
        group = self.h5["checkpoints"]
        for name in ("tick", "time", "state"):
            group[name].resize((idx + 1, *group[name].shape[1:]))
        group["tick"][idx] = tick
        group["time"][idx] = float(data.time)
        group["state"][idx] = capture_state(model, data, self.state_sig)
        self.checkpoint_count += 1


class RecordingReader:
    def __init__(self, path: Path) -> None:
        h5py = _require_h5py()
        self.path = path
        self.h5 = h5py.File(path, "r")

    @property
    def meta(self) -> RecordingMeta:
        return RecordingMeta(
            path=self.path,
            events_path=self.path.with_suffix(".events.jsonl"),
            scene_path=Path(str(self.h5.attrs["scene_path"])),
            scene_sha256=str(self.h5.attrs["scene_sha256"]),
            mujoco_version=str(self.h5.attrs["mujoco_version"]),
            timestep=float(self.h5.attrs["timestep"]),
            nq=int(self.h5.attrs["nq"]),
            nv=int(self.h5.attrs["nv"]),
            nu=int(self.h5.attrs["nu"]),
            state_sig=int(self.h5.attrs["state_sig"]),
            state_size=int(self.h5.attrs["state_size"]),
            preview_count=len(self.h5["preview/tick"]),
            checkpoint_count=len(self.h5["checkpoints/tick"]),
        )

    def preview_count(self) -> int:
        return len(self.h5["preview/tick"])

    def checkpoint_count(self) -> int:
        return len(self.h5["checkpoints/tick"])

    def total_tick(self) -> int:
        ticks = self.h5["preview/tick"]
        if len(ticks) == 0:
            return 0
        return int(ticks[-1])

    def preview_at_tick(self, tick: int) -> dict[str, np.ndarray | float | int]:
        ticks = self.h5["preview/tick"]
        if len(ticks) == 0:
            raise ValueError("recording has no preview frames")
        idx = int(np.searchsorted(ticks, tick, side="right") - 1)
        idx = min(max(idx, 0), len(ticks) - 1)
        preview = self.h5["preview"]
        return {
            "tick": int(preview["tick"][idx]),
            "time": float(preview["time"][idx]),
            "qpos": np.asarray(preview["qpos"][idx], dtype=STATE_DTYPE),
            "qvel": np.asarray(preview["qvel"][idx], dtype=STATE_DTYPE),
            "ctrl": np.asarray(preview["ctrl"][idx], dtype=STATE_DTYPE),
        }

    def checkpoint_at_or_before(self, tick: int) -> dict[str, np.ndarray | float | int]:
        ticks = self.h5["checkpoints/tick"]
        if len(ticks) == 0:
            raise ValueError("recording has no checkpoints")
        idx = int(np.searchsorted(ticks, tick, side="right") - 1)
        idx = min(max(idx, 0), len(ticks) - 1)
        checkpoints = self.h5["checkpoints"]
        return {
            "tick": int(checkpoints["tick"][idx]),
            "time": float(checkpoints["time"][idx]),
            "state": np.asarray(checkpoints["state"][idx], dtype=STATE_DTYPE),
        }

    def close(self) -> None:
        self.h5.close()
