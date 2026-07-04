from __future__ import annotations

import argparse
import asyncio
import http.client
import signal
import socket
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import mujoco
import numpy as np
import uvicorn
import viser
from fastapi import FastAPI
from fastapi import Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from mjviser.scene import ViserMujocoScene
import websockets

from mujoco_recording import RecordingReader


ROOT = Path(__file__).resolve().parent


def choose_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class MujocoReplayAdapter:
    def __init__(
        self,
        recording: Path,
        *,
        scene: Path | None = None,
        viewer_port: int | None = None,
        speed: float = 1.0,
        paused: bool = True,
        loop: bool = True,
    ) -> None:
        self.reader = RecordingReader(recording)
        self.meta = self.reader.meta
        self.recording = recording
        self.scene_path = (scene or self.meta.scene_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.viewer_port = viewer_port or choose_free_port()
        self.viewer_url = f"http://127.0.0.1:{self.viewer_port}/"
        self.server = viser.ViserServer(port=self.viewer_port)
        self.scene = ViserMujocoScene(self.server, self.model, num_envs=1)

        self.current_tick = 0
        self.total_tick = self.reader.total_tick()
        self.tick_rate = max(1, round(1.0 / self.meta.timestep))
        self.duration_ms = round((self.total_tick / self.tick_rate) * 1000)
        self.active_ranges = self._load_active_ranges()
        self.speed = max(0.05, float(speed))
        self.paused = bool(paused)
        self.loop = bool(loop)
        self.skip_idle = False
        self.pause_at_tick: int | None = None

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._last_wall_time = time.perf_counter()
        self._thread = threading.Thread(target=self._run_loop, name="mujoco-replay", daemon=True)
        self._setup_viewer_gui()
        self._apply_current_frame_locked()
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.reader.close()
        self.server.stop()

    def info(self) -> dict[str, Any]:
        with self._lock:
            return {
                "first_tick": 0,
                "current_tick": int(self.current_tick),
                "total_ticks": int(self.total_tick),
                "tick_rate": int(self.tick_rate),
                "duration_ms": int(self.duration_ms),
                "speed": float(self.speed),
                "paused": bool(self.paused),
                "skip_idle": bool(self.skip_idle),
                "pause_at_tick": self.pause_at_tick,
                "viewer_url": self.viewer_url,
            }

    def seek(self, tick: int) -> None:
        with self._lock:
            self.current_tick = min(max(0, int(tick)), self.total_tick)
            self._skip_idle_gap_locked()
            self._last_wall_time = time.perf_counter()
            self._apply_current_frame_locked()
            self._sync_viewer_gui_locked()

    def play(self) -> None:
        with self._lock:
            self.paused = False
            self._last_wall_time = time.perf_counter()
            self._sync_viewer_gui_locked()

    def pause(self) -> None:
        with self._lock:
            self.paused = True
            self._sync_viewer_gui_locked()

    def set_speed(self, value: float) -> None:
        with self._lock:
            self.speed = max(0.05, float(value))
            self._sync_viewer_gui_locked()

    def play_range(self, start_tick: int, end_tick: int) -> None:
        with self._lock:
            self.current_tick = min(max(0, int(start_tick)), self.total_tick)
            self.pause_at_tick = min(max(self.current_tick, int(end_tick)), self.total_tick)
            self.paused = False
            self._skip_idle_gap_locked()
            self._apply_pause_bound_locked()
            self._last_wall_time = time.perf_counter()
            self._apply_current_frame_locked()
            self._sync_viewer_gui_locked()

    def set_skip_idle(self, enabled: bool) -> None:
        with self._lock:
            self.skip_idle = bool(enabled)
            self._skip_idle_gap_locked()
            self._last_wall_time = time.perf_counter()
            self._apply_current_frame_locked()
            self._sync_viewer_gui_locked()

    def _load_active_ranges(self) -> list[tuple[int, int]]:
        ticks = self.reader.h5["preview/tick"]
        if len(ticks) < 2:
            return []

        qpos = self.reader.h5["preview/qpos"][:]
        active = np.max(np.abs(np.diff(qpos, axis=0)), axis=1) > 1e-4
        active_frames = np.flatnonzero(active)
        if len(active_frames) == 0:
            return []

        pre_roll_ticks = max(1, round(2.0 * self.tick_rate))
        post_roll_ticks = max(1, round(1.0 * self.tick_rate))
        merge_gap_ticks = max(1, round(2.0 * self.tick_rate))
        ranges: list[tuple[int, int]] = []

        min_run_frames = 3
        start_frame = prev_frame = int(active_frames[0])
        frame_runs: list[tuple[int, int]] = []
        for frame in active_frames[1:]:
            frame = int(frame)
            if frame == prev_frame + 1:
                prev_frame = frame
                continue
            if prev_frame - start_frame + 1 >= min_run_frames:
                frame_runs.append((start_frame, prev_frame))
            start_frame = prev_frame = frame
        if prev_frame - start_frame + 1 >= min_run_frames:
            frame_runs.append((start_frame, prev_frame))

        for start_frame, end_frame in frame_runs:
            start = max(0, int(ticks[start_frame]) - pre_roll_ticks)
            end_tick_frame = min(end_frame + 1, len(ticks) - 1)
            end = min(self.total_tick, int(ticks[end_tick_frame]) + post_roll_ticks)
            if ranges and start <= ranges[-1][1] + merge_gap_ticks:
                ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
            else:
                ranges.append((start, end))

        return ranges

    def _setup_viewer_gui(self) -> None:
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

        with self.server.gui.add_folder("Scene", expand_by_default=False):
            self.scene.create_scene_gui()
        with self.server.gui.add_folder("Visualization", expand_by_default=False):
            self.scene.create_overlay_gui()
        with self.server.gui.add_folder("Groups", expand_by_default=False):
            self.scene.create_groups_gui()

        @self.play_button.on_click
        def _(_) -> None:
            with self._lock:
                self.paused = not self.paused
                self._last_wall_time = time.perf_counter()
                self._sync_viewer_gui_locked()

        @self.tick_slider.on_update
        def _(_) -> None:
            self.seek(int(self.tick_slider.value))

        @self.speed_buttons.on_click
        def _(event) -> None:
            self.set_speed(float(str(event.target.value).removesuffix("x")))

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                self._tick_locked()
            time.sleep(1.0 / 60.0)

    def _tick_locked(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_wall_time
        self._last_wall_time = now

        if self.paused or self.total_tick <= 0:
            self._sync_viewer_gui_locked()
            return

        self._skip_idle_gap_locked()

        if self.current_tick >= self.total_tick:
            if self.loop:
                self.current_tick = 0
            else:
                self.paused = True
        else:
            delta = max(1, round(elapsed * self.tick_rate * self.speed))
            self.current_tick = min(self.total_tick, self.current_tick + delta)

        self._skip_idle_gap_locked()
        self._apply_pause_bound_locked()

        self._apply_current_frame_locked()
        self._sync_viewer_gui_locked()

    def _apply_pause_bound_locked(self) -> None:
        if self.pause_at_tick is not None and self.current_tick >= self.pause_at_tick:
            self.current_tick = self.pause_at_tick
            self.pause_at_tick = None
            self.paused = True

    def _skip_idle_gap_locked(self) -> None:
        if (
            not self.skip_idle
            or self.current_tick >= self.total_tick
            or not self.active_ranges
            or self._is_active_tick_locked(self.current_tick)
        ):
            return
        next_tick = self._next_active_tick_after_locked(self.current_tick)
        self.current_tick = self.total_tick if next_tick is None else next_tick

    def _is_active_tick_locked(self, tick: int) -> bool:
        for start, end in self.active_ranges:
            if tick < start:
                return False
            if start <= tick <= end:
                return True
        return False

    def _next_active_tick_after_locked(self, tick: int) -> int | None:
        for start, end in self.active_ranges:
            if end <= tick:
                continue
            return start
        return None

    def _apply_current_frame_locked(self) -> None:
        frame = self.reader.preview_at_tick(self.current_tick)
        self.data.qpos[:] = frame["qpos"]
        self.data.qvel[:] = frame["qvel"]
        self.data.ctrl[:] = frame["ctrl"]
        self.data.time = float(frame["time"])
        mujoco.mj_forward(self.model, self.data)
        self.scene.update_from_mjdata(self.data)

    def _sync_viewer_gui_locked(self) -> None:
        self.play_button.label = "Play" if self.paused else "Pause"
        self.play_button.icon = viser.Icon.PLAYER_PLAY if self.paused else viser.Icon.PLAYER_PAUSE
        self.tick_slider.value = int(self.current_tick)
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


def create_app(adapter: MujocoReplayAdapter) -> FastAPI:
    app = FastAPI()

    def viewer_target(path: str = "") -> str:
        target = adapter.viewer_url.rstrip("/")
        if path:
            target = f"{target}/{path.lstrip('/')}"
        return target

    def proxy_viewer_http(path: str = "") -> Response:
        url = viewer_target(path)
        parsed = urlsplit(url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
        request_path = parsed.path or "/"
        if parsed.query:
            request_path = f"{request_path}?{parsed.query}"
        try:
            conn.request("GET", request_path, headers={"Accept-Encoding": "identity"})
            resp = conn.getresponse()
            body = resp.read()
            headers = {}
            for name, value in resp.getheaders():
                lower = name.lower()
                if lower in {"content-type", "cache-control", "etag", "last-modified"}:
                    headers[name] = value
            return Response(content=body, status_code=resp.status, headers=headers)
        finally:
            conn.close()

    async def proxy_viewer_websocket(websocket: WebSocket, path: str = "") -> None:
        suffix = f"/{path.lstrip('/')}" if path else ""
        upstream_url = adapter.viewer_url.replace("http://", "ws://", 1).replace(
            "https://", "wss://", 1
        ).rstrip("/") + suffix
        if websocket.url.query:
            upstream_url = f"{upstream_url}?{websocket.url.query}"
        requested_protocols = websocket.headers.get("sec-websocket-protocol", "")
        protocols = [item.strip() for item in requested_protocols.split(",") if item.strip()]
        selected_protocol = protocols[0] if protocols else None
        await websocket.accept(subprotocol=selected_protocol)
        try:
            async with websockets.connect(
                upstream_url,
                subprotocols=protocols or None,
                max_size=50 * 1024 * 1024,
                compression=None,
            ) as upstream:
                async def browser_to_upstream() -> None:
                    while True:
                        message = await websocket.receive()
                        msg_type = message.get("type")
                        if msg_type == "websocket.disconnect":
                            await upstream.close()
                            return
                        if "bytes" in message and message["bytes"] is not None:
                            await upstream.send(message["bytes"])
                        elif "text" in message and message["text"] is not None:
                            await upstream.send(message["text"])

                async def upstream_to_browser() -> None:
                    async for message in upstream:
                        if isinstance(message, bytes):
                            await websocket.send_bytes(message)
                        else:
                            await websocket.send_text(message)

                done, pending = await asyncio.wait(
                    {
                        asyncio.create_task(browser_to_upstream()),
                        asyncio.create_task(upstream_to_browser()),
                    },
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                for task in done:
                    task.result()
        except (WebSocketDisconnect, websockets.ConnectionClosed):
            return

    @app.get("/")
    def root() -> dict[str, Any]:
        return {"ok": True, "viewer_url": adapter.viewer_url}

    @app.get("/replay/view", response_class=HTMLResponse)
    def replay_view() -> str:
        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MuJoCo Replay View</title>
  <style>
    html, body, iframe {{
      width: 100%;
      height: 100%;
      margin: 0;
      border: 0;
      background: #05070f;
      overflow: hidden;
    }}
  </style>
</head>
<body>
  <iframe src="./viewer/" title="MuJoCo replay"></iframe>
</body>
</html>"""

    @app.get("/replay/viewer")
    def replay_viewer_root() -> Response:
        return proxy_viewer_http()

    @app.get("/replay/viewer/{path:path}")
    def replay_viewer_path(path: str, request: Request) -> Response:
        target_path = path
        if request.url.query:
            target_path = f"{target_path}?{request.url.query}"
        return proxy_viewer_http(target_path)

    @app.websocket("/replay/viewer")
    async def replay_viewer_ws_root(websocket: WebSocket) -> None:
        await proxy_viewer_websocket(websocket)

    @app.websocket("/replay/viewer/{path:path}")
    async def replay_viewer_ws_path(websocket: WebSocket, path: str) -> None:
        await proxy_viewer_websocket(websocket, path)

    @app.get("/replay/info")
    def replay_info() -> dict[str, Any]:
        return adapter.info()

    @app.post("/replay/seek")
    def replay_seek(tick: int) -> dict[str, bool]:
        adapter.seek(tick)
        return {"ok": True}

    @app.post("/replay/play")
    def replay_play() -> dict[str, bool]:
        adapter.play()
        return {"ok": True}

    @app.post("/replay/pause")
    def replay_pause() -> dict[str, bool]:
        adapter.pause()
        return {"ok": True}

    @app.post("/replay/speed")
    def replay_speed(value: float) -> dict[str, bool]:
        adapter.set_speed(value)
        return {"ok": True}

    @app.post("/replay/skip-idle")
    def replay_skip_idle(enabled: bool = False) -> dict[str, bool]:
        adapter.set_skip_idle(enabled)
        return {"ok": True}

    @app.post("/replay/play-range")
    def replay_play_range(start_tick: int, end_tick: int) -> dict[str, bool]:
        adapter.play_range(start_tick, end_tick)
        return {"ok": True}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Clawblox-compatible MuJoCo replay adapter.")
    parser.add_argument("recording", type=Path)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--viewer-port", type=int)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--playing", action="store_true")
    parser.add_argument("--no-loop", action="store_true")
    args = parser.parse_args()

    adapter = MujocoReplayAdapter(
        args.recording.resolve(),
        scene=args.scene.resolve() if args.scene else None,
        viewer_port=args.viewer_port,
        speed=args.speed,
        paused=not args.playing,
        loop=not args.no_loop,
    )

    previous_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum, frame):
        adapter.close()
        signal.signal(signal.SIGINT, previous_handler)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)
    try:
        uvicorn.run(create_app(adapter), host="127.0.0.1", port=args.port, log_level="warning")
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        adapter.close()


if __name__ == "__main__":
    main()
