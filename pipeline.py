"""Clip -> TwelveLabs segments -> GPT analysis -> image variations -> render.

Renderers:
- Lucy Edit (Decart) — default, requires DECART_API_KEY
- Runway Aleph (gen4_aleph) — opt-in via run metadata, requires RUNWAYML_API_SECRET

Usage (new run):
    uv run --with httpx --with runwayml python pipeline.py --input path/to/clip.mp4

Usage (resume an existing run):
    uv run --with httpx --with runwayml python pipeline.py --run-dir runs/20260425-140000_clip

Reads OPENROUTER_API_KEY, TWELVELABS_API_KEY, plus DECART_API_KEY or
RUNWAYML_API_SECRET (depending on the chosen render model) from .env (or env).
Per-run artifacts land in runs/<timestamp>_<basename>/.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


PROJ = Path(__file__).resolve().parent
RUNS_DIR = PROJ / "runs"

# Concurrency cap for image gen + Lucy Edit jobs.
PARALLEL = 4
TL_MIN_DURATION = 4.0  # TwelveLabs minimum input clip length, seconds.
N_PROMPTS = 8


class BlockedOnUser(SystemExit):
    """Raised (as clean exit) when an interactive choice is needed but no TTY."""

    def __init__(self, gate: str) -> None:
        super().__init__(0)
        self.gate = gate


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _block(gate: str) -> None:
    print(f"\n*** BLOCKED: {gate} (needs decision via web UI) ***", flush=True)
    raise BlockedOnUser(gate)


# --------------------------------------------------------------------- env

def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ------------------------------------------------------------------ ffmpeg

def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def ffprobe_dims(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x", str(path),
        ],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


def ffmpeg_cut(src: Path, start: float, end: float, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}",
            "-i", str(src),
            "-t", f"{max(0.0, end - start):.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-an", "-movflags", "+faststart",
            str(out),
        ],
        check=True,
    )


def ffmpeg_strip_audio(src: Path, out: Path) -> None:
    """Copy video stream, drop audio. Cheap (no re-encode)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-c:v", "copy", "-an", "-movflags", "+faststart",
            str(out),
        ],
        check=True,
    )


def ffmpeg_first_frame(src: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-vf", "select=eq(n\\,0)",
            "-frames:v", "1",
            str(out),
        ],
        check=True,
    )


def ffmpeg_concat(parts: list[Path], out: Path) -> None:
    """Lossless concat via the demuxer (parts must share codec params)."""
    if not parts:
        raise RuntimeError("ffmpeg_concat: empty parts list")
    if len(parts) == 1:
        # Just copy.
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(parts[0].read_bytes())
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    list_file = out.parent / f".{out.stem}_concat.txt"
    list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c", "copy",
                "-movflags", "+faststart",
                str(out),
            ],
            check=True,
        )
    finally:
        list_file.unlink(missing_ok=True)


# ----------------------------------------------------------- http helpers

class HTTPError(RuntimeError):
    pass


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: dict | None = None,
    json_body: Any = None,
    files: list | None = None,
    data: dict | None = None,
    max_retries: int = 5,
    label: str = "",
) -> httpx.Response:
    """HTTP call with exponential backoff on 429 / 5xx / connection errors."""
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            r = await client.request(
                method, url, headers=headers, json=json_body, files=files, data=data,
            )
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            if attempt == max_retries:
                raise HTTPError(f"{label}: connect/timeout {e}") from e
            print(f"  [retry] {label}: {type(e).__name__}, sleep {delay:.1f}s", flush=True)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 16.0)
            continue

        if r.status_code == 429 or 500 <= r.status_code < 600:
            if attempt == max_retries:
                raise HTTPError(f"{label}: {r.status_code} {r.text[:300]}")
            print(f"  [retry] {label}: {r.status_code}, sleep {delay:.1f}s", flush=True)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 16.0)
            continue
        return r
    raise HTTPError(f"{label}: exhausted retries")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


# ------------------------------------------------------------ TwelveLabs

TL_BASE = "https://api.twelvelabs.io/v1.3"


async def tl_upload_asset(client: httpx.AsyncClient, key: str, video_path: Path) -> dict:
    headers = {"x-api-key": key}
    with video_path.open("rb") as f:
        files = [("file", (video_path.name, f.read(), "video/mp4"))]
    data = {"method": "direct"}
    r = await request_with_retry(
        client, "POST", f"{TL_BASE}/assets",
        headers=headers, files=files, data=data, label="tl/assets",
    )
    if r.status_code >= 400:
        raise HTTPError(f"tl/assets: {r.status_code} {r.text[:500]}")
    return r.json()


async def tl_wait_asset_ready(client: httpx.AsyncClient, key: str, asset_id: str) -> dict:
    headers = {"x-api-key": key}
    for _ in range(180):  # ~15 min
        r = await request_with_retry(
            client, "GET", f"{TL_BASE}/assets/{asset_id}",
            headers=headers, label="tl/assets/get",
        )
        if r.status_code >= 400:
            raise HTTPError(f"tl/assets/get: {r.status_code} {r.text[:300]}")
        body = r.json()
        status = body.get("status") or body.get("data", {}).get("status")
        print(f"  tl asset {asset_id} status={status}", flush=True)
        if status == "ready":
            return body
        if status in ("failed", "error"):
            raise HTTPError(f"tl asset failed: {body}")
        await asyncio.sleep(5)
    raise HTTPError("tl asset readiness timeout")


async def tl_submit_segment(client: httpx.AsyncClient, key: str, asset_id: str) -> dict:
    headers = {"x-api-key": key, "Content-Type": "application/json"}
    body = {
        "video": {"type": "asset_id", "asset_id": asset_id},
        "model_name": "pegasus1.5",
        "analysis_mode": "time_based_metadata",
        "min_segment_duration": 2,
        "response_format": {
            "type": "segment_definitions",
            "segment_definitions": [
                {
                    "id": "scenes",
                    "description": (
                        "Segment the video into short clips of approximately 5 "
                        "seconds each. No clip may exceed 5 seconds. Split longer "
                        "scenes by sub-action, shot change, or camera movement. "
                        "Each clip should have a stable, identifiable subject "
                        "suitable for stylistic reskinning."
                    ),
                    "fields": [
                        {"name": "title", "type": "string",
                         "description": "Concise scene title."},
                        {"name": "description", "type": "string",
                         "description": "1-2 sentence description of what happens."},
                    ],
                }
            ],
        },
    }
    r = await request_with_retry(
        client, "POST", f"{TL_BASE}/analyze/tasks",
        headers=headers, json_body=body, label="tl/analyze",
    )
    if r.status_code >= 400:
        raise HTTPError(f"tl/analyze: {r.status_code} {r.text[:500]}")
    return r.json()


async def tl_wait_task_ready(client: httpx.AsyncClient, key: str, task_id: str) -> dict:
    headers = {"x-api-key": key}
    for _ in range(180):
        r = await request_with_retry(
            client, "GET", f"{TL_BASE}/analyze/tasks/{task_id}",
            headers=headers, label="tl/task/get",
        )
        if r.status_code >= 400:
            raise HTTPError(f"tl/task/get: {r.status_code} {r.text[:300]}")
        body = r.json()
        status = body.get("status") or body.get("data", {}).get("status")
        print(f"  tl task {task_id} status={status}", flush=True)
        if status == "ready":
            return body
        if status in ("failed", "error"):
            raise HTTPError(f"tl task failed: {body}")
        await asyncio.sleep(5)
    raise HTTPError("tl task readiness timeout")


SEG_TARGET_MIN = 2.5
SEG_TARGET_MAX = 5.0


def _split_long_segment(seg: dict) -> list[dict]:
    """Split a segment > SEG_TARGET_MAX into evenly-sized sub-windows in [4, 10]s."""
    dur = seg["end_time"] - seg["start_time"]
    if dur <= SEG_TARGET_MAX:
        return [seg]
    import math
    n = max(1, math.ceil(dur / SEG_TARGET_MAX))
    win = dur / n
    # Aim for sub-windows ≥ SEG_TARGET_MIN; if win drops below it, use fewer pieces.
    while win < SEG_TARGET_MIN and n > 1:
        n -= 1
        win = dur / n
    out = []
    base_title = seg.get("title", "")
    base_desc = seg.get("description", "")
    for i in range(n):
        s = seg["start_time"] + i * win
        e = seg["start_time"] + (i + 1) * win if i < n - 1 else seg["end_time"]
        title = f"{base_title} (part {i+1}/{n})" if base_title and n > 1 else base_title
        out.append({
            "start_time": s, "end_time": e,
            "title": title, "description": base_desc,
        })
    return out


def tl_parse_segments(task_response: dict) -> list[dict]:
    """Pull the {scenes:[...]} list out of task.result.data and enforce 4-10s."""
    result = task_response.get("result") or task_response.get("data", {}).get("result") or {}
    raw = result.get("data")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            raise HTTPError(f"tl: result.data not JSON: {raw[:300]}")
    elif isinstance(raw, dict):
        parsed = raw
    else:
        raise HTTPError(f"tl: result.data missing or unexpected type: {type(raw).__name__}")
    scenes = parsed.get("scenes") or []
    out = []
    for s in scenes:
        meta = s.get("metadata") or {}
        seg = {
            "start_time": float(s["start_time"]),
            "end_time": float(s["end_time"]),
            "title": meta.get("title", ""),
            "description": meta.get("description", ""),
        }
        # Drop sub-minimum scenes; chop over-maximum scenes into 4-10s sub-windows.
        dur = seg["end_time"] - seg["start_time"]
        if dur < SEG_TARGET_MIN:
            continue
        out.extend(_split_long_segment(seg))
    return out


# -------------------------------------------------------- OpenRouter (text+image)

OR_BASE = "https://openrouter.ai/api/v1"
OR_HEADERS = lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def b64_data_url(path: Path, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


ANALYSIS_PROMPT = """You are shown the FIRST FRAME of a short robotics-data video clip. The full clip will be re-rendered as 8 stylistic variations using a video-to-video model that takes one reference image (a re-skinned version of this exact frame) and preserves motion/geometry.

This is robotics training data. The whole point of the variations is to teach a downstream model that surface appearance is a nuisance variable. So we keep the *physical* scene constant and **boldly** change everything about how it looks.

INVARIANTS (must NOT change across variations):
- Geometry of objects (shapes, sizes, proportions)
- Object identity and count (a mug stays a mug; one mug stays one mug)
- Trajectory and motion (handled by the video model, but don't suggest changes that imply different motion)
- Spatial relationships (object A on top of object B, robot arm reaching toward target)
- Camera framing (what's in frame and where)

ALWAYS-VARY (these are NEVER invariants — even if they appear visually striking in the source frame, they are the *point* of the variations and must be different in each prompt):
- Floor / ground surface (wood plank, polished concrete, dirt, grass, tile, metal grating, sand, carpet, vinyl, marble, gravel, rubber mat, etc. — *do not preserve a checkered or patterned floor; replace it*).
- Walls / background environment (lab, warehouse, kitchen, outdoors, industrial, abstract void, jungle floor, neon-lit alley, etc.)
- Surface textures and materials of any non-target objects in the scene
- Lighting (time of day, harsh shadows vs. diffuse, colored vs. neutral, single-source vs. ambient)
- Color palette

Specifically: do NOT include the floor pattern, wall color, or lighting from the source frame in your `invariants` list. Those belong in `variation_axes`.

Return ONLY valid JSON, no commentary, with this exact shape:
{
  "invariants": ["string", ...],          // ONLY geometry, object identity, motion, spatial relations, framing
  "variation_axes": [                     // axes the 8 prompts span — must include floor and lighting
    {"name": "floor", "options": ["...", "..."]},
    {"name": "environment", "options": ["...", "..."]},
    {"name": "lighting", "options": ["...", "..."]},
    {"name": "color_palette", "options": ["...", "..."]},
    {"name": "texture", "options": ["...", "..."]}
  ],
  "prompts": ["...", "...", "...", "...", "...", "...", "...", "..."]  // exactly 8 prompts
}

Each prompt should:
- Start by re-stating the invariants in one short clause (geometry / objects / motion / framing only).
- Name a *concrete different floor*, environment, lighting setup, and palette for THAT variation. Do not reuse the source frame's floor / walls / lighting.
- Be 1-3 sentences, concrete and visual. No bullet points inside the prompt.
- Across the 8 prompts, the floors must all be different. The environments must all be different.
"""


async def or_analyze_first_frame(
    client: httpx.AsyncClient, key: str, frame_path: Path,
) -> dict:
    body = {
        "model": "openai/gpt-5.4-nano",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": ANALYSIS_PROMPT},
                {"type": "image_url", "image_url": {"url": b64_data_url(frame_path)}},
            ]},
        ],
    }
    r = await request_with_retry(
        client, "POST", f"{OR_BASE}/chat/completions",
        headers=OR_HEADERS(key), json_body=body, label="or/analyze",
    )
    if r.status_code >= 400:
        raise HTTPError(f"or/analyze: {r.status_code} {r.text[:500]}")
    return r.json()


def parse_analysis(response: dict) -> dict:
    content = response["choices"][0]["message"]["content"]
    # Strip markdown fences if present.
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)
    prompts = parsed.get("prompts") or []
    if len(prompts) != N_PROMPTS:
        raise HTTPError(f"analysis returned {len(prompts)} prompts, expected {N_PROMPTS}")
    return parsed


def parse_prompt_list(content: str, expected: int) -> list[str]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)
    prompts = parsed.get("prompts") or []
    if len(prompts) != expected:
        raise HTTPError(f"regen returned {len(prompts)} prompts, expected {expected}")
    return prompts


async def or_regen_prompts(
    client: httpx.AsyncClient, key: str, frame_path: Path,
    current: list[str], axes: list[dict], indices: list[int],
) -> list[str]:
    """Replace prompts at the given 1-based indices with fresh distinct variants."""
    n = len(current)
    idx_str = ", ".join(str(i) for i in indices)
    listed = "\n".join(f"  [{i+1}] {p}" for i, p in enumerate(current))
    instr = (
        "You previously produced these prompts for stylistic variations of the "
        f"attached image (composition unchanged across all):\n{listed}\n\n"
        f"Replace ONLY the prompts at indices [{idx_str}] with new, distinct variants "
        "along the same axes (lighting, environment, texture, color_palette). Keep "
        "the others byte-for-byte identical. Output strictly valid JSON of shape "
        f'{{"prompts": [string × {n}]}}, preserving order.'
    )
    if axes:
        instr += f"\n\nAxes context: {json.dumps(axes)[:1500]}"
    body = {
        "model": "openai/gpt-5.4-nano",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": instr},
                {"type": "image_url", "image_url": {"url": b64_data_url(frame_path)}},
            ]},
        ],
    }
    r = await request_with_retry(
        client, "POST", f"{OR_BASE}/chat/completions",
        headers=OR_HEADERS(key), json_body=body, label="or/regen",
    )
    if r.status_code >= 400:
        raise HTTPError(f"or/regen: {r.status_code} {r.text[:500]}")
    return parse_prompt_list(r.json()["choices"][0]["message"]["content"], n)


async def or_image_variation(
    client: httpx.AsyncClient, key: str, frame_path: Path, prompt: str, out_path: Path,
    response_path: Path,
) -> None:
    body = {
        "model": "openai/gpt-5.4-image-2",
        "modalities": ["image", "text"],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": b64_data_url(frame_path)}},
            ]},
        ],
    }
    r = await request_with_retry(
        client, "POST", f"{OR_BASE}/chat/completions",
        headers=OR_HEADERS(key), json_body=body, label=f"or/image:{out_path.name}",
    )
    if r.status_code >= 400:
        raise HTTPError(f"or/image: {r.status_code} {r.text[:500]}")
    body_json = r.json()
    write_json(response_path, body_json)
    images = body_json["choices"][0]["message"].get("images") or []
    if not images:
        raise HTTPError(f"or/image: no images in response: {body_json}")
    url = images[0]["image_url"]["url"]
    if not url.startswith("data:"):
        raise HTTPError(f"or/image: expected data URL, got {url[:60]}")
    payload = url.split(",", 1)[1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(payload))


# ------------------------------------------------------------------- Decart

DECART_BASE = "https://api.decart.ai/v1"


async def decart_submit(
    client: httpx.AsyncClient, key: str, video_path: Path, ref_image_path: Path,
) -> dict:
    headers = {"X-API-KEY": key}
    files = [
        ("data", (video_path.name, video_path.read_bytes(), "video/mp4")),
        ("reference_image", (ref_image_path.name, ref_image_path.read_bytes(), "image/png")),
    ]
    data = {"prompt": "", "enhance_prompt": "false", "resolution": "720p"}
    r = await request_with_retry(
        client, "POST", f"{DECART_BASE}/jobs/lucy-2.1",
        headers=headers, files=files, data=data, label=f"decart/submit:{ref_image_path.name}",
    )
    if r.status_code >= 400:
        raise HTTPError(f"decart/submit: {r.status_code} {r.text[:500]}")
    return r.json()


async def decart_wait(client: httpx.AsyncClient, key: str, job_id: str) -> dict:
    headers = {"X-API-KEY": key}
    for _ in range(360):  # ~30 min
        r = await request_with_retry(
            client, "GET", f"{DECART_BASE}/jobs/{job_id}",
            headers=headers, label=f"decart/poll:{job_id[:8]}",
        )
        if r.status_code >= 400:
            raise HTTPError(f"decart/poll: {r.status_code} {r.text[:300]}")
        body = r.json()
        status = body.get("status")
        if status in ("succeeded", "completed", "success"):
            return body
        if status in ("failed", "error", "cancelled"):
            raise HTTPError(f"decart job {job_id} failed: {body}")
        print(f"  decart {job_id[:8]} status={status}", flush=True)
        await asyncio.sleep(5)
    raise HTTPError(f"decart timeout job={job_id}")


async def decart_download(
    client: httpx.AsyncClient, key: str, job_id: str, out_path: Path,
) -> None:
    headers = {"X-API-KEY": key}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    async with client.stream(
        "GET", f"{DECART_BASE}/jobs/{job_id}/content", headers=headers,
    ) as resp:
        resp.raise_for_status()
        with out_path.open("wb") as f:
            async for chunk in resp.aiter_bytes():
                f.write(chunk)


# ----------------------------------------------------- Runway Aleph (gen4_aleph)
# Aleph honors the input video's duration automatically — no `duration` param.
# We pick a `ratio` from a fixed allowlist closest to the source aspect.

RUNWAY_ALEPH_RATIOS = [
    (1280, 720), (720, 1280), (1104, 832), (960, 960),
    (832, 1104), (1584, 672), (848, 480), (640, 480),
]


def runway_pick_ratio(src_w: int, src_h: int) -> str:
    target = src_w / src_h if src_h else 16 / 9
    best = min(RUNWAY_ALEPH_RATIOS, key=lambda wh: abs(wh[0] / wh[1] - target))
    return f"{best[0]}:{best[1]}"


async def runway_upload(file_path: Path) -> str:
    """Upload a local file via the SDK; return the ephemeral URI Runway will fetch."""
    from runwayml import AsyncRunwayML
    client = AsyncRunwayML()
    upload = await client.uploads.create_ephemeral(file=file_path)
    return upload.uri


async def runway_submit(
    video_uri: str, image_uri: str, prompt_text: str, ratio: str,
) -> str:
    from runwayml import AsyncRunwayML
    client = AsyncRunwayML()
    task = await client.video_to_video.create(
        model="gen4_aleph",
        video_uri=video_uri,
        prompt_text=prompt_text or "stylistic reskin preserving motion and geometry",
        references=[{"type": "image", "uri": image_uri}],
        ratio=ratio,
    )
    return task.id


async def runway_wait(task_id: str) -> list[str]:
    """Poll until SUCCEEDED; return output URLs. Raises HTTPError on failure."""
    from runwayml import AsyncRunwayML
    client = AsyncRunwayML()
    deadline = time.monotonic() + 30 * 60  # 30 min cap
    while True:
        task = await client.tasks.retrieve(task_id)
        status = getattr(task, "status", None)
        print(f"  runway {task_id[:8]} status={status}", flush=True)
        if status == "SUCCEEDED":
            outputs = list(getattr(task, "output", []) or [])
            if not outputs:
                raise HTTPError(f"runway {task_id} succeeded but no output URLs")
            return outputs
        if status in ("FAILED", "CANCELLED"):
            raise HTTPError(
                f"runway {task_id} {status}: "
                f"{getattr(task, 'failure', None) or getattr(task, 'failure_code', '')}"
            )
        if time.monotonic() > deadline:
            raise HTTPError(f"runway timeout task={task_id}")
        await asyncio.sleep(5)


async def runway_download(
    client: httpx.AsyncClient, url: str, out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    async with client.stream("GET", url) as resp:
        resp.raise_for_status()
        with out_path.open("wb") as f:
            async for chunk in resp.aiter_bytes():
                f.write(chunk)


# ------------------------------------------------------------------ stages

def stage_input(input_path: Path, run_dir: Path) -> Path:
    """Copy source clip into the run dir, capture metadata."""
    in_dir = run_dir / "00_input"
    src_dst = in_dir / "source.mp4"
    meta_path = in_dir / "metadata.json"
    if src_dst.exists() and meta_path.exists():
        print(f"[skip] 00_input already present: {src_dst}", flush=True)
        return src_dst
    in_dir.mkdir(parents=True, exist_ok=True)
    # Strip any audio track at the door so no downstream API charges for it.
    ffmpeg_strip_audio(input_path, src_dst)
    duration = ffprobe_duration(src_dst)
    if duration < TL_MIN_DURATION:
        raise SystemExit(
            f"clip too short for TwelveLabs: {duration:.2f}s < {TL_MIN_DURATION}s minimum"
        )
    w, h = ffprobe_dims(src_dst)
    write_json(meta_path, {
        "source_path": str(input_path),
        "duration": duration, "width": w, "height": h,
    })
    print(f"[done] 00_input: duration={duration:.2f}s {w}x{h}", flush=True)
    return src_dst


async def stage_twelvelabs(
    client: httpx.AsyncClient, run_dir: Path, src: Path, key: str,
) -> list[dict]:
    tl_dir = run_dir / "01_twelvelabs"
    asset_path = tl_dir / "asset.json"
    task_path = tl_dir / "task.json"
    seg_path = tl_dir / "segments.json"
    if seg_path.exists():
        print("[skip] 01_twelvelabs: segments.json present", flush=True)
        return read_json(seg_path)

    tl_dir.mkdir(parents=True, exist_ok=True)
    if asset_path.exists():
        asset_resp = read_json(asset_path)
    else:
        print("[run]  01_twelvelabs: uploading asset...", flush=True)
        asset_resp = await tl_upload_asset(client, key, src)
        write_json(asset_path, asset_resp)
    asset_id = (
        asset_resp.get("_id")
        or asset_resp.get("id")
        or asset_resp.get("asset_id")
        or asset_resp.get("data", {}).get("id")
    )
    if not asset_id:
        raise HTTPError(f"tl: no asset id in {asset_resp}")

    if asset_resp.get("status") != "ready":
        await tl_wait_asset_ready(client, key, asset_id)

    if task_path.exists():
        task_resp = read_json(task_path)
    else:
        print("[run]  01_twelvelabs: submitting segmentation task...", flush=True)
        submit = await tl_submit_segment(client, key, asset_id)
        task_id = (
            submit.get("_id")
            or submit.get("id")
            or submit.get("task_id")
            or submit.get("data", {}).get("id")
        )
        if not task_id:
            raise HTTPError(f"tl: no task id in {submit}")
        task_resp = await tl_wait_task_ready(client, key, task_id)
        write_json(task_path, task_resp)

    segs = tl_parse_segments(task_resp)
    if not segs:
        raise HTTPError("tl: zero segments returned")
    write_json(seg_path, segs)
    print(f"[done] 01_twelvelabs: {len(segs)} segments", flush=True)
    return segs


def stage_pick_segment(run_dir: Path, segments: list[dict]) -> dict:
    chosen_path = run_dir / "02_segment" / "chosen.json"
    if chosen_path.exists():
        chosen = read_json(chosen_path)
        print(f"[skip] 02_segment: using existing pick "
              f"({chosen['start_time']:.2f}-{chosen['end_time']:.2f}s)", flush=True)
        return chosen

    if not _interactive():
        _block("pick-segment")
    print("\nSegments:")
    for i, s in enumerate(segments, start=1):
        dur = s["end_time"] - s["start_time"]
        title = s["title"] or "(untitled)"
        print(f"  [{i}] {s['start_time']:6.2f}-{s['end_time']:6.2f}s  "
              f"({dur:5.2f}s)  {title}")
        if s["description"]:
            print(f"        {s['description']}")
    while True:
        raw = input(f"\nPick segment [1-{len(segments)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(segments):
            chosen = segments[int(raw) - 1]
            break
        print("invalid choice")
    write_json(chosen_path, chosen)
    return chosen


def stage_cut_and_frame(run_dir: Path, src: Path, chosen: dict) -> tuple[Path, Path]:
    seg_dir = run_dir / "02_segment"
    seg_mp4 = seg_dir / "segment.mp4"
    first_png = seg_dir / "first_frame.png"
    if seg_mp4.exists() and first_png.exists():
        print(f"[skip] 02_segment: cut + frame already present", flush=True)
        return seg_mp4, first_png
    print(f"[run]  02_segment: cutting [{chosen['start_time']:.2f}-{chosen['end_time']:.2f}]s",
          flush=True)
    ffmpeg_cut(src, chosen["start_time"], chosen["end_time"], seg_mp4)
    ffmpeg_first_frame(seg_mp4, first_png)
    print(f"[done] 02_segment: {seg_mp4.name}, {first_png.name}", flush=True)
    return seg_mp4, first_png


async def stage_analyze(
    client: httpx.AsyncClient, run_dir: Path, frame: Path, key: str,
) -> dict:
    a_dir = run_dir / "03_analysis"
    raw_path = a_dir / "response.json"
    parsed_path = a_dir / "prompts.json"
    if parsed_path.exists():
        print("[skip] 03_analysis: prompts.json present", flush=True)
        return read_json(parsed_path)
    print("[run]  03_analysis: GPT-5.4-nano analyzing first frame...", flush=True)
    resp = await or_analyze_first_frame(client, key, frame)
    write_json(raw_path, resp)
    parsed = parse_analysis(resp)
    write_json(parsed_path, parsed)
    print(f"[done] 03_analysis: {len(parsed['prompts'])} prompts, "
          f"{len(parsed.get('variation_axes', []))} axes", flush=True)
    return parsed


def stage_choose_count(run_dir: Path) -> int:
    count_path = run_dir / "04_variations" / ".count"
    if count_path.exists():
        n = int(count_path.read_text().strip())
        print(f"[skip] count selection: using existing {n}", flush=True)
        return n
    if not _interactive():
        _block("choose-count")
    while True:
        raw = input("\nGenerate how many variations? [1/3/8]: ").strip()
        if raw in ("1", "3", "8"):
            n = int(raw)
            count_path.parent.mkdir(parents=True, exist_ok=True)
            count_path.write_text(str(n))
            return n
        print("must be 1, 3, or 8")


async def stage_approve_prompts(
    client: httpx.AsyncClient, run_dir: Path, frame: Path,
    analysis: dict, count: int, key: str,
) -> list[str]:
    """Show prompts, allow regen / edit / approve. Returns the approved list."""
    var_dir = run_dir / "04_variations"
    approved_path = var_dir / "approved_prompts.json"
    if approved_path.exists():
        approved = read_json(approved_path)
        if isinstance(approved, list) and len(approved) == count:
            print(f"[skip] approval: {count} approved prompts on disk", flush=True)
            return approved

    var_dir.mkdir(parents=True, exist_ok=True)
    if not _interactive():
        _block("approve-prompts")

    prompts = list(analysis["prompts"][:count])
    axes = analysis.get("variation_axes") or []

    def render() -> None:
        print("\n=== prompts up for approval ===")
        for i, p in enumerate(prompts, start=1):
            wrapped = p.replace("\n", " ")
            print(f"\n[{i}] {wrapped}")
        print(
            "\nCommands:\n"
            "  enter / 'ok'   approve all and continue\n"
            "  'r N [N ...]'  regenerate prompts at those indices via nano\n"
            "  'e N'          edit prompt N in $EDITOR (or type a replacement line)\n"
            "  'show'         re-display\n"
        )

    render()
    while True:
        try:
            raw = input("> ").strip()
        except EOFError:
            raw = "ok"
        if raw == "" or raw.lower() in ("ok", "approve", "y", "yes"):
            break
        if raw == "show":
            render()
            continue
        parts = raw.split()
        cmd = parts[0].lower()

        if cmd == "r" and len(parts) >= 2:
            try:
                idxs = sorted({int(x) for x in parts[1:]})
            except ValueError:
                print("indices must be integers"); continue
            if not all(1 <= i <= count for i in idxs):
                print(f"indices must be in 1..{count}"); continue
            print(f"regenerating: {idxs}", flush=True)
            try:
                prompts = await or_regen_prompts(client, key, frame, prompts, axes, idxs)
            except Exception as e:
                print(f"regen failed: {e}"); continue
            render()
            continue

        if cmd == "e" and len(parts) == 2:
            try:
                i = int(parts[1])
            except ValueError:
                print("usage: e N"); continue
            if not 1 <= i <= count:
                print(f"index must be in 1..{count}"); continue
            new_text = _edit_text(prompts[i - 1])
            if new_text is None or not new_text.strip():
                print("(unchanged)"); continue
            prompts[i - 1] = new_text.strip()
            render()
            continue

        print("unrecognized command (try 'ok', 'r 1 3', 'e 2', 'show')")

    write_json(approved_path, prompts)
    return prompts


def _edit_text(initial: str) -> str | None:
    """Open $EDITOR on the prompt text, or fall back to single-line input."""
    editor = os.environ.get("EDITOR")
    if not editor:
        print("$EDITOR not set; type a single-line replacement (blank to keep):")
        try:
            line = input("  ").rstrip("\n")
        except EOFError:
            return None
        return line if line else None
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(initial)
        tmp = f.name
    try:
        subprocess.run([editor, tmp], check=False)
        return Path(tmp).read_text()
    finally:
        try: os.unlink(tmp)
        except OSError: pass


async def stage_image_variations(
    client: httpx.AsyncClient, run_dir: Path, frame: Path,
    prompts: list[str], count: int, key: str,
) -> list[Path]:
    var_dir = run_dir / "04_variations"
    var_dir.mkdir(parents=True, exist_ok=True)
    chosen_prompts = prompts[:count]
    sem = asyncio.Semaphore(PARALLEL)

    async def one(idx: int, prompt: str) -> Path:
        sub = var_dir / f"var_{idx:02d}"
        sub.mkdir(parents=True, exist_ok=True)
        prompt_path = sub / "prompt.txt"
        img_path = sub / "image.png"
        resp_path = sub / "response.json"
        if img_path.exists() and resp_path.exists():
            print(f"  [skip] var_{idx:02d}: image present", flush=True)
            return img_path
        prompt_path.write_text(prompt)
        async with sem:
            print(f"  [run]  var_{idx:02d}: image gen", flush=True)
            await or_image_variation(client, key, frame, prompt, img_path, resp_path)
        print(f"  [done] var_{idx:02d}: {img_path.name}", flush=True)
        return img_path

    print(f"[run]  04_variations: {count} image(s), parallel={PARALLEL}", flush=True)
    paths = await asyncio.gather(
        *(one(i + 1, p) for i, p in enumerate(chosen_prompts)),
        return_exceptions=False,
    )
    print(f"[done] 04_variations: {count} image(s)", flush=True)
    return list(paths)


async def stage_lucy_edit(
    client: httpx.AsyncClient, run_dir: Path, segment_mp4: Path,
    image_paths: list[Path], key: str,
) -> list[Path]:
    le_dir = run_dir / "05_lucy_edit"
    le_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(PARALLEL)

    async def one(idx: int, image_path: Path) -> Path:
        sub = le_dir / f"var_{idx:02d}"
        sub.mkdir(parents=True, exist_ok=True)
        submit_path = sub / "submit.json"
        poll_path = sub / "poll.json"
        out_mp4 = sub / "output.mp4"
        if out_mp4.exists():
            print(f"  [skip] lucy var_{idx:02d}: output present", flush=True)
            return out_mp4
        async with sem:
            if submit_path.exists():
                submit = read_json(submit_path)
            else:
                print(f"  [run]  lucy var_{idx:02d}: submit", flush=True)
                submit = await decart_submit(client, key, segment_mp4, image_path)
                write_json(submit_path, submit)
            job_id = submit.get("id") or submit.get("job_id")
            if not job_id:
                raise HTTPError(f"decart: no job id in {submit}")
            print(f"  [run]  lucy var_{idx:02d}: poll {job_id[:8]}...", flush=True)
            poll = await decart_wait(client, key, job_id)
            write_json(poll_path, poll)
            print(f"  [run]  lucy var_{idx:02d}: download", flush=True)
            await decart_download(client, key, job_id, out_mp4)
        print(f"  [done] lucy var_{idx:02d}: {out_mp4.name}", flush=True)
        return out_mp4

    print(f"[run]  05_lucy_edit: {len(image_paths)} job(s), parallel={PARALLEL}", flush=True)
    outs = await asyncio.gather(
        *(one(i + 1, p) for i, p in enumerate(image_paths)),
        return_exceptions=False,
    )
    print(f"[done] 05_lucy_edit: {len(outs)} output video(s)", flush=True)
    return list(outs)


# Aleph caps each generation at 5 seconds. We rely on the upstream segment
# pipeline (SEG_TARGET_MAX) to keep segments at ≤5s, so a single Aleph call
# covers the whole clip 1:1. The guard below fails loudly if that contract
# is ever violated rather than silently truncating.
RUNWAY_ALEPH_MAX_S = 5.0


async def stage_runway_aleph(
    client: httpx.AsyncClient, run_dir: Path, segment_mp4: Path,
    image_paths: list[Path], prompts: list[str], src_w: int, src_h: int,
) -> list[Path]:
    seg_dur = ffprobe_duration(segment_mp4)
    if seg_dur > RUNWAY_ALEPH_MAX_S + 0.1:
        raise HTTPError(
            f"runway: segment is {seg_dur:.2f}s but Aleph caps at "
            f"{RUNWAY_ALEPH_MAX_S}s. Tighten SEG_TARGET_MAX in the "
            f"segment pipeline."
        )

    rw_dir = run_dir / "05_runway"
    rw_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(PARALLEL)
    ratio = runway_pick_ratio(src_w, src_h)
    print(f"[run]  05_runway: ratio={ratio}, {len(image_paths)} job(s), "
          f"parallel={PARALLEL}", flush=True)

    # Upload the segment once — same input across all variants.
    seg_uri_path = rw_dir / "_segment_uri.json"
    if seg_uri_path.exists():
        segment_uri = read_json(seg_uri_path)["uri"]
        print(f"  [skip] segment upload: cached", flush=True)
    else:
        print(f"  [run]  uploading segment.mp4...", flush=True)
        segment_uri = await runway_upload(segment_mp4)
        write_json(seg_uri_path, {"uri": segment_uri})

    async def one(idx: int, image_path: Path, prompt_text: str) -> Path:
        sub = rw_dir / f"var_{idx:02d}"
        sub.mkdir(parents=True, exist_ok=True)
        submit_path = sub / "submit.json"
        poll_path = sub / "poll.json"
        out_mp4 = sub / "output.mp4"
        if out_mp4.exists():
            print(f"  [skip] runway var_{idx:02d}: output present", flush=True)
            return out_mp4
        async with sem:
            if submit_path.exists():
                submit = read_json(submit_path)
                task_id = submit.get("task_id")
            else:
                print(f"  [run]  runway var_{idx:02d}: upload reference", flush=True)
                image_uri = await runway_upload(image_path)
                print(f"  [run]  runway var_{idx:02d}: submit", flush=True)
                task_id = await runway_submit(segment_uri, image_uri, prompt_text, ratio)
                write_json(submit_path, {"task_id": task_id, "image_uri": image_uri})
            print(f"  [run]  runway var_{idx:02d}: poll {task_id[:8]}...", flush=True)
            outputs = await runway_wait(task_id)
            write_json(poll_path, {"output_urls": outputs})
            print(f"  [run]  runway var_{idx:02d}: download", flush=True)
            await runway_download(client, outputs[0], out_mp4)
        print(f"  [done] runway var_{idx:02d}: {out_mp4.name}", flush=True)
        return out_mp4

    outs = await asyncio.gather(
        *(one(i + 1, p, pr) for i, (p, pr) in enumerate(zip(image_paths, prompts))),
        return_exceptions=False,
    )
    print(f"[done] 05_runway: {len(outs)} output video(s)", flush=True)
    return list(outs)


# ------------------------------------------------------------------- main

def make_run_dir(input_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^\w.-]", "_", input_path.stem)[:40]
    d = RUNS_DIR / f"{ts}_{safe}"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def main_async(args: argparse.Namespace) -> None:
    load_dotenv(PROJ / ".env")
    or_key = os.environ.get("OPENROUTER_API_KEY")
    tl_key = os.environ.get("TWELVELABS_API_KEY")
    missing_core = [n for n, v in [
        ("OPENROUTER_API_KEY", or_key),
        ("TWELVELABS_API_KEY", tl_key),
    ] if not v]
    if missing_core:
        raise SystemExit(f"missing keys in .env: {', '.join(missing_core)}")

    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        if not run_dir.exists():
            raise SystemExit(f"run dir does not exist: {run_dir}")
        meta_path = run_dir / "00_input" / "metadata.json"
        if not meta_path.exists():
            raise SystemExit(f"run dir missing 00_input/metadata.json: {run_dir}")
        src = run_dir / "00_input" / "source.mp4"
    elif args.input:
        input_path = Path(args.input).resolve()
        if not input_path.exists():
            raise SystemExit(f"input not found: {input_path}")
        run_dir = make_run_dir(input_path)
        src = stage_input(input_path, run_dir)
    else:
        raise SystemExit("provide --input <clip> or --run-dir <existing>")

    # Render model: lucy (default, Decart) or runway (gen4_aleph). Stored in
    # 00_input/metadata.json by the webapp at run creation; defaults to lucy
    # for older runs / direct CLI invocations.
    meta = read_json(run_dir / "00_input" / "metadata.json") or {}
    model = (meta.get("model") or "lucy").lower()
    if model not in ("lucy", "runway"):
        raise SystemExit(f"unknown model in metadata: {model}")
    if model == "lucy":
        dc_key = os.environ.get("DECART_API_KEY")
        if not dc_key:
            raise SystemExit("missing DECART_API_KEY (required for model=lucy)")
    else:
        if not os.environ.get("RUNWAYML_API_SECRET"):
            raise SystemExit("missing RUNWAYML_API_SECRET (required for model=runway)")

    print(f"\n=== run dir: {run_dir}  ·  model={model} ===\n", flush=True)

    timeout = httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        segments = await stage_twelvelabs(client, run_dir, src, tl_key)
        chosen = stage_pick_segment(run_dir, segments)
        seg_mp4, first_png = stage_cut_and_frame(run_dir, src, chosen)
        analysis = await stage_analyze(client, run_dir, first_png, or_key)
        count = stage_choose_count(run_dir)
        approved = await stage_approve_prompts(
            client, run_dir, first_png, analysis, count, or_key,
        )
        image_paths = await stage_image_variations(
            client, run_dir, first_png, approved, count, or_key,
        )
        if model == "lucy":
            outputs = await stage_lucy_edit(
                client, run_dir, seg_mp4, image_paths, dc_key,
            )
            out_dir = run_dir / "05_lucy_edit"
        else:
            outputs = await stage_runway_aleph(
                client, run_dir, seg_mp4, image_paths, approved,
                int(meta.get("width") or 0), int(meta.get("height") or 0),
            )
            out_dir = run_dir / "05_runway"

    print(f"\n=== done. {len(outputs)} output video(s) under {out_dir} ===")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="path to source clip")
    ap.add_argument("--run-dir", help="resume an existing runs/<dir>")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
