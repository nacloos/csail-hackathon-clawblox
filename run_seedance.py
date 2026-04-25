"""Submit a seedance-2.0 image-to-video job (first + last frame) and download result.

Reads OPENROUTER_API_KEY from .env (or the environment).

Usage:
    uv run --with httpx python run_seedance.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE per line, '#' comments, no quoting magic."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


PROJ = Path(__file__).resolve().parent
OUT = PROJ / "videos" / "robo_styled.mp4"

FIRST_URL = "https://files.catbox.moe/ua78no.png"
LAST_URL = "https://files.catbox.moe/9lghkn.png"

PROMPT = (
    "A black robotic arm picking up a red apple resting on a weathered metal "
    "shipping crate, set in a dark rainy forest at night. Heavy rainfall, wet "
    "surfaces, mossy ground, distant fog between dark tree trunks, dim cool "
    "moonlight with subtle rim light on the arm. Photorealistic cinematic "
    "lighting, shallow depth of field. Preserve the exact motion, identical "
    "camera trajectory, identical object positions and timing of the source "
    "frames; only change surface appearance and environment. No changes to "
    "geometry, layout, or motion."
)

BODY = {
    "model": "bytedance/seedance-2.0",
    "prompt": PROMPT,
    "resolution": "480p",
    "duration": 4,
    "aspect_ratio": "1:1",
    "frame_images": [
        {
            "type": "image_url",
            "image_url": {"url": FIRST_URL},
            "frame_type": "first_frame",
        },
        {
            "type": "image_url",
            "image_url": {"url": LAST_URL},
            "frame_type": "last_frame",
        },
    ],
}


def main() -> None:
    _load_dotenv(PROJ / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("set OPENROUTER_API_KEY")

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    with httpx.Client(timeout=60.0) as client:
        print("submitting...", flush=True)
        r = client.post(
            "https://openrouter.ai/api/v1/videos", headers=headers, json=BODY
        )
        if r.status_code >= 400:
            print(f"submit failed: {r.status_code}\n{r.text}", flush=True)
            sys.exit(1)
        job = r.json()
        print(f"submitted: {job}", flush=True)
        job_id = job.get("id") or job.get("job_id") or job.get("data", {}).get("id")
        if not job_id:
            sys.exit(f"no job id in response: {job}")

        print(f"polling job {job_id}...", flush=True)
        urls: list[str] = []
        for attempt in range(180):  # up to ~15 min
            time.sleep(5)
            s = client.get(
                f"https://openrouter.ai/api/v1/videos/{job_id}", headers=headers
            )
            if s.status_code >= 400:
                print(f"poll {attempt}: {s.status_code} {s.text[:200]}", flush=True)
                continue
            data = s.json()
            status = data.get("status") or data.get("data", {}).get("status")
            print(f"poll {attempt}: status={status}", flush=True)
            if status in ("completed", "succeeded", "success", "finished"):
                urls = (
                    data.get("unsigned_urls")
                    or data.get("urls")
                    or data.get("data", {}).get("unsigned_urls")
                    or data.get("output", {}).get("urls")
                    or []
                )
                if not urls:
                    print(f"completed but no urls: {data}", flush=True)
                break
            if status in ("failed", "error", "cancelled"):
                sys.exit(f"job ended: {data}")

        if not urls:
            sys.exit("timed out without urls")

        video_url = urls[0]
        print(f"downloading {video_url}", flush=True)
        with client.stream("GET", video_url, headers=headers) as resp:
            resp.raise_for_status()
            OUT.parent.mkdir(parents=True, exist_ok=True)
            with OUT.open("wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
        print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
