"""Web UI for the clip -> segment -> variations -> Lucy Edit pipeline.

Run:
    uv run --with fastapi --with 'uvicorn[standard]' --with python-multipart \
        --with httpx python webapp.py

Then open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles


PROJ = Path(__file__).resolve().parent
RUNS_DIR = PROJ / "runs"
STATIC = PROJ / "static"
PIPELINE = PROJ / "pipeline.py"
AGENT_LAUNCHER = PROJ / "pipeline_agent.sh"


# --------------------------------------------------------------- subprocess mgr

class JobManager:
    """One pipeline subprocess per run, with log capture."""

    def __init__(self) -> None:
        self.procs: dict[str, subprocess.Popen] = {}

    def is_running(self, name: str) -> bool:
        p = self.procs.get(name)
        return p is not None and p.poll() is None

    def kick(self, run_dir: Path) -> None:
        name = run_dir.name
        if self.is_running(name):
            return
        log = run_dir / "pipeline.log"
        log_f = log.open("ab")
        log_f.write(f"\n--- spawn {datetime.now().isoformat()} ---\n".encode())
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # uv ensures httpx + runwayml are available regardless of which model
        # the run uses (runwayml is only imported when model=runway, but
        # bundling it here keeps the spawn cost predictable).
        cmd = [
            "uv", "run", "--with", "httpx", "--with", "runwayml",
            "python", str(PIPELINE),
            "--run-dir", str(run_dir),
        ]
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,  # forces non-TTY -> pipeline blocks gracefully
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(PROJ),
            env=env,
        )
        self.procs[name] = p

    def stop(self, name: str) -> None:
        p = self.procs.get(name)
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


class AgentManager:
    """One Claude Code agent per run, polling the webapp until done."""

    def __init__(self) -> None:
        self.procs: dict[str, subprocess.Popen] = {}

    def is_running(self, name: str) -> bool:
        p = self.procs.get(name)
        return p is not None and p.poll() is None

    def kick(self, run_dir: Path, webapp_url: str) -> None:
        name = run_dir.name
        if self.is_running(name):
            return
        if not AGENT_LAUNCHER.exists():
            raise RuntimeError(f"agent launcher missing: {AGENT_LAUNCHER}")
        env = os.environ.copy()
        env["WEBAPP_URL"] = webapp_url
        # Mark this run as agent-driven so the UI hides gate prompts.
        (run_dir / "agent_mode.flag").write_text(datetime.now().isoformat())
        # The launcher writes its own log to runs/<name>/agent.log; keep stdio
        # attached to the same file in case the launcher itself errors before
        # exec'ing claude.
        log = run_dir / "agent.log"
        log_f = log.open("ab")
        log_f.write(f"\n--- spawn agent {datetime.now().isoformat()} ---\n".encode())
        p = subprocess.Popen(
            [str(AGENT_LAUNCHER), name],
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(PROJ),
            env=env,
        )
        self.procs[name] = p

    def stop(self, name: str) -> None:
        p = self.procs.get(name)
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        flag = RUNS_DIR / name / "agent_mode.flag"
        flag.unlink(missing_ok=True)


jobs = JobManager()
agents = AgentManager()


def _webapp_self_url() -> str:
    host = os.environ.get("HOST", "127.0.0.1")
    port = os.environ.get("PORT", "8000")
    # 0.0.0.0 isn't reachable from a child process the same way; rewrite.
    if host in ("0.0.0.0", ""):
        host = "127.0.0.1"
    return f"http://{host}:{port}"


# ---------------------------------------------------------- run state model

def _read_json_safe(p: Path) -> Any:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _file_url(name: str, rel: Path) -> str:
    return f"/api/runs/{name}/file?p={rel.as_posix()}"


def compute_state(run_dir: Path) -> dict:
    name = run_dir.name
    state: dict[str, Any] = {
        "name": name,
        "running": jobs.is_running(name),
        "agent_running": agents.is_running(name),
        "mode": "agent" if (run_dir / "agent_mode.flag").exists() else "manual",
        "model": "lucy",  # overridden below from metadata.json
        "input": None,
        "segments": None,
        "chosen": None,
        "segment_video": None,
        "first_frame": None,
        "analysis": None,
        "count": None,
        "approved_prompts": None,
        "variations": [],
        "outputs": [],
        "log_tail": "",
        "gate": None,  # 'pick-segment' | 'choose-count' | 'approve-prompts' | None
    }

    meta = _read_json_safe(run_dir / "00_input" / "metadata.json")
    if meta:
        state["input"] = meta
        state["source_video"] = _file_url(name, Path("00_input/source.mp4"))
        if meta.get("model") in ("lucy", "runway"):
            state["model"] = meta["model"]

    segs = _read_json_safe(run_dir / "01_twelvelabs" / "segments.json")
    if segs is not None:
        state["segments"] = segs

    chosen = _read_json_safe(run_dir / "02_segment" / "chosen.json")
    if chosen:
        state["chosen"] = chosen

    seg_mp4 = run_dir / "02_segment" / "segment.mp4"
    first_png = run_dir / "02_segment" / "first_frame.png"
    if seg_mp4.exists():
        state["segment_video"] = _file_url(name, Path("02_segment/segment.mp4"))
    if first_png.exists():
        state["first_frame"] = _file_url(name, Path("02_segment/first_frame.png"))

    analysis = _read_json_safe(run_dir / "03_analysis" / "prompts.json")
    if analysis:
        state["analysis"] = analysis

    count_p = run_dir / "04_variations" / ".count"
    if count_p.exists():
        try:
            state["count"] = int(count_p.read_text().strip())
        except ValueError:
            pass

    approved = _read_json_safe(run_dir / "04_variations" / "approved_prompts.json")
    if isinstance(approved, list):
        state["approved_prompts"] = approved

    var_dir = run_dir / "04_variations"
    reconstructed: list[str] = []
    if var_dir.exists():
        for sub in sorted(var_dir.glob("var_*")):
            idx = int(sub.name.split("_")[1])
            prompt_file = sub / "prompt.txt"
            img = sub / "image.png"
            ptext = prompt_file.read_text() if prompt_file.exists() else None
            if ptext:
                reconstructed.append(ptext)
            state["variations"].append({
                "index": idx,
                "prompt": ptext,
                "image": _file_url(name, Path(f"04_variations/{sub.name}/image.png"))
                if img.exists() else None,
            })
    # If approved_prompts.json is missing but per-var prompts exist, treat the
    # presence of the variation directories as evidence of approval. This keeps
    # older runs (made before the approval sentinel) from rendering as "locked".
    if state["approved_prompts"] is None and reconstructed and state.get("count"):
        if len(reconstructed) >= state["count"]:
            state["approved_prompts"] = reconstructed[: state["count"]]

    # Stage-5 output dir depends on the render model. Old runs may only have
    # 05_lucy_edit; new runway runs land in 05_runway. Read whichever exists.
    out_dir_name = "05_runway" if state["model"] == "runway" else "05_lucy_edit"
    out_dir = run_dir / out_dir_name
    if not out_dir.exists():
        # fall back to whichever dir the run actually has on disk
        for candidate in ("05_lucy_edit", "05_runway"):
            if (run_dir / candidate).exists():
                out_dir = run_dir / candidate
                out_dir_name = candidate
                break
    if out_dir.exists():
        for sub in sorted(out_dir.glob("var_*")):
            if not sub.name.startswith("var_"):
                continue
            try:
                idx = int(sub.name.split("_")[1])
            except (ValueError, IndexError):
                continue
            out = sub / "output.mp4"
            poll = _read_json_safe(sub / "poll.json")
            state["outputs"].append({
                "index": idx,
                "video": _file_url(name, Path(f"{out_dir_name}/{sub.name}/output.mp4"))
                if out.exists() else None,
                "status": (poll or {}).get("status") if poll else None,
            })

    # Determine gate / overall status.
    if (state["input"] is not None
            and state["segments"] is None
            and not state["running"]):
        # Pre-pipeline: clip uploaded but the user hasn't approved sending to
        # TwelveLabs/Pegasus yet. Manual runs sit here until the user clicks
        # "send to TwelveLabs"; agent-mode runs are auto-kicked at upload.
        state["gate"] = "start"
    elif state["segments"] is not None and not state["chosen"]:
        state["gate"] = "pick-segment"
    elif state["analysis"] and state["count"] is None:
        state["gate"] = "choose-count"
    elif state["count"] and state["analysis"] and state["approved_prompts"] is None:
        state["gate"] = "approve-prompts"

    log = run_dir / "pipeline.log"
    if log.exists():
        try:
            with log.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 8192))
                state["log_tail"] = f.read().decode("utf-8", errors="replace")
        except OSError:
            pass

    decisions = run_dir / "agent_decisions.log"
    if decisions.exists():
        try:
            state["agent_decisions"] = decisions.read_text(errors="replace")
        except OSError:
            state["agent_decisions"] = ""
    else:
        state["agent_decisions"] = ""

    agent_log = run_dir / "agent.log"
    state["agent_log_tail"] = ""
    if agent_log.exists():
        try:
            with agent_log.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 16384))
                state["agent_log_tail"] = f.read().decode("utf-8", errors="replace")
        except OSError:
            pass

    return state


def list_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    out = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta = _read_json_safe(d / "00_input" / "metadata.json")
        chosen = _read_json_safe(d / "02_segment" / "chosen.json")
        running = jobs.is_running(d.name)
        outputs = list((d / "05_lucy_edit").glob("var_*/output.mp4")) if (d / "05_lucy_edit").exists() else []
        # Cheap status label.
        if running:
            status = "running"
        elif outputs:
            count_p = d / "04_variations" / ".count"
            try:
                expected = int(count_p.read_text().strip()) if count_p.exists() else 0
            except ValueError:
                expected = 0
            status = "done" if expected and len(outputs) >= expected else "partial"
        elif (d / "01_twelvelabs" / "segments.json").exists() and not chosen:
            status = "needs-segment"
        elif (d / "03_analysis" / "prompts.json").exists() and not (d / "04_variations" / ".count").exists():
            status = "needs-count"
        elif (d / "04_variations" / ".count").exists() and not (d / "04_variations" / "approved_prompts.json").exists():
            status = "needs-approval"
        elif meta and not (d / "01_twelvelabs" / "segments.json").exists():
            status = "needs-start"
        elif meta:
            status = "idle"
        else:
            status = "empty"
        out.append({
            "name": d.name,
            "status": status,
            "duration": (meta or {}).get("duration"),
        })
    return out


# -------------------------------------------------------------------- app

app = FastAPI(title="RoboClaw")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/api/runs")
async def api_list_runs() -> JSONResponse:
    return JSONResponse(list_runs())


@app.get("/api/runs/{name}")
async def api_run(name: str) -> JSONResponse:
    run = RUNS_DIR / name
    if not run.exists():
        raise HTTPException(404, "no such run")
    return JSONResponse(compute_state(run))


@app.get("/api/runs/{name}/file")
async def api_run_file(name: str, p: str) -> Response:
    run = RUNS_DIR / name
    target = (run / p).resolve()
    if not target.exists() or not str(target).startswith(str(run.resolve())):
        raise HTTPException(404, "no such file")
    return FileResponse(target)


def _safe_name(stem: str) -> str:
    return re.sub(r"[^\w.-]", "_", stem)[:40]


@app.post("/api/runs")
async def api_create_run(
    file: UploadFile = File(...),
    mode: str = Form("manual"),
    model: str = Form("lucy"),
) -> JSONResponse:
    if mode not in ("manual", "agent"):
        raise HTTPException(400, "mode must be 'manual' or 'agent'")
    if model not in ("lucy", "runway"):
        raise HTTPException(400, "model must be 'lucy' or 'runway'")
    if not file.filename:
        raise HTTPException(400, "missing filename")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = _safe_name(Path(file.filename).stem) or "clip"
    run_dir = RUNS_DIR / f"{ts}_{stem}"
    in_dir = run_dir / "00_input"
    in_dir.mkdir(parents=True, exist_ok=True)
    upload_path = in_dir / "_upload.bin"
    with upload_path.open("wb") as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    src_dst = in_dir / "source.mp4"
    # Strip audio + remux to mp4.
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(upload_path),
        "-c:v", "copy", "-an", "-movflags", "+faststart",
        str(src_dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        # video stream may need re-encode (e.g. odd codec). Retry with libx264.
        cmd2 = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(upload_path),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
            str(src_dst),
        ]
        r2 = subprocess.run(cmd2, capture_output=True, text=True)
        if r2.returncode != 0:
            raise HTTPException(400, f"ffmpeg failed: {r.stderr or r2.stderr}")
    upload_path.unlink(missing_ok=True)

    # Probe for metadata + duration check.
    try:
        dur = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(src_dst)],
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        wh = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0:s=x", str(src_dst)],
            capture_output=True, text=True, check=True,
        ).stdout.strip().split("x")
        w, h = int(wh[0]), int(wh[1])
    except Exception as e:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(400, f"could not probe video: {e}")

    if dur < 4.0:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(
            400, f"clip too short: {dur:.2f}s (TwelveLabs requires >= 4.0s)",
        )

    (in_dir / "metadata.json").write_text(json.dumps({
        "source_path": file.filename,
        "duration": dur, "width": w, "height": h,
        "model": model,
    }, indent=2))
    # Manual mode: do NOT auto-kick. The user must click "send to TwelveLabs"
    # so they can review the model choice and avoid burning Pegasus credits
    # by accident. Agent mode kicks immediately because the agent will
    # auto-approve through the start gate via its polling loop.
    if mode == "agent":
        jobs.kick(run_dir)
        try:
            agents.kick(run_dir, _webapp_self_url())
        except Exception as e:
            # Don't fail the run if the agent can't start; surface a hint instead.
            (run_dir / "agent.log").write_text(f"failed to spawn agent: {e}\n")
    return JSONResponse({"name": run_dir.name, "mode": mode, "model": model})


@app.post("/api/runs/{name}/start")
async def api_start(name: str, request: Request) -> JSONResponse:
    """Pass the start gate: kick the pipeline subprocess. Optionally
    update the render model choice at the same time."""
    run = RUNS_DIR / name
    if not run.exists():
        raise HTTPException(404, "no such run")
    meta_path = run / "00_input" / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(400, "no metadata yet")
    try:
        body = await request.json()
    except Exception:
        body = {}
    new_model = body.get("model")
    if new_model is not None:
        if new_model not in ("lucy", "runway"):
            raise HTTPException(400, "model must be 'lucy' or 'runway'")
        meta = _read_json_safe(meta_path) or {}
        if meta.get("model") != new_model:
            meta["model"] = new_model
            meta_path.write_text(json.dumps(meta, indent=2))
    jobs.kick(run)
    return JSONResponse({"ok": True})


@app.post("/api/runs/{name}/summon-agent")
async def api_summon_agent(name: str) -> JSONResponse:
    run = RUNS_DIR / name
    if not run.exists():
        raise HTTPException(404, "no such run")
    if agents.is_running(name):
        return JSONResponse({"ok": True, "already_running": True})
    try:
        agents.kick(run, _webapp_self_url())
    except Exception as e:
        raise HTTPException(500, f"could not start agent: {e}")
    return JSONResponse({"ok": True})


@app.post("/api/runs/{name}/dismiss-agent")
async def api_dismiss_agent(name: str) -> JSONResponse:
    run = RUNS_DIR / name
    if not run.exists():
        raise HTTPException(404, "no such run")
    agents.stop(name)
    return JSONResponse({"ok": True})


@app.post("/api/runs/{name}/pick-segment")
async def api_pick_segment(name: str, request: Request) -> JSONResponse:
    body = await request.json()
    idx = int(body.get("index", -1))
    run = RUNS_DIR / name
    segs = _read_json_safe(run / "01_twelvelabs" / "segments.json") or []
    if not 0 <= idx < len(segs):
        raise HTTPException(400, f"index out of range (0..{len(segs)-1})")
    out = run / "02_segment" / "chosen.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(segs[idx], indent=2))
    jobs.kick(run)
    return JSONResponse({"ok": True})


@app.post("/api/runs/{name}/count")
async def api_set_count(name: str, request: Request) -> JSONResponse:
    body = await request.json()
    n = int(body.get("n", 0))
    if n not in (1, 3, 8):
        raise HTTPException(400, "n must be 1, 3, or 8")
    run = RUNS_DIR / name
    p = run / "04_variations" / ".count"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(n))
    jobs.kick(run)
    return JSONResponse({"ok": True})


@app.post("/api/runs/{name}/regen")
async def api_regen(name: str, request: Request) -> JSONResponse:
    """Server-side regen of selected prompts via gpt-5.4-nano. Updates the working set."""
    body = await request.json()
    indices = sorted({int(i) for i in body.get("indices", [])})  # 1-based
    run = RUNS_DIR / name
    analysis = _read_json_safe(run / "03_analysis" / "prompts.json")
    if not analysis:
        raise HTTPException(400, "no analysis yet")
    count_p = run / "04_variations" / ".count"
    if not count_p.exists():
        raise HTTPException(400, "count not set")
    n = int(count_p.read_text().strip())
    working_p = run / "04_variations" / "working_prompts.json"
    if working_p.exists():
        current = _read_json_safe(working_p)
    else:
        current = list(analysis["prompts"][:n])
    if not all(1 <= i <= n for i in indices):
        raise HTTPException(400, f"indices must be in 1..{n}")
    frame = run / "02_segment" / "first_frame.png"
    if not frame.exists():
        raise HTTPException(400, "no first frame yet")

    # Call regen helper from pipeline.
    sys.path.insert(0, str(PROJ))
    try:
        import importlib
        import pipeline as pl
        importlib.reload(pl)
        pl.load_dotenv(PROJ / ".env")
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise HTTPException(500, "OPENROUTER_API_KEY missing")
        import httpx
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            new_list = await pl.or_regen_prompts(
                client, key, frame, current,
                analysis.get("variation_axes") or [], indices,
            )
    finally:
        if str(PROJ) in sys.path:
            sys.path.remove(str(PROJ))
    working_p.write_text(json.dumps(new_list, indent=2))
    return JSONResponse({"prompts": new_list})


@app.post("/api/runs/{name}/edit-prompt")
async def api_edit_prompt(name: str, request: Request) -> JSONResponse:
    body = await request.json()
    idx = int(body.get("index", 0))  # 1-based
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text required")
    run = RUNS_DIR / name
    count_p = run / "04_variations" / ".count"
    if not count_p.exists():
        raise HTTPException(400, "count not set")
    n = int(count_p.read_text().strip())
    if not 1 <= idx <= n:
        raise HTTPException(400, f"index must be in 1..{n}")
    working_p = run / "04_variations" / "working_prompts.json"
    if working_p.exists():
        current = _read_json_safe(working_p)
    else:
        analysis = _read_json_safe(run / "03_analysis" / "prompts.json") or {}
        current = list((analysis.get("prompts") or [])[:n])
    current[idx - 1] = text
    working_p.write_text(json.dumps(current, indent=2))
    return JSONResponse({"prompts": current})


@app.post("/api/runs/{name}/approve")
async def api_approve(name: str) -> JSONResponse:
    run = RUNS_DIR / name
    count_p = run / "04_variations" / ".count"
    if not count_p.exists():
        raise HTTPException(400, "count not set")
    n = int(count_p.read_text().strip())
    working_p = run / "04_variations" / "working_prompts.json"
    if working_p.exists():
        approved = _read_json_safe(working_p)
    else:
        analysis = _read_json_safe(run / "03_analysis" / "prompts.json") or {}
        approved = list((analysis.get("prompts") or [])[:n])
    if not isinstance(approved, list) or len(approved) != n:
        raise HTTPException(400, f"expected {n} prompts, got {approved}")
    (run / "04_variations" / "approved_prompts.json").write_text(
        json.dumps(approved, indent=2)
    )
    jobs.kick(run)
    return JSONResponse({"ok": True})


@app.post("/api/runs/{name}/advance")
async def api_advance(name: str) -> JSONResponse:
    run = RUNS_DIR / name
    if not run.exists():
        raise HTTPException(404, "no such run")
    jobs.kick(run)
    return JSONResponse({"ok": True})


@app.post("/api/runs/{name}/working-prompts")
async def api_get_working(name: str) -> JSONResponse:
    """Return the live editing buffer (or seed from analysis)."""
    run = RUNS_DIR / name
    working_p = run / "04_variations" / "working_prompts.json"
    if working_p.exists():
        return JSONResponse({"prompts": _read_json_safe(working_p)})
    analysis = _read_json_safe(run / "03_analysis" / "prompts.json") or {}
    count_p = run / "04_variations" / ".count"
    n = int(count_p.read_text().strip()) if count_p.exists() else 0
    return JSONResponse({"prompts": list((analysis.get("prompts") or [])[:n])})


if __name__ == "__main__":
    import uvicorn

    RUNS_DIR.mkdir(exist_ok=True)
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run("webapp:app", host=host, port=port, reload=False)
