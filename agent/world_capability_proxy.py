#!/usr/bin/env python3
"""A narrow HTTP capability proxy for Clawblox agents.

The proxy is intended to be the only network path visible to sandboxed agent
subprocesses. It accepts HTTP proxy absolute-form requests for one synthetic
world hostname and forwards only approved world API paths to one assigned world
server.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

ALLOWED_EXACT_PATHS = {
    "/observe",
    "/input",
    "/chat",
    "/chat/messages",
    "/api.md",
    "/skill.md",
    "/renderer/manifest",
}

ALLOWED_PREFIXES = (
    "/renderer-files/",
    "/assets/",
    "/static/",
)

DENIED_PREFIXES = (
    "/api/",
    "/replay/",
    "/sessions/",
)


class ProxyConfig:
    def __init__(
        self,
        *,
        target_base_url: str,
        public_host: str,
        session_token_file: Path,
        session_header: str,
        port_file: Path | None,
        log_file: Path | None,
    ) -> None:
        target = urllib.parse.urlparse(target_base_url)
        if target.scheme != "http" or not target.hostname:
            raise ValueError("--target-base-url must be an http URL with a host")
        self.target_host = target.hostname
        self.target_port = target.port or 80
        self.target_base_path = target.path.rstrip("/")
        self.public_host = public_host.lower()
        self.session_token_file = session_token_file
        self.session_header = session_header
        self.port_file = port_file
        self.log_file = log_file

    def expected_session_token(self) -> str:
        try:
            return self.session_token_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""

    def log(self, message: str) -> None:
        line = message.rstrip() + "\n"
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with self.log_file.open("a", encoding="utf-8") as fh:
                fh.write(line)
        else:
            sys.stderr.write(line)
            sys.stderr.flush()


class CapabilityProxyHandler(BaseHTTPRequestHandler):
    server_version = "ClawbloxCapabilityProxy/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def cfg(self) -> ProxyConfig:
        return self.server.cfg  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: object) -> None:
        self.cfg.log("%s - %s" % (self.address_string(), fmt % args))

    def do_CONNECT(self) -> None:  # noqa: N802
        self.reject(403, "CONNECT is not allowed")

    def do_GET(self) -> None:  # noqa: N802
        self.forward()

    def do_POST(self) -> None:  # noqa: N802
        self.forward()

    def do_HEAD(self) -> None:  # noqa: N802
        self.forward()

    def reject(self, status: int, message: str) -> None:
        body = (json.dumps({"error": message}) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def parse_target(self) -> tuple[str, str] | None:
        raw = self.path
        parsed = urllib.parse.urlparse(raw)
        if parsed.scheme:
            if parsed.scheme != "http":
                self.reject(403, "Only http requests are allowed")
                return None
            host = (parsed.hostname or "").lower()
            if host != self.cfg.public_host:
                self.reject(403, "Request host is not this agent's world capability")
                return None
            if parsed.port not in (None, 80):
                self.reject(403, "Request port is not allowed")
                return None
            path = urllib.parse.urlunparse(("", "", parsed.path or "/", "", parsed.query, ""))
            return host, path

        host_header = self.headers.get("Host", "")
        host = host_header.split(":", 1)[0].strip().lower()
        if host != self.cfg.public_host:
            self.reject(403, "Request host is not this agent's world capability")
            return None
        return host, raw or "/"

    def validate_path(self, path_with_query: str) -> bool:
        parsed = urllib.parse.urlparse(path_with_query)
        path = parsed.path or "/"
        if any(path == prefix[:-1] or path.startswith(prefix) for prefix in DENIED_PREFIXES):
            self.reject(403, "Replay/app/admin paths are not allowed")
            return False
        if path in ALLOWED_EXACT_PATHS:
            return True
        if any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES):
            return True
        self.reject(403, "Path is not in the world API capability")
        return False

    def validate_session(self) -> bool:
        expected = self.cfg.expected_session_token()
        if not expected:
            self.reject(503, "World session token is not ready")
            return False
        provided = self.headers.get(self.cfg.session_header, "").strip()
        if provided != expected:
            self.reject(401, "Missing or invalid world session token")
            return False
        return True

    def forward(self) -> None:
        parsed = self.parse_target()
        if parsed is None:
            return
        _, path_with_query = parsed
        if not self.validate_path(path_with_query):
            return
        if not self.validate_session():
            return

        content_length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(content_length) if content_length else None

        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
        }
        headers["Host"] = f"{self.cfg.target_host}:{self.cfg.target_port}"

        upstream_path = self.cfg.target_base_path + path_with_query
        conn = http.client.HTTPConnection(
            self.cfg.target_host,
            self.cfg.target_port,
            timeout=30,
        )
        try:
            conn.request(self.command, upstream_path, body=body, headers=headers)
            response = conn.getresponse()
            data = response.read()
        except Exception as exc:  # pragma: no cover - defensive operational path
            self.reject(502, f"World upstream request failed: {exc}")
            return
        finally:
            conn.close()

        self.send_response(response.status, response.reason)
        for key, value in response.getheaders():
            if key.lower() in HOP_BY_HOP_HEADERS or key.lower() == "content-length":
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)


class CapabilityProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr: tuple[str, int], cfg: ProxyConfig) -> None:
        self.cfg = cfg
        super().__init__(addr, CapabilityProxyHandler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=0)
    parser.add_argument("--target-base-url", required=True)
    parser.add_argument("--public-host", required=True)
    parser.add_argument("--session-token-file", type=Path, required=True)
    parser.add_argument("--session-header", default="X-Session")
    parser.add_argument("--port-file", type=Path)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    cfg = ProxyConfig(
        target_base_url=args.target_base_url,
        public_host=args.public_host,
        session_token_file=args.session_token_file,
        session_header=args.session_header,
        port_file=args.port_file,
        log_file=args.log_file,
    )

    server = CapabilityProxyServer((args.listen_host, args.listen_port), cfg)
    actual_port = server.server_address[1]
    if args.port_file:
        args.port_file.parent.mkdir(parents=True, exist_ok=True)
        args.port_file.write_text(f"{actual_port}\n", encoding="utf-8")
    if args.pid_file:
        args.pid_file.parent.mkdir(parents=True, exist_ok=True)
        args.pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")

    stop_event = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    cfg.log(
        f"listening on {args.listen_host}:{actual_port}, "
        f"public_host={args.public_host}, target={args.target_base_url}"
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
