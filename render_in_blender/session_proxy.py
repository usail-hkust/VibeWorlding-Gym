"""
session_proxy.py — session-hash 粘滞的反向代理

将外部 :8080 的请求路由到多个 Gradio worker（:7000~:7063）。
- /queue/join, /queue/data, /upload：按 session_hash 粘滞到同一 worker
- 其余请求（/config, /info 等）：round-robin

用法：
    python session_proxy.py --port 8080 --base_port 7000 --workers 64
"""

import argparse
import asyncio
import hashlib
import http
import logging
import os
import re
import socket
import sys
import urllib.parse

# ── 兼容性：优先用 aiohttp，没有就用内置 http.server（单线程降级）─────────────
try:
    import aiohttp
    from aiohttp import web, ClientSession, ClientTimeout, TCPConnector
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [proxy] %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("proxy")

# ── 参数 ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--port",       type=int, default=8080)
parser.add_argument("--base_port",  type=int, default=7000)
parser.add_argument("--workers",    type=int, default=64)
parser.add_argument("--host",       default="127.0.0.1")
args = parser.parse_args()

WORKERS     = [f"http://{args.host}:{args.base_port + i}" for i in range(args.workers)]
_rr_counter = 0

def _pick_worker(session_hash: str | None) -> str:
    global _rr_counter
    if session_hash:
        idx = int(hashlib.md5(session_hash.encode()).hexdigest(), 16) % len(WORKERS)
        return WORKERS[idx]
    _rr_counter = (_rr_counter + 1) % len(WORKERS)
    return WORKERS[_rr_counter]

def _extract_session_hash(req_path: str, query_str: str) -> str | None:
    qs = urllib.parse.parse_qs(query_str or "")
    if "session_hash" in qs:
        return qs["session_hash"][0]
    try:
        body_match = re.search(r'"session_hash"\s*:\s*"([^"]+)"', req_path)
        if body_match:
            return body_match.group(1)
    except Exception:
        pass
    return None

# ── aiohttp 版（高性能，多 worker 并发）────────────────────────────────────────
if HAS_AIOHTTP:
    # Heartbeat 请求路径前缀（新服务不支持，静默返回 404，不打日志）
    _SILENT_PREFIXES = ("/heartbeat",)

    async def handle(request: web.Request) -> web.StreamResponse:
        parsed   = urllib.parse.urlparse(str(request.url))

        # 静默处理 heartbeat（新服务返回 404，客户端会自行处理，不需要打日志）
        if parsed.path.startswith(_SILENT_PREFIXES):
            return web.Response(status=404, text="Not Found")

        sh       = _extract_session_hash(parsed.path, parsed.query)
        upstream = _pick_worker(sh)
        target   = upstream + parsed.path + (("?" + parsed.query) if parsed.query else "")

        # Read request body
        try:
            body = await request.read()
        except Exception:
            body = b""

        # If session_hash in body JSON
        if not sh and body:
            sh = _extract_session_hash("", "")
            try:
                import json
                d = json.loads(body)
                sh = d.get("session_hash")
                if sh:
                    upstream = _pick_worker(sh)
                    target = upstream + parsed.path + (("?" + parsed.query) if parsed.query else "")
            except Exception:
                pass

        timeout = ClientTimeout(total=1800, connect=10)
        connector = TCPConnector(limit=0)

        try:
            async with ClientSession(connector=connector, timeout=timeout) as sess:
                # SSE streams need special handling
                is_sse = "text/event-stream" in request.headers.get("Accept", "")
                method = request.method
                headers = {k: v for k, v in request.headers.items()
                           if k.lower() not in ("host", "content-length")}

                async with sess.request(
                    method, target,
                    headers=headers,
                    data=body if body else None,
                    allow_redirects=False,
                ) as upstream_resp:
                    # Stream response
                    resp = web.StreamResponse(
                        status=upstream_resp.status,
                        headers={k: v for k, v in upstream_resp.headers.items()
                                 if k.lower() not in ("transfer-encoding", "content-length")},
                    )
                    await resp.prepare(request)
                    async for chunk in upstream_resp.content.iter_any():
                        await resp.write(chunk)
                    await resp.write_eof()
                    return resp

        except Exception as e:
            log.warning(f"upstream error {target}: {e}")
            return web.Response(status=502, text=f"Bad Gateway: {e}")

    app = web.Application()
    app.router.add_route("*", "/{path_info:.*}", handle)

    async def main():
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", args.port)
        await site.start()
        log.info(f"session_proxy listening on :{args.port} → {len(WORKERS)} workers "
                 f"({args.host}:{args.base_port}~{args.base_port+len(WORKERS)-1})")
        await asyncio.Event().wait()

    asyncio.run(main())

# ── 降级：内置 http.server（无 aiohttp 时，单进程转发，适合少并发场景）──────────
else:
    import threading
    import http.server
    import urllib.request

    log.warning("aiohttp not found, using single-threaded fallback proxy")

    class ProxyHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass  # 静默

        def _forward(self, body=None):
            parsed   = urllib.parse.urlparse(self.path)
            # 静默 heartbeat
            if parsed.path.startswith("/heartbeat"):
                self.send_response(404)
                self.end_headers()
                return
            sh       = _extract_session_hash(parsed.path, parsed.query)
            upstream = _pick_worker(sh)
            target   = upstream + self.path

            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host",)}
            req = urllib.request.Request(target, data=body, headers=headers,
                                         method=self.command)
            try:
                with urllib.request.urlopen(req, timeout=1800) as resp:
                    self.send_response(resp.status)
                    for k, v in resp.headers.items():
                        if k.lower() not in ("transfer-encoding",):
                            self.send_header(k, v)
                    self.end_headers()
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except Exception as e:
                self.send_error(502, str(e))

        def do_GET(self):    self._forward()
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else None
            self._forward(body)
        def do_OPTIONS(self): self._forward()

    server = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), ProxyHandler)
    log.info(f"session_proxy (fallback) listening on :{args.port} → {len(WORKERS)} workers")
    server.serve_forever()
