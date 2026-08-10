"""
html_viewer.py — minimal, light-themed 3D scene viewer.

Shows exactly two things, nothing else:
  1. The user's query history (a clean conversation timeline).
  2. The latest interactive 3D scene (auto-rotating <model-viewer>).

Design notes:
- Warm, bright "spring" palette (cream background, sage-green accents) — no dark blue.
- The page is written ONCE as a static shell; per-turn updates are pushed via a
  small `viewer_state.json` that the page polls. The <model-viewer> src is only
  swapped when a new GLB actually appears, so the rotation is never interrupted
  by a full-page reload (the old meta-refresh approach reloaded the whole GLB).
- All UI text is English.

Local HTTP server serves the session dir; the GLB loads by relative path.
"""

import http.server
import json
import os
import socketserver
import threading
import webbrowser
from typing import List, Optional


# ── Static page shell (written once). All live data comes from viewer_state.json ──
_SHELL = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VibeWorld</title>
<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
<style>
  :root {
    --bg-1: #fbf8f1;          /* warm cream */
    --bg-2: #f3f0e6;          /* soft sand  */
    --card: #ffffff;
    --ink: #3c3a33;           /* warm charcoal */
    --ink-soft: #9a9484;      /* muted taupe */
    --line: #ece6d7;
    --sage: #7f9b6e;          /* spring sage */
    --sage-soft: #eef2e6;
    --terra: #cf9366;         /* warm terracotta accent */
    --shadow: 0 18px 48px rgba(90, 80, 55, 0.10);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    color: var(--ink);
    font-family: -apple-system, "Segoe UI", "PingFang SC", "Helvetica Neue", sans-serif;
    background:
      radial-gradient(1200px 600px at 82% -8%, #fff6e9 0%, rgba(255,246,233,0) 60%),
      radial-gradient(900px 500px at -8% 110%, #eaf1e4 0%, rgba(234,241,228,0) 55%),
      linear-gradient(160deg, var(--bg-1), var(--bg-2));
    background-attachment: fixed;
  }
  .app { display: flex; flex-direction: column; height: 100vh; padding: 22px; gap: 18px; }

  /* Header */
  .head { display: flex; align-items: center; justify-content: space-between; padding: 2px 6px; }
  .brand { display: flex; align-items: baseline; gap: 10px; }
  .brand .mark {
    font-family: Georgia, "Songti SC", serif; font-size: 1.5rem; font-weight: 700;
    letter-spacing: 0.2px; color: var(--ink);
  }
  .brand .mark .w { color: var(--sage); }
  .brand .tag { font-size: 0.82rem; color: var(--ink-soft); letter-spacing: 0.3px; }
  .chip {
    display: inline-flex; align-items: center; gap: 8px; background: var(--card);
    border: 1px solid var(--line); border-radius: 999px; padding: 7px 14px;
    font-size: 0.8rem; color: var(--ink-soft); box-shadow: 0 4px 14px rgba(90,80,55,0.05);
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--sage); }
  .dot.busy { animation: pulse 1.1s ease-in-out infinite; background: var(--terra); }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.45;transform:scale(.8)} }
  .chip b { color: var(--ink); font-weight: 600; }

  /* Body: history rail + stage */
  .body { flex: 1; display: flex; gap: 18px; min-height: 0; }

  .rail {
    width: 320px; flex-shrink: 0; background: var(--card); border: 1px solid var(--line);
    border-radius: 20px; box-shadow: var(--shadow); display: flex; flex-direction: column;
    overflow: hidden;
  }
  .rail h2 {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 1.4px; text-transform: uppercase;
    color: var(--ink-soft); padding: 18px 20px 12px; border-bottom: 1px solid var(--line);
  }
  .rail .list { overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 10px; }
  .rail .list::-webkit-scrollbar { width: 8px; }
  .rail .list::-webkit-scrollbar-thumb { background: var(--line); border-radius: 8px; }
  .q {
    position: relative; background: #fcfaf4; border: 1px solid var(--line);
    border-left: 3px solid var(--sage); border-radius: 12px; padding: 11px 13px 12px 15px;
    animation: rise 0.35s ease both;
  }
  @keyframes rise { from{opacity:0; transform:translateY(6px)} to{opacity:1; transform:none} }
  .q .n { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.5px; color: var(--sage); margin-bottom: 3px; }
  .q .t { font-size: 0.92rem; line-height: 1.45; color: var(--ink); word-break: break-word; }
  .rail .empty { color: var(--ink-soft); font-size: 0.9rem; padding: 22px 20px; line-height: 1.5; }

  .stage {
    flex: 1; min-width: 0; background: var(--card); border: 1px solid var(--line);
    border-radius: 20px; box-shadow: var(--shadow); display: flex; flex-direction: column;
    padding: 16px; position: relative; overflow: hidden;
  }
  .stage::before {  /* subtle top sheen */
    content: ""; position: absolute; inset: 0 0 auto 0; height: 120px; pointer-events: none;
    background: linear-gradient(180deg, rgba(127,155,110,0.07), rgba(127,155,110,0));
  }
  .stage .caption {
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
    padding: 2px 6px 14px; z-index: 1;
  }
  .stage .caption .title { font-size: 0.95rem; font-weight: 600; color: var(--ink); }
  .stage .caption .meta { font-size: 0.8rem; color: var(--ink-soft); }
  .stage .caption .meta b { color: var(--sage); font-weight: 700; }
  .viewer-wrap {
    flex: 1; min-height: 0; border-radius: 16px; overflow: hidden;
    background:
      radial-gradient(120% 90% at 50% 0%, #ffffff 0%, #f6f3ea 70%, #f0ece0 100%);
    border: 1px solid var(--line); position: relative;
  }
  model-viewer { width: 100%; height: 100%; --poster-color: transparent; background: transparent; }
  .empty-3d {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    text-align: center; color: var(--ink-soft); font-size: 0.95rem; padding: 30px; line-height: 1.6;
  }
  .empty-3d span { max-width: 340px; }

  @media (max-width: 820px) {
    .body { flex-direction: column; }
    .rail { width: auto; max-height: 34vh; }
  }
</style>
</head>
<body>
<div class="app">
  <div class="head">
    <div class="brand">
      <div class="mark">Vibe<span class="w">World</span></div>
      <div class="tag">AI-driven 3D world construction</div>
    </div>
    <div class="chip">
      <span id="dot" class="dot"></span>
      <span id="status">Ready</span>
    </div>
  </div>

  <div class="body">
    <aside class="rail">
      <h2>Conversation</h2>
      <div id="qlist" class="list">
        <div class="empty">Describe a scene to start building your first 3D world.</div>
      </div>
    </aside>

    <section class="stage">
      <div class="caption">
        <div class="title">Latest 3D scene</div>
        <div class="meta">Assets <b id="assets">0</b></div>
      </div>
      <div class="viewer-wrap">
        <model-viewer id="mv" alt="Interactive 3D scene"
          camera-controls auto-rotate auto-rotate-delay="0" rotation-per-second="24deg"
          interaction-prompt="none" shadow-intensity="0.85" exposure="1.05"
          environment-image="neutral" camera-orbit="35deg 72deg 108%"
          style="display:none"></model-viewer>
        <div id="empty3d" class="empty-3d">
          <span>Your interactive 3D scene will appear here once the first assets are placed.</span>
        </div>
      </div>
    </section>
  </div>
</div>

<script>
  const STATUS = {
    thinking:  {t: "Thinking…",  busy: true},
    rendering: {t: "Rendering…", busy: true},
    waiting:   {t: "Ready",      busy: false},
    done:      {t: "Done",       busy: false},
  };
  let lastSrc = "__none__", lastQKey = "__none__";

  function renderQueries(qs) {
    const key = (qs || []).join("");
    if (key === lastQKey) return;      // no change → don't rebuild (keeps scroll)
    lastQKey = key;
    const list = document.getElementById("qlist");
    if (!qs || !qs.length) {
      list.innerHTML = '<div class="empty">Describe a scene to start building your first 3D world.</div>';
      return;
    }
    list.innerHTML = "";
    qs.forEach((q, i) => {
      const item = document.createElement("div");
      item.className = "q";
      const n = document.createElement("div"); n.className = "n"; n.textContent = "Prompt " + (i + 1);
      const t = document.createElement("div"); t.className = "t"; t.textContent = q;
      item.appendChild(n); item.appendChild(t); list.appendChild(item);
    });
    list.scrollTop = list.scrollHeight;   // keep newest in view
  }

  async function tick() {
    try {
      const r = await fetch("viewer_state.json?ts=" + Date.now(), {cache: "no-store"});
      if (!r.ok) return;
      const s = await r.json();

      document.getElementById("assets").textContent = s.actor_count || 0;

      const st = STATUS[s.status] || {t: s.status || "Ready", busy: false};
      document.getElementById("status").textContent = st.t;
      document.getElementById("dot").className = "dot" + (st.busy ? " busy" : "");

      renderQueries(s.queries);

      const mv = document.getElementById("mv"), empty = document.getElementById("empty3d");
      const src = s.glb ? (s.glb + "?v=" + (s.ver || 0)) : null;
      if (src !== lastSrc) {
        lastSrc = src;
        if (src) { mv.setAttribute("src", src); mv.style.display = ""; empty.style.display = "none"; }
        else { mv.removeAttribute("src"); mv.style.display = "none"; empty.style.display = ""; }
      }
    } catch (e) { /* transient; retry next tick */ }
  }
  setInterval(tick, 1500);
  tick();
</script>
</body>
</html>
"""


def _find_free_port(start: int = 7860) -> int:
    import socket
    for port in range(start, start + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue
    return start


def _start_http_server(directory: str, port: int):
    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def log_message(self, *args):
            pass

        def end_headers(self):
            # GLB / state change every turn; never cache.
            self.send_header("Cache-Control", "no-store, must-revalidate")
            super().end_headers()

    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer(("", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


class HtmlViewer:
    """Minimal light-themed viewer: query history + latest rotating 3D scene.

    Public API is unchanged (update / set_query / set_model / set_status / reset,
    plus ._url ._dir ._opened) so session.py / cli.py keep working. Live data is
    pushed to viewer_state.json; the static viewer.html polls it.
    """

    def __init__(self, output_path: str, query: str = "", model_name: str = "", port: int = 0):
        self.output_path = output_path
        self.model_name = model_name
        self.queries: List[str] = []
        self.status = "waiting"
        self.actor_count = 0
        self.turn_count = 0
        self.latest_glb: Optional[str] = None   # relative path to newest scene GLB
        self.latest_ver = 0                      # bumped per new GLB (cache-bust)
        self._opened = False
        self._dir = os.path.dirname(os.path.abspath(output_path))
        self._state_path = os.path.join(self._dir, "viewer_state.json")
        self._port = _find_free_port(port if port else 7860)
        self._filename = os.path.basename(output_path)
        _start_http_server(self._dir, self._port)
        self._url = f"http://localhost:{self._port}/{self._filename}"
        if query:
            self.add_query(query)
        self._write_shell()
        self._write_state()

    # ── query history ─────────────────────────────────────────────────────────
    def add_query(self, query: str):
        """Append a user prompt to the history (skips blanks / exact repeats)."""
        q = (query or "").strip()
        if q and (not self.queries or self.queries[-1] != q):
            self.queries.append(q)
        self._write_state()

    # ── per-turn update (called by Session) ────────────────────────────────────
    def update(
        self,
        turn_idx: int,
        reasoning: str = "",
        tool_calls: Optional[List[dict]] = None,
        images: Optional[List[str]] = None,   # accepted but not displayed (kept minimal)
        glb: Optional[str] = None,
        is_current: bool = True,
        status: str = "thinking",
        actor_count: int = 0,
    ):
        if turn_idx > self.turn_count:
            self.turn_count = turn_idx
        if glb:
            self.latest_glb = glb
            self.latest_ver += 1
        self.status = status
        self.actor_count = actor_count
        self._write_state()
        self._maybe_open()

    def _maybe_open(self):
        if not self._opened:
            threading.Thread(target=lambda: webbrowser.open(self._url), daemon=True).start()
            self._opened = True

    def reset(self):
        """/clear: wipe history and 3D scene."""
        self.queries = []
        self.actor_count = 0
        self.turn_count = 0
        self.latest_glb = None
        self.latest_ver += 1
        self.status = "waiting"
        self._write_state()

    # ── back-compat setters ─────────────────────────────────────────────────────
    def set_query(self, query: str, model_name: str = None):
        if model_name is not None:
            self.model_name = model_name
        self.add_query(query)

    def set_model(self, model_name: str):
        self.model_name = model_name
        self._write_state()

    def set_status(self, status: str, actor_count: int = 0):
        self.status = status
        if actor_count:
            self.actor_count = actor_count
        self._write_state()

    # ── writers ─────────────────────────────────────────────────────────────────
    def _write_shell(self):
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(_SHELL)

    def _write_state(self):
        state = {
            "model": self.model_name or "—",
            "status": self.status,
            "turn_count": self.turn_count,
            "actor_count": self.actor_count,
            "glb": self.latest_glb,
            "ver": self.latest_ver,
            "queries": self.queries,
        }
        tmp = self._state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        os.replace(tmp, self._state_path)   # atomic swap so the poll never reads a half-written file
