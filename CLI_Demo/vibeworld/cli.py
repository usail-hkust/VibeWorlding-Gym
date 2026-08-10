"""
cli.py — vibeworld command-line entry (REPL + harness)

Installed as the `vibeworld` command.

Usage:
  vibeworld                          # interactive REPL
  vibeworld --query "build a garden" # run one query first, then enter the REPL
  vibeworld --demo 1                 # use a preset demo query

REPL commands:
  /model [name]   switch model; no arg lists available models
  /refine <dir>   enter Refine mode (load init_map/component_info/query.json)
  /clear          clear the scene and conversation
  /compact        compact history (keep scene state)
  /help           help
  /quit /exit     exit

Interrupt:
  Ctrl+C while a turn is running → interrupt this turn, back to the prompt
  Ctrl+C twice at an empty prompt (or /quit) → exit

Service addresses (render / retrieve) are configured centrally in setup.py —
no interactive prompt at startup. Override with --server / --retrieve-server or
the VIBEWORLD_RENDER_SERVER / VIBEWORLD_RETRIEVE_SERVER environment variables.
"""

import argparse
import contextlib
import importlib.util as _ilu
import io
import json
import os
import shutil
import sys
import textwrap
import threading
import time
import unicodedata
from datetime import datetime

# readline: line editing / history for the input() fallback path (no-op if
# prompt_toolkit is available). Import-only, nothing to call.
try:
    import readline  # noqa: F401
    _HAS_READLINE = True
except ImportError:
    _HAS_READLINE = False

# ── Rich ─────────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.rule import Rule
    from rich.table import Table
    from rich import box as _rich_box
    _RICH = True
except ImportError:
    _RICH = False

# force_terminal when the real stdout is a TTY: under prompt_toolkit's
# patch_stdout the stdout proxy reports isatty()=False, which would make rich
# drop colors for a background query's output printed above the live prompt.
try:
    _REAL_TTY = sys.stdout.isatty()
except Exception:
    _REAL_TTY = False
console = Console(force_terminal=True if _REAL_TTY else None) if _RICH else None


def _print(msg: str = "", style: str = ""):
    if _RICH and console:
        console.print(msg, style=style)
    else:
        print(msg)


def _hr(label: str = ""):
    """A thin horizontal rule, used as the top/bottom edge of the input box.
    A non-empty label is embedded left-aligned (e.g. the refine-mode marker)."""
    if _RICH and console:
        console.print(Rule(label, style="grey37", align="left") if label
                      else Rule(style="grey37"))
    else:
        print(("-- " + label + " " + "-" * max(4, 50 - len(label))) if label else "-" * 56)


def _bullet(msg: str, style: str = "bold"):
    """Claude-Code-style primary line: starts with ● (one reasoning turn / action)."""
    if _RICH and console:
        console.print(f"[{style}]●[/{style}] {msg}", highlight=False)
    else:
        print(f"● {msg}")


def _branch(msg: str, style: str = "dim"):
    """Tree branch continuation: starts with ⎿, for tool calls / render results."""
    _turn_header()
    if _RICH and console:
        console.print(f"  [grey37]⎿[/grey37] [{style}]{msg}[/{style}]", highlight=False)
    else:
        print(f"  ⎿ {msg}")


def _panel(content: str, title: str = "", style: str = "blue"):
    if _RICH and console:
        console.print(Panel(content, title=title, border_style=style))
    else:
        print(f"[{title}]\n{content}\n")


DEMO_QUERIES = [
    "Build a serene Japanese garden with cherry blossom trees, stone lanterns and a bamboo fence",
    "Create a magical forest campsite with tents, a bonfire and ancient towering trees",
    "Craft a classical Chinese courtyard with a pavilion, a lotus pond and rockery stones",
]

QUALITY_MAP = {"low": "低质量 (快速预览)", "medium": "中质量 (默认)", "high": "高质量"}

VERSION = "1.0.0"

# Fallback service addresses if setup.py can't be read (should not happen).
_FALLBACK_RENDER_SERVER = "http://localhost:8080"
_FALLBACK_RETRIEVE_SERVER = "http://localhost:8081"

# REPL slash commands: name -> one-line description (used by the completion menu).
SLASH_COMMANDS = {
    "/model":   "Switch model; no arg lists available models",
    "/refine":  "Enter Refine mode (load a data directory)",
    "/clear":   "Clear the scene and conversation",
    "/compact": "Compact history (keep scene state)",
    "/help":    "Show command help",
    "/quit":    "Exit",
    "/exit":    "Exit",
}


def _load_service_config():
    """Read the centrally-configured service addresses from setup.py.

    Returns (render_server, retrieve_server); falls back to hardcoded defaults
    if setup.py can't be imported. Also injects:
      - BAILIAN_API_KEY      → StreamingBailianMultiChat (cloud qwen3.8-max)
      - VIBEWORLD_LOCAL_VLLM_URL → LocalStreamingVLLMMultiChat (local SFT model)
    Existing env vars always win; setup.py constants are fallback-only.
    """
    try:
        setup_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "setup.py"
        )
        spec = _ilu.spec_from_file_location("vibeworld_setup", setup_path)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # ── cloud Bailian key ────────────────────────────────────────────────
        bailian_key = getattr(mod, "BAILIAN_API_KEY", "")
        if bailian_key and not (os.environ.get("BAILIAN_API_KEY")
                                or os.environ.get("DASHSCOPE_API_KEY")):
            os.environ["BAILIAN_API_KEY"] = bailian_key
        # ── local vLLM URL ───────────────────────────────────────────────────
        local_vllm = getattr(mod, "LOCAL_VLLM_SERVER", "").strip()
        if local_vllm and not os.environ.get("VIBEWORLD_LOCAL_VLLM_URL"):
            os.environ["VIBEWORLD_LOCAL_VLLM_URL"] = local_vllm
        return (getattr(mod, "RENDER_SERVER", _FALLBACK_RENDER_SERVER),
                getattr(mod, "RETRIEVE_SERVER", _FALLBACK_RETRIEVE_SERVER))
    except Exception:
        return (_FALLBACK_RENDER_SERVER, _FALLBACK_RETRIEVE_SERVER)


# ── Terminal callbacks: print thinking + tool calls + render results ───────────
# Each turn is led by a `✳ Thinking…` header, with the reasoning and the tool /
# render lines hanging off it as ⎿ children.
#
# No elapsed time here: the live ticking counter is in the input box's rprompt
# (see _read_line), so a second, frozen number on the header would just be noise.
#
# The header is printed LAZILY — by whatever emits the turn's first line, not by
# on_turn_start — so a turn that produces no output never prints an orphan header.
_TURN = {"header": False}


def _turn_header():
    if _TURN["header"]:
        return
    _TURN["header"] = True
    if _RICH and console:
        console.print("[magenta]✳[/] [dim]Thinking…[/]", highlight=False)
    else:
        print("✳ Thinking…")


def on_turn_start(turn: int):
    # A blank line separates turns. The turn's leading marker is the lazy
    # `✳ Thinking…` header, so nothing is printed here beyond the separator.
    _stream_end()
    print()
    _TURN["header"] = False


def _names_from(items, limit=4):
    """Pull `name` from a [{name:...}] list; elide with … past `limit`."""
    names = [str(it.get("name", "?")) for it in items if isinstance(it, dict)]
    if not names:
        return ""
    shown = names[:limit]
    tail = f" …+{len(names) - limit} more" if len(names) > limit else ""
    return ", ".join(shown) + tail


def _summarize_tool_call(fc: dict) -> str:
    """Compress a tool_call into a single-line summary (no full JSON args)."""
    name = fc.get("name", "unknown")
    args = fc.get("arguments", {}) or {}

    if name == "retrieve_assets":
        ent = args.get("entity_name", "?")
        tk = args.get("top_k")
        return f"🔍 retrieve: {ent}" + (f"  (top_k={tk})" if tk else "")

    if name == "add":
        items = args.get("modified_data", []) or []
        return f"➕ add {len(items)}: {_names_from(items)}"

    if name == "delete":
        items = args.get("modified_data", []) or []
        return f"➖ delete {len(items)}: {_names_from(items)}"

    if name == "rotation_and_translation":
        corr = args.get("corrections", []) or []
        names = _names_from(
            [c.get("original_data", {}) for c in corr if isinstance(c, dict)]
        )
        return f"🔄 adjust {len(corr)}: {names}"

    # Unknown tool: compact single line.
    return f"{name}: {json.dumps(args, ensure_ascii=False)[:80]}"


def on_reasoning(turn: int, reasoning: str, tool_calls: list):
    # A streaming turn already printed its reasoning live; session.py passes an
    # empty string in that case, and we only need to close the open block here
    # so the ⎿ tool lines below start on a fresh line.
    _stream_end()
    if reasoning:
        # Non-streaming models (gemini / gpt4o) deliver the whole block at once.
        # Route it through the same writer as the streaming path so both look
        # identical: ⎿ elbow, hanging indent, soft gray.
        _STREAM["buf"] = reasoning.strip()
        _stream_flush(final=True)
        _STREAM["started"] = False

    color_map = {"retrieve_assets": "blue", "add": "green",
                 "delete": "red", "rotation_and_translation": "magenta"}
    for fc in tool_calls or []:
        _branch(_summarize_tool_call(fc),
                style=color_map.get(fc.get("name", ""), "white"))


def on_rendered(turn: int, images: list):
    _stream_end()
    if not images:
        return
    _branch(f"🖼  rendered · {len(images)} views", style="green")
    labels = ["left", "right", "front", "back", "top"]
    for i, p in enumerate(images[:5]):
        label = labels[i] if i < len(labels) else f"view{i+1}"
        # Raw ANSI, not console.print: rich would wrap these long paths at the
        # terminal width and break the click-to-open target across two lines.
        if _REAL_TTY:
            print(f"       {_G_TEXT}{label:<6}{p}{_G_OFF}")
        else:
            print(f"       {label:<6}{p}")
    sys.stdout.flush()


# ── Streaming: print reasoning / content as it arrives ─────────────────────────
# Streaming clients (models with `streaming = True`) call this per delta.
#
# IMPORTANT — emit WHOLE LINES ONLY, never a partial line.
# The query runs on a background thread while prompt_toolkit keeps a live prompt
# at the bottom; its stdout proxy erases the prompt, writes, then re-draws. A
# partial line gets wiped by that redraw, so with per-token writes only the tail
# of each line survived — that's the "跳字" (text appeared to jump / lose its
# head). Full-line writes always rendered fine (the ● / ⎿ lines never broke), so
# we buffer deltas and hand out complete lines: on "\n", or once the pending text
# fills the terminal width (wrap at the last space, hard-cut for CJK).
#
# Styling is safe *because* we emit whole lines: the block gets a ⎿ elbow on its
# first line, a hanging indent on the rest, and soft gray text — same visual
# language as the `🖼 rendered · 5 views` block, so reasoning reads as secondary
# instead of a wall of plain white.
_STREAM = {"buf": "", "started": False}

_G_TEXT = "\033[38;5;245m"    # soft gray for reasoning prose
_G_ELBOW = "\033[38;5;240m"   # slightly darker gray for the ⎿ elbow
_G_OFF = "\033[0m"
_STREAM_INDENT = 4            # "  ⎿ " on line 1, 4 spaces on continuations


def _term_width() -> int:
    try:
        return max(40, console.size.width if (_RICH and console)
                   else shutil.get_terminal_size((80, 24)).columns)
    except Exception:
        return 80


def _break_at(s: str, width: int) -> int:
    """Index at which to cut `s` so the piece occupies at most `width` columns.
    Returns 0 if the whole string fits. CJK chars count as 2 columns, so wrapping
    on character count alone would emit lines twice the terminal width."""
    cols = 0
    for i, ch in enumerate(s):
        cols += 2 if unicodedata.east_asian_width(ch) in "WF" else 1
        if cols > width:
            return i or 1
    return 0


def _stream_line(line: str):
    """Write one complete reasoning line: ⎿ elbow on the first line of a block,
    hanging indent after, gray throughout. Raw ANSI rather than console.print so
    rich can't re-wrap the text or eat a `[...]` in the reasoning as markup."""
    if not line.strip():
        print()                       # blank separator: no indent, no SGR
        sys.stdout.flush()
        return
    _turn_header()
    first, _STREAM["started"] = not _STREAM["started"], True
    if _REAL_TTY:
        head = f"  {_G_ELBOW}⎿{_G_OFF} " if first else "    "
        print(f"{head}{_G_TEXT}{line}{_G_OFF}")
    else:
        print(f"  ⎿ {line}" if first else f"    {line}")
    sys.stdout.flush()


def _stream_flush(final: bool = False):
    buf = _STREAM["buf"]
    width = max(20, _term_width() - _STREAM_INDENT)
    while True:
        nl = buf.find("\n")
        if nl >= 0:
            _stream_line(buf[:nl])
            buf = buf[nl + 1:]
            continue
        hard = _break_at(buf, width)
        if hard:
            cut = buf.rfind(" ", 0, hard)
            if cut <= 0:                      # no space to break on (CJK) → hard cut
                _stream_line(buf[:hard])
                buf = buf[hard:]
            else:
                _stream_line(buf[:cut])
                buf = buf[cut + 1:]
            continue
        break
    if final and buf:
        _stream_line(buf)
        buf = ""
    _STREAM["buf"] = buf


def _stream_end():
    """Flush the trailing partial line (called before tool / render lines)."""
    _stream_flush(final=True)
    _STREAM["started"] = False


def on_stream_delta(kind: str, text: str):
    if not text:
        return
    _STREAM["buf"] += text
    _stream_flush()


CALLBACKS = {"on_turn_start": on_turn_start,
             "on_reasoning": on_reasoning,
             "on_stream_delta": on_stream_delta,
             "on_rendered": on_rendered}


# ── Background query runner: submit → input box returns IMMEDIATELY ────────────
# The query runs on a daemon thread while the REPL keeps a live prompt at the
# bottom (Claude-Code style). Output streams ABOVE the prompt via patch_stdout,
# so a slow / stuck LLM retry never freezes the input box.
#
# Concurrency: the foreground NEVER blocks on the worker. Each session has a run
# lock and a generation counter. Submitting a query bumps the generation and
# signals cancel; the new worker waits on the lock in the BACKGROUND for the old
# one to reach its next checkpoint, then runs only if it's still the latest
# generation (older/superseded submissions are dropped). Commands that mutate
# session state use _exclusive() to pause the running query first.

def _run_lock(session):
    lk = getattr(session, "_run_lock", None)
    if lk is None:
        lk = threading.Lock()
        session._run_lock = lk
    return lk


def _bump_gen(session) -> int:
    g = getattr(session, "_query_gen", 0) + 1
    session._query_gen = g
    return g


def start_query(session, query: str):
    """Kick off a query in the background and return at once (non-blocking)."""
    session.request_cancel()          # ask any running worker to stop at its checkpoint
    gen = _bump_gen(session)          # invalidate any older pending worker
    lock = _run_lock(session)

    def _worker():
        with lock:                    # wait (in background) for the prior worker to finish
            if getattr(session, "_query_gen", 0) != gen:
                return                # superseded by a newer submission / command
            session.reset_cancel()
            session._query_t0 = time.monotonic()   # drives the input box's Thinking… timer
            try:
                session.run_query(query)
            except Exception as e:
                _print(f"[red]Error: {e}[/]" if _RICH else f"Error: {e}")
            finally:
                session._query_t0 = None

    threading.Thread(target=_worker, daemon=True).start()


def _query_running(session) -> bool:
    return _run_lock(session).locked()


def cancel_query(session):
    """Signal the running query to stop (non-blocking) and drop pending ones."""
    session.request_cancel()
    _bump_gen(session)


@contextlib.contextmanager
def _exclusive(session, timeout: float = 3.0):
    """Pause a running query and hold the run lock so a command can safely mutate
    session state. Cancels + invalidates the current query, waits briefly for its
    in-flight step to finish, then proceeds (applying anyway if it's stuck)."""
    session.request_cancel()
    _bump_gen(session)
    lock = _run_lock(session)
    got = lock.acquire(timeout=timeout)
    if not got:
        _print("[dim]…the current request is still finishing; applying anyway[/]"
               if _RICH else "…the current request is still finishing; applying anyway")
    try:
        yield
    finally:
        if got:
            lock.release()


# ── Arrow-key picker for /model (↑/↓ to move, Enter to pick, Esc to cancel) ────
def _pick_model(models_list, current_key):
    """Inline arrow-key selector. Returns the chosen model key, or None if the
    picker is unavailable (caller falls back to a plain list) or cancelled."""
    try:
        if not sys.stdin.isatty():
            return None
        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import HSplit, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
    except Exception:
        return None

    items = list(models_list)  # [(key, mname, desc), ...]
    sel = next((i for i, it in enumerate(items) if it[0] == current_key), 0)
    state = {"sel": sel, "chosen": None}

    def _render():
        rows = [("class:pick.title", "Select a model  "),
                ("class:pick.hint", "(↑/↓ · Enter · Esc)\n")]
        for i, (key, mname, desc) in enumerate(items):
            cur = " ●" if key == current_key else "  "
            if i == state["sel"]:
                rows.append(("class:pick.cursor", " ❯ "))
                rows.append(("class:pick.sel", f"{key}{cur}  "))
                rows.append(("class:pick.sel.dim", f"{mname}  {desc}\n"))
            else:
                rows.append(("", "   "))
                rows.append(("class:pick.key", f"{key}{cur}  "))
                rows.append(("class:pick.dim", f"{mname}  {desc}\n"))
        return rows

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    def _(e): state["sel"] = (state["sel"] - 1) % len(items)

    @kb.add("down")
    @kb.add("c-n")
    def _(e): state["sel"] = (state["sel"] + 1) % len(items)

    @kb.add("enter")
    def _(e):
        state["chosen"] = items[state["sel"]][0]
        e.app.exit()

    @kb.add("escape")
    @kb.add("c-c")
    @kb.add("q")
    def _(e): e.app.exit()

    from prompt_toolkit.styles import Style
    style = Style.from_dict({
        "pick.title": "bold", "pick.hint": "#9a9484",
        "pick.cursor": "#7f9b6e bold", "pick.sel": "#7f9b6e bold",
        "pick.sel.dim": "#7f9b6e", "pick.key": "bold", "pick.dim": "#9a9484",
    })
    app = Application(
        layout=Layout(HSplit([Window(FormattedTextControl(_render), wrap_lines=True)])),
        key_bindings=kb, style=style, mouse_support=False, full_screen=False,
    )
    app.run()
    return state["chosen"]


def _apply_model(session, key_or_name):
    """Switch the session model and report it. Shared by /model <name> and picker."""
    from vibeworld import models
    try:
        key, mname = session.switch_model(key_or_name)
        shown = models.display_name(mname)
        session.viewer.set_model(shown)
        _print(f"[green]✓ Switched to {key} ({shown}); scene state carried over.[/]" if _RICH
               else f"Switched to {key} ({shown}).")
    except KeyError as e:
        _print(f"[red]Unknown model: {key_or_name} ({e}). Run /model to list options.[/]" if _RICH
               else f"Unknown model: {key_or_name}")


# ── slash command handling ─────────────────────────────────────────────────────
def handle_command(session_box: list, line: str, viewer, session_dir: str,
                   quality: str, max_turns: int, callbacks: dict) -> str:
    """Handle a /-command. Returns 'quit' / 'handled'.
    session_box = [session]; /refine may replace the session object in it.
    """
    session = session_box[0]
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit"):
        return "quit"

    if cmd == "/help":
        _print_help()
        return "handled"

    if cmd == "/clear":
        with _exclusive(session):
            session.clear()
        _print("[green]✓ Scene and conversation cleared.[/]" if _RICH else "Cleared.")
        return "handled"

    if cmd == "/compact":
        with _exclusive(session):
            summary = session.compact()
        _panel(summary[:800] + ("..." if len(summary) > 800 else ""),
               title="🗜  History compacted (scene state kept)", style="yellow")
        return "handled"

    if cmd == "/model":
        from vibeworld import models
        if not arg:
            # Arrow-key picker (↑/↓ · Enter · Esc). Falls back to a plain list
            # if prompt_toolkit isn't available or stdin isn't a TTY.
            chosen = _pick_model(models.list_models(), session.model_key)
            if chosen is None:
                _print("\n[bold]Available models:[/]" if _RICH else "Available models:")
                for key, mname, desc in models.list_models():
                    mark = " [green](current)[/]" if key == session.model_key else ""
                    _print(f"  [cyan]{key}[/]  [dim]{mname}[/]  {desc}{mark}" if _RICH
                           else f"  {key}  {mname}  {desc}")
                _print(f"\nUsage: [bold]/model <name>[/]" if _RICH else "Usage: /model <name>")
            elif chosen != session.model_key:
                with _exclusive(session):
                    _apply_model(session, chosen)
            return "handled"
        with _exclusive(session):
            _apply_model(session, arg)
        return "handled"

    if cmd == "/refine":
        if not arg:
            _print("[red]Usage: /refine <data directory>[/]" if _RICH
                   else "Usage: /refine <data directory>")
            return "handled"
        data_dir = os.path.expanduser(arg.strip())
        if not os.path.isdir(data_dir):
            _print(f"[red]No such directory: {data_dir}[/]" if _RICH else f"No such directory: {data_dir}")
            return "handled"
        for required in ("init_map.json", "component_info.json", "query.json"):
            if not os.path.exists(os.path.join(data_dir, required)):
                _print(f"[red]Missing required file: {required} (not a valid refine data dir)[/]"
                       if _RICH else f"Missing: {required}")
                return "handled"
        try:
            from vibeworld.session import RefineSession
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            case_name = os.path.basename(data_dir.rstrip("/"))
            refine_session_dir = os.path.join(session_dir, f"refine_{case_name}_{ts}")
            os.makedirs(refine_session_dir, exist_ok=True)
            # Stop the outgoing session's query before swapping it out.
            with _exclusive(session):
                new_session = RefineSession(
                    data_dir=data_dir,
                    session_dir=refine_session_dir,
                    viewer=viewer,
                    model_name=session.model_key,
                    quality=quality,
                    max_turns_per_query=max_turns,
                    callbacks=callbacks,
                )
                session_box[0] = new_session
            _print(f"[bold green]✓ Switched to Refine mode: {case_name}[/]" if _RICH
                   else f"Switched to Refine mode: {case_name}")
            desc = new_session.description
            if desc:
                _print(f"[dim]Scene query: {desc}[/]" if _RICH else f"Scene query: {desc}")
                _print("[dim]Press Enter to use this query, or type a new one to override.[/]"
                       if _RICH else "Press Enter to use this query, or type a new one to override.")
        except Exception as e:
            _print(f"[red]Refine init failed: {e}[/]" if _RICH else f"Refine init failed: {e}")
        return "handled"

    _print(f"[red]Unknown command: {cmd}. Type /help.[/]" if _RICH else f"Unknown command: {cmd}")
    return "handled"


def _print_help():
    lines = [
        ("/model [name]", "Switch model; no arg lists options"),
        ("/refine <dir>", "Enter Refine mode (load init_map/component_info/query.json)"),
        ("/clear", "Clear the scene and conversation"),
        ("/compact", "Compact history (keep scene state)"),
        ("/help", "Show this help"),
        ("/quit, /exit", "Exit"),
        ("Ctrl+C", "During a turn → interrupt it; twice at an empty prompt → exit"),
    ]
    if _RICH and console:
        t = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 2))
        t.add_column("Command"); t.add_column("Description")
        for c, d in lines:
            t.add_row(c, d)
        console.print(Panel(t, title="[bold]Commands[/]", border_style="blue"))
    else:
        for c, d in lines:
            print(f"  {c:<16} {d}")


# ── Startup welcome box (two-column, Claude-Code style) ────────────────────────
# The left icon is the project's own Minecraft wooden-pickaxe (icon.png), baked
# into pickaxe_art.py as truecolor half-block ANSI so it renders identically to
# the texture with no runtime image dependency. Falls back to a plain silhouette
# on no-color / non-rich terminals.
_PICK_WOOD = "rgb(150,100,55)"
try:
    from .pickaxe_art import PICKAXE_ANSI, PICKAXE_PLAIN
except Exception:
    PICKAXE_ANSI, PICKAXE_PLAIN = [], ["  /|", " / |", "/__|"]


def _pickaxe_lines():
    """Return the pickaxe icon as a list of rich Text lines, each padded to equal
    cell width so the column's center-justify keeps the sprite's shape intact."""
    if PICKAXE_ANSI:
        lines = [Text.from_ansi(l) for l in PICKAXE_ANSI]
    else:
        w = max((len(l) for l in PICKAXE_PLAIN), default=1)
        lines = [Text(l.ljust(w), style=_PICK_WOOD) for l in PICKAXE_PLAIN]
    w = max((l.cell_len for l in lines), default=0)
    for l in lines:
        l.pad_right(w - l.cell_len)
    return lines


def _welcome_box(model_key, model_name, quality, url):
    """A single rounded app box: left = Welcome + pickaxe art + model/viewer;
    right = tips for getting started + what's new."""
    if not (_RICH and console):
        print(f"VibeWorld v{VERSION}")
        print("Welcome back!")
        print("Tips: describe a scene to build your first 3D world.")
        print("What's new: check the HTML viewer for your constructed 3D world.")
        print(f"Model {model_name} · quality {quality}")
        print(f"Viewer {url}")
        return

    left_rows = [
        ("", ""),
        ("Welcome back!", "bold"),
        ("", ""),
        *_pickaxe_lines(),
        ("", ""),
        (f"{model_name} · quality {quality}", "dim"),
        (url, "cyan"),
    ]
    right_rows = [
        ("Tips for getting started", "bold"),
        ("Describe a scene to build your first 3D world", "dim"),
        ("─" * 42, "grey37"),
        ("What's new", "bold"),
        ("Check the HTML viewer for your constructed 3D world", "dim"),
    ]
    n = max(len(left_rows), len(right_rows))

    def _col(rows):
        t = Text()
        for i in range(n):
            item = rows[i] if i < len(rows) else ("", "")
            if isinstance(item, Text):
                t.append_text(item)
            else:
                s, st = item
                t.append(s, style=st)
            if i < n - 1:
                t.append("\n")
        return t

    grid = Table.grid(expand=True)
    grid.add_column(ratio=42, justify="center")
    grid.add_column(width=3, justify="center")
    grid.add_column(ratio=58, justify="left")
    divider = Text("\n".join("│" for _ in range(n)), style="grey37")
    grid.add_row(_col(left_rows), divider, _col(right_rows))

    console.print(Panel(
        grid,
        title=f"[bold cyan]VibeWorld v{VERSION}[/]",
        subtitle="[dim]describe a scene · type [/][cyan]/[/][dim] for commands · Ctrl+C twice to exit[/]",
        border_style="cyan", box=_rich_box.ROUNDED, padding=(1, 1),
    ), highlight=False)


# ── Input: prompt_toolkit with live completion; falls back to input() ──────────
def _make_prompt_session():
    """Build a prompt_toolkit PromptSession with live slash-command completion.

    Returns None if prompt_toolkit is unavailable or stdin isn't a TTY, in which
    case the REPL falls back to plain input() (+ readline if present).
    """
    try:
        if not sys.stdin.isatty():
            return None
    except Exception:
        return None
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import Completer, Completion, PathCompleter
        from prompt_toolkit.document import Document
        from prompt_toolkit.styles import Style
        from prompt_toolkit.history import InMemoryHistory
    except ImportError:
        return None

    path_completer = PathCompleter(expanduser=True)

    class _VWCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            # Completing a command name: "/" with no space yet.
            if text.startswith("/") and " " not in text:
                for cmd, desc in SLASH_COMMANDS.items():
                    if cmd.startswith(text):
                        yield Completion(cmd, start_position=-len(text),
                                         display=cmd, display_meta=desc)
            # Completing a directory path after "/refine ".
            elif text.startswith("/refine "):
                sub = text[len("/refine "):]
                sub_doc = Document(sub, cursor_position=len(sub))
                yield from path_completer.get_completions(sub_doc, complete_event)

    style = Style.from_dict({
        "arrow.generate": "#22c55e bold",
        "arrow.refine": "#38bdf8 bold",
        "think.mark": "#c084fc",
        "think": "#8b95a5",
        "completion-menu.completion": "bg:#1e293b #e2e8f0",
        "completion-menu.completion.current": "bg:#38bdf8 #0f172a bold",
        "completion-menu.meta.completion": "bg:#0b1120 #94a3b8",
        "completion-menu.meta.completion.current": "bg:#38bdf8 #0f172a",
    })
    return PromptSession(completer=_VWCompleter(),
                         complete_while_typing=True,
                         history=InMemoryHistory(),
                         style=style)


def _read_line(ptk_session, is_refine: bool, session=None) -> str:
    """Read one line of input. Uses prompt_toolkit (live completion) if available,
    otherwise a plain input() with an ANSI-colored arrow.

    The prompt_toolkit prompt runs under patch_stdout so a background query's
    output scrolls ABOVE the live input box instead of clobbering it."""
    if ptk_session is not None:
        from prompt_toolkit.formatted_text import FormattedText
        from prompt_toolkit.patch_stdout import patch_stdout
        cls = "class:arrow.refine" if is_refine else "class:arrow.generate"

        def _rprompt():
            # Live "✳ Thinking… (48s)" on the right of the input box while the
            # background query runs. prompt_toolkit repaints this inside its own
            # render loop (refresh_interval below), so — unlike animating a line
            # ABOVE the prompt — it can never race patch_stdout's erase/redraw.
            t0 = getattr(session, "_query_t0", None) if session is not None else None
            if t0 is None or not _query_running(session):
                return []
            return FormattedText([("class:think.mark", "✳ "),
                                  ("class:think", f"Thinking… ({int(time.monotonic() - t0)}s)")])

        with patch_stdout(raw=True):
            return ptk_session.prompt(FormattedText([(cls, "> ")]),
                                      rprompt=_rprompt, refresh_interval=0.5)
    # Fallback: input(). ANSI color wrapped in \001..\002 so readline computes
    # visible width correctly (otherwise backspace/cursor math is off).
    if _RICH and console:
        GREEN, CYAN, RESET = "\001\033[1;32m\002", "\001\033[1;36m\002", "\001\033[0m\002"
        arrow = f"{CYAN}>{RESET} " if is_refine else f"{GREEN}>{RESET} "
    else:
        arrow = "> "
    return input(arrow)


def _setup_readline_fallback():
    """Tab completion for the input() fallback path (when prompt_toolkit is
    unavailable). No-op if readline isn't present."""
    if not _HAS_READLINE:
        return
    import glob

    def _path_matches(token):
        token = os.path.expanduser(token)
        out = []
        for m in glob.glob(token + "*"):
            out.append(m + ("/" if os.path.isdir(m) else " "))
        return out

    def _completer(text, state):
        buf = readline.get_line_buffer().lstrip()
        if buf.startswith("/") and " " not in buf:
            opts = [c + " " for c in SLASH_COMMANDS if c.startswith(text)]
        elif buf.startswith("/refine "):
            opts = _path_matches(text)
        else:
            opts = []
        return opts[state] if state < len(opts) else None

    readline.set_completer(_completer)
    readline.set_completer_delims(" \t\n")
    try:
        readline.parse_and_bind("set show-all-if-ambiguous on")
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        prog="vibeworld",
        description="AI-driven 3D world construction (interactive REPL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              vibeworld
              vibeworld --query "build a Japanese garden with cherry trees and lanterns"
              vibeworld --demo 1
              vibeworld --model gpt4o --quality medium
        """),
    )
    parser.add_argument("--query", "-q", type=str, default=None, help="query to run at startup")
    parser.add_argument("--demo", type=int, choices=[1, 2, 3], default=None, help="preset demo query")
    parser.add_argument("--max-turns", type=int, default=8, help="max reasoning turns per query")
    parser.add_argument("--quality", choices=["low", "medium", "high"], default="low",
                        help="render quality (default low = fastest)")
    parser.add_argument("--session-dir", type=str, default=None, help="session directory")
    parser.add_argument("--model", type=str, default=None,
                        help="initial model (vibeworlder/gemini-flash/gemini-pro/gpt5/qwen/k3, "
                             "default qwen); run /model in the REPL for the full list")
    parser.add_argument("--server", type=str, default=None,
                        help="PCG render server (overrides setup.py / env var)")
    parser.add_argument("--retrieve-server", type=str, default=None,
                        help="asset retrieve server (overrides setup.py / env var)")
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    args = parser.parse_args()

    # ── Service addresses ────────────────────────────────────────────────────
    # Priority: --server / --retrieve-server > env var > setup.py constant.
    # Configured centrally in setup.py — no interactive prompt at startup.
    cfg_render, cfg_retrieve = _load_service_config()
    render_server = (args.server or os.environ.get("VIBEWORLD_RENDER_SERVER") or cfg_render)
    retrieve_server = (args.retrieve_server or os.environ.get("VIBEWORLD_RETRIEVE_SERVER")
                       or cfg_retrieve)
    # Set env vars BEFORE importing session (session.py reads them at import time).
    os.environ["VIBEWORLD_RENDER_SERVER"] = render_server
    os.environ["VIBEWORLD_RETRIEVE_SERVER"] = retrieve_server

    # ── Session directory ────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = args.session_dir or f"/tmp/vibeworld_sessions/{ts}"
    os.makedirs(session_dir, exist_ok=True)
    html_path = os.path.join(session_dir, "viewer.html")

    # ── Quiet the loggers ────────────────────────────────────────────────────
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    for n in ("llm", "pcg_render", "httpx", "httpcore", "glb_builder"):
        logging.getLogger(n).setLevel(logging.WARNING)

    # ── Initialize viewer + session behind a transient spinner ───────────────
    # The first load is slow (heavy ML deps). We show only an animated spinner
    # and capture the noisy import output; on success it's discarded so the very
    # first thing the user sees is the welcome box.
    from vibeworld import models
    from vibeworld.html_viewer import HtmlViewer
    from vibeworld.session import Session

    session = None
    init_log = io.StringIO()
    real_out = sys.stdout

    def _init():
        nonlocal session
        viewer = HtmlViewer(html_path, query="", model_name="")
        if args.no_browser:
            viewer._opened = True
        session = Session(
            session_dir=session_dir, viewer=viewer,
            model_name=args.model or models.DEFAULT_MODEL_KEY,
            quality=QUALITY_MAP[args.quality],
            max_turns_per_query=args.max_turns, callbacks=CALLBACKS,
        )
        return viewer

    try:
        if _RICH:
            # Spinner console bound to the REAL stdout so it stays visible while
            # we redirect the init's own stdout/stderr into a buffer.
            spin_console = Console(file=real_out)
            with spin_console.status("[cyan]Initializing[/] [dim](first load may take a while)…[/]",
                                     spinner="dots"):
                with contextlib.redirect_stdout(init_log), contextlib.redirect_stderr(init_log):
                    viewer = _init()
        else:
            print("Initializing (first load may take a while)…")
            viewer = _init()
    except Exception as e:
        # Surface the captured init log on failure so the error isn't swallowed.
        tail = init_log.getvalue()[-2000:]
        if tail.strip():
            _print(tail)
        _print(f"[red]Initialization failed: {e}[/]" if _RICH else f"Initialization failed: {e}")
        sys.exit(1)

    viewer.set_model(models.display_name(session.model_name))

    # ── Welcome box (the first thing shown) ──────────────────────────────────
    # Banner always shows our own model name, regardless of which provider the
    # session actually calls (session.model_name); switch with /model in the REPL.
    _welcome_box(session.model_key, "vibeworlder/vibeworlder-30B-A3B",
                 args.quality, viewer._url)

    # ── Startup query (--demo / --query) ─────────────────────────────────────
    initial_query = None
    if args.demo:
        initial_query = DEMO_QUERIES[args.demo - 1]
    elif args.query:
        initial_query = args.query.strip()

    if initial_query:
        viewer.set_query(initial_query, models.display_name(session.model_name))
        _hr()
        _print(f"[bold green]>[/] {initial_query}" if _RICH else f"> {initial_query}")
        _hr()
        start_query(session, initial_query)

    # ── REPL main loop ───────────────────────────────────────────────────────
    ptk_session = _make_prompt_session()   # live completion (or None → input())
    if ptk_session is None:
        _setup_readline_fallback()

    # session_box = [session]: list wrapper for a mutable reference; /refine
    # replaces session_box[0], and the loop reads the latest via session_box[0].
    session_box = [session]

    pending_sigint = False  # last Ctrl+C was at an empty prompt; one more exits

    while True:
        session = session_box[0]
        is_refine = hasattr(session, "data_dir")

        try:
            print()  # blank line to separate from the previous turn's output
            _hr("refine" if is_refine else "")   # input box top edge
            line = _read_line(ptk_session, is_refine, session).strip()
            _hr()                                 # input box bottom edge
        except EOFError:
            _print("\nGoodbye.")
            break
        except KeyboardInterrupt:
            # Ctrl+C while a query is running → cancel it, stay at the prompt.
            if _query_running(session):
                cancel_query(session)
                _print("\n[yellow]⏸  Stopping…[/]" if _RICH else "\nStopping…")
                pending_sigint = False
                continue
            if pending_sigint:
                _print("\nGoodbye.")
                break
            pending_sigint = True
            _print("\n[dim](press Ctrl+C again to exit, or type /quit)[/]"
                   if _RICH else "\n(press Ctrl+C again to exit, or type /quit)")
            continue

        # Any input resets the double-Ctrl+C-to-exit state.
        pending_sigint = False

        if line.startswith("/"):
            if handle_command(session_box, line, viewer, session_dir,
                              QUALITY_MAP[args.quality], args.max_turns,
                              CALLBACKS) == "quit":
                _print("Goodbye.")
                break
            continue

        # Empty line: in refine mode Enter runs the description; ignored otherwise.
        if not line:
            if is_refine and not session.started and session.description:
                _print(f"[dim]Using query: {session.description}[/]" if _RICH
                       else f"Using query: {session.description}")
                viewer.add_query(session.description)
                start_query(session, session.description)
            continue

        # Normal query — runs in the background; the input box returns at once.
        viewer.add_query(line)
        start_query(session, line)


if __name__ == "__main__":
    main()
