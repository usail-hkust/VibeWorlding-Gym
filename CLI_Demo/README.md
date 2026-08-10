# VibeWorld CLI

An interactive terminal agent for building and editing 3D worlds. You describe a
world in natural language; the agent retrieves real assets, places them, renders
the scene with Blender, looks at the result, and iterates — all streamed live in
your terminal, with a browser-based 3D viewer alongside.

Two modes: **Generate** (build from an empty map) and **Refine** (edit an
existing scene). The HTML viewer shows the 5-view renders per turn on top and the
latest interactive 3D scene below (auto-rotating, drag to orbit / zoom).

```
vibeworld
> 一个雪山环绕的温泉祭坛
```

## 1. Install

```bash
pip install -e CLI_Demo
```

This installs the `vibeworld` console command (Rich + prompt_toolkit REPL).
Dependencies: `rich prompt_toolkit openai google-genai gradio-client httpx trimesh numpy pillow`.

## 2. Start the backing services

The CLI needs both services running — see `assets_retrieval/README.md` and
`render_in_blender/README.md`:

```bash
export VIBEWORLD_RETRIEVE_SERVER=http://localhost:8081
export VIBEWORLD_RENDER_SERVER=http://localhost:8080
```

Also make sure the GLB assets are in place, otherwise scenes render empty.

## 3. Pick a model

All providers stream token-by-token (reasoning + content). Set the matching key:

| `/model` key | Model | Provider | Credential |
|---|---|---|---|
| `vibeworlder` | vibeworlder/vibeworlder-30B-A3B | local vLLM (**ours**) | — |
| `gemini-flash` | gemini/gemini-3.5-flash | Google official | `GEMINI_API_KEY` |
| `gemini-pro` | gemini/gemini-3.1-pro | Google official | `GEMINI_API_KEY` |
| `gpt5` | openai/gpt-5.5 | OpenAI official | `OPENAI_API_KEY` |
| `qwen` *(default)* | bailian/qwen3.8-max | Bailian / DashScope | `DASHSCOPE_API_KEY` |
| `k3` | bailian/K3 | Bailian / DashScope | `DASHSCOPE_API_KEY` |

Model references: [Gemini](https://ai.google.dev/gemini-api/docs/models),
[OpenAI](https://developers.openai.com/api/docs/models),
[Bailian / DashScope](https://bailian.console.aliyun.com).

The default model on first launch is `qwen` (Bailian), so set
`DASHSCOPE_API_KEY` before running `vibeworld` if you don't pass `--model`:

```bash
export DASHSCOPE_API_KEY=your_dashscope_api_key
vibeworld
```

### Using our trained models

Download from <https://huggingface.co/collections/usail-hkust/vibeworlder>, then
serve them locally:

```bash
huggingface-cli download usail-hkust/VibeWorlder-30B-A3B \
  --local-dir ./models/VibeWorlder-30B-A3B

# on the GPU machine
python start_verl_server_CLI.py \
  --model_path ./models/VibeWorlder-30B-A3B \
  --tp_size 4 --port 8000

export VIBEWORLD_LOCAL_VLLM_URL=http://localhost:8000/v1
```

`start_verl_server_CLI.py` sets the flags the CLI relies on: native Qwen tool
calling, `<think>` reasoning parsing (so reasoning streams separately from the
answer), and a 5-image-per-prompt cap matching the 5 rendered views.

### Using an API model instead

```bash
export GEMINI_API_KEY=your_gemini_api_key
vibeworld --model gemini-flash
```

```bash
vibeworld                              # interactive REPL
vibeworld -q "一个雪山环绕的温泉祭坛"    # run one query, then drop into the REPL
vibeworld --demo 1                     # preset demo query
vibeworld --model gemini-pro --quality low # choose model / render quality
```

### REPL commands

| Command | Description |
|---|---|
| `/model [name]` | Switch model; no argument lists all options |
| `/refine <dir>` | Refine mode: load an existing case (`init_map.json` + `component_info.json` + `query.json`) |
| `/clear` | Clear the scene and conversation |
| `/compact` | Compact history, keeping scene state |
| `/help` | Show command help |
| `/quit`, `/exit` | Exit |

Switching models keeps the current scene — only the client changes.

### Refine mode

Point it at any case directory from `data/test` to edit an existing world:

```
> /refine data/test/004
```

## 5. CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--query`, `-q` | — | Query to run at startup |
| `--demo {1,2,3}` | — | Preset demo query |
| `--model` | `qwen` | Initial model |
| `--max-turns` | — | Max reasoning turns per query |
| `--quality {low,medium,high}` | `low` | Render quality (low is fastest) |
| `--server` | env / `localhost:8080` | PCG render server |
| `--retrieve-server` | env / `localhost:8081` | Asset retrieval server |
| `--session-dir` | auto | Session output directory |
| `--no-browser` | off | Don't auto-open the 3D viewer |

Precedence for service addresses: CLI flag > environment variable > `setup.py`.

## 6. Output

Each session writes to `--session-dir` (default under the system temp dir):

```
<session>/
├── index.html            # live 3D viewer (auto-refreshes)
├── final_map.json        # the 3D world map
├── scene.glb             # assembled GLB scene
└── image/                # 5-view renders per turn
```

## 7. Code layout

| File | Role |
|---|---|
| `vibeworld/cli.py` | REPL entry point (`vibeworld` command), arg parsing, commands |
| `vibeworld/session.py` | Agent loop: tool dispatch, render, history management |
| `vibeworld/models.py` | Model registry for `/model` |
| `vibeworld/streaming_bailian.py` | The 4 streaming clients (all OpenAI-compatible) |
| `vibeworld/glb_builder.py` | Assemble a GLB scene from the 3D map |
| `vibeworld/html_viewer.py` | Browser 3D viewer |
| `setup.py` | Install entry point + service address config |

`qwen_stream_demo.py` is a standalone single-file streaming demo, useful for
debugging provider connectivity without the full REPL.
