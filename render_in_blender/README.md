# PCG Rendering Service (Blender)

Blender-based rendering service for VibeWorld-Gym. It turns a **PCG scene JSON**
(the agent's 3D world map) into rendered images, so the agent can *see* the
world it just built and iterate on it. This visual feedback loop is what makes
multi-turn 3D world refinement possible.

Served as a Gradio app; both `main.py` and `render_raw_data_images.py` talk to
it through `gradio_client`.

---

## 1. Install Blender 4.2.x

```bash
wget https://download.blender.org/release/Blender4.2/blender-4.2.0-linux-x64.tar.xz
tar -xf blender-4.2.0-linux-x64.tar.xz -C /opt
export BLENDER_EXE=/opt/blender-4.2.0-linux-x64/blender
```

Rendering uses Blender's **Cycles** engine. A CUDA GPU is strongly recommended.

## 2. Python dependencies

```bash
pip install -r requirements.txt
pip install gradio==4.16.0 gradio_client aiohttp
```

`aiohttp` is only needed for the multi-worker proxy (`session_proxy.py`).
Blender ships its own Python, so `render_scene.py` does not need `bpy` installed
in your environment.

## 3. The 3D assets

**The GLB assets ship in this repo** at `assets/models/clone/` (2,617 assets,
~71 GB) — nothing to download. Layout is one directory per 5-digit `type_id`:

```
assets/models/clone/
├── 00001/00001.glb
├── 00443/00443.glb
└── ...
```

Point `VIBEWORLD_MODELS_DIR` elsewhere if you keep assets outside the repo. The
same assets are published in **VWE-Bench**
(<https://huggingface.co/datasets/usail-hkust/VWE-Bench>).

Supporting metadata (also included):

| File | Purpose |
|---|---|
| `assets/item_infos.json` | `type_id` → display name / metadata |
| `assets/glb_corrections.json` | Per-asset scale & orientation corrections |
| `assets/cmds/1/` | One example scene, used for the smoke test below |

## 4. Start the service

### Quick smoke test — one worker

```bash
export BLENDER_EXE=/opt/blender-4.2.0-linux-x64/blender
WORKERS=1 PORT=8080 bash deploy.sh
```

This runs a single worker in the **foreground** (your terminal stays attached).
Fine for verifying the install, but one worker renders **serially** — each
request waits for the previous Blender process to finish.

### Real deployment — many workers (required for RL training)

Rendering is GPU-bound (Cycles with CUDA/OptiX), and one Blender render does not
saturate a GPU, so throughput comes from running several workers **per GPU**:

```bash
export BLENDER_EXE=/opt/blender-4.2.0-linux-x64/blender
WORKERS_PER_GPU=8 PORT=8080 bash deploy.sh
```

On an 8-GPU node that starts **64 workers** on ports `7000-7063`, each pinned to
one GPU via `CUDA_VISIBLE_DEVICES`, all fronted by `session_proxy.py` on `:8080`.
Clients keep talking to a single address — `PORT` is the only one they need.

```
   client                ┌──────────────────────────────────────┐
 (agent / RL) ──:8080──► │ session_proxy.py (session-sticky LB) │
                         └──┬───────┬───────┬───────────────────┘
                :7000  │ :7001 │  ...  │ :7063
                    gpu0_0  ▼gpu0_1 ▼       ▼ gpu7_7
                        ┌──────┐┌──────┐ ┌──────┐   each worker:
                        │worker││worker│…│worker│   own output dir +
                        └──────┘└──────┘ └──────┘   own Blender subprocess
                          GPU 0   GPU 0    GPU 7
```

The proxy is **session-sticky**, not plain round-robin: Gradio's queue/SSE model
requires all requests carrying a given `session_hash` to land on the same worker,
otherwise renders break partway through. Different clients hash to different
workers, so load still spreads.

`deploy.sh` waits for workers to answer and reports how many came up, so a
partial failure shows up now instead of at the first render:

```
[INFO]  starting 64 workers = 8 GPU(s) x 8/GPU
[INFO]  worker ports: 7000..7063
[INFO]  workers up: 64/64
[INFO]  starting session-sticky proxy on :8080
[INFO]  service ready: http://0.0.0.0:8080  (64 workers behind a sticky proxy)
```

Sizing guidance:

| Scenario | Setting | Total workers (8-GPU node) |
|---|---|---|
| Install check | `WORKERS=1` | 1 (foreground, serial) |
| Interactive / small eval | `WORKERS_PER_GPU=2` | 16 |
| **Sampling & RL training** | `WORKERS_PER_GPU=8` | **64** |

Keep renderer capacity at or above the concurrency the trainer asks for
(`pcg_max_concurrency`, default 16 in `verl/run_map_gen_grpo*.sh`), otherwise
rollouts queue up here and the training GPUs idle waiting on screenshots.

Stop everything this script started (workers + proxy):

```bash
bash deploy.sh --stop
```

Per-worker logs are in `logs/worker_<port>.log`, the proxy's in `logs/proxy.log`.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `BLENDER_EXE` | `/opt/blender-4.2.0-linux-x64/blender` | Blender binary |
| `PORT` | `8080` | Port clients connect to (proxy port in multi-worker mode) |
| `WORKERS_PER_GPU` | `1` | Workers started **per GPU** — set to `8` for RL training |
| `WORKERS` | *(unset)* | Override the total directly; `WORKERS=1` = single foreground worker |
| `BASE_PORT` | `7000` | First worker port in multi-worker mode |
| `VIBEWORLD_MODELS_DIR` | `assets/models/clone` | GLB asset root |
| `PYTHON` | `python3` | Interpreter used for workers and proxy |

GPU count is detected with `nvidia-smi`; without a GPU it falls back to one
"device" and Blender renders on CPU (much slower, but functional).

## 5. Test it

Render the bundled example scene:

```bash
python render_raw_data_images.py \
  --raw_data_dir ./render_in_blender/assets/cmds/ \
  --server http://localhost:8080 \
  --quality "低质量 (快速预览)"
```

This writes 5-view images into `assets/cmds/1/image/`. (That directory already
ships with reference renders — delete it first if you want to prove the service
regenerated them.)

Single scene via the CLI client:

```bash
python call_render.py --json assets/cmds/1/pcg_scene.json \
  --server http://localhost:8080 \
  --quality "低质量 (快速预览)" --output render.png
```

You can also just open `http://localhost:8080` in a browser for the Gradio UI.

## 6. Quality levels

The `quality` argument is passed through verbatim and selects the Cycles sample
count. Use the low setting for RL rollouts, where throughput dominates.

| Value | Cycles samples | Typical time |
|---|---|---|
| `低质量 (快速预览)` | 8 | ~1 min |
| `中质量 (默认)` | 32 | ~3 min |
| `高质量` | 128 | ~10 min |

## 7. Camera presets

`render_raw_data_images.py` computes 5 canonical views (front / back / left /
right / top-down) from the scene bounds, which is exactly the observation format
the agent is trained on. `call_render.py` also exposes manual control:

| Preset | Meaning |
|---|---|
| `自动 (根据场景计算)` | Fit the camera to scene bounds |
| `等轴测 (斜45°俯视)` | Isometric |
| `正面 (从前方平视)` | Front elevation |
| `俯视 (正上方朝下)` | Top-down |
| `自定义` | Explicit `--cam-pos` / `--cam-target` |

## 8. Wiring it into the agent

```bash
export VIBEWORLD_RENDER_SERVER=http://localhost:8080
```

## 9. Files

| File | Role |
|---|---|
| `gradio_app.py` | Gradio server; the HTTP entry point |
| `render_scene.py` | Runs **inside** Blender; imports GLBs, lights, renders |
| `call_render.py` | CLI client for a single scene |
| `session_proxy.py` | Session-sticky reverse proxy for multi-worker setups |
| `deploy.sh` | Launcher (single or multi worker) |
